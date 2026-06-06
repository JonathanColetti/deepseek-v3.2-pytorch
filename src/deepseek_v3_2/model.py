"""PyTorch DeepSeek V3.2 model (MLA + MoE + DeepSeek Sparse Attention). taken mostly from here (huggingface) so shoutout to them: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modular_deepseek_v3.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .config import DeepSeekV32Config
from .indexer import (
    LightningIndexer,
    compute_rope_inv_freq,
    rotate_half,
    yarn_get_mscale,
)


logger = logging.get_logger(__name__)


class DeepSeekV32RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepSeekV32RotaryEmbedding(nn.Module):
    """Rotary embedding supporting optional YaRN extension scaling.

    Uses the same compute_rope_inv_freq helper as the Lightning Indexer so the two
    share identical frequencies. The YaRN mscale is applied here to cos/sin (an attention softmax temperature trick) the indexer deliberately omits it.
    """

    def __init__(self, config: DeepSeekV32Config, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        inv_freq, mscale = compute_rope_inv_freq(
            config.qk_rope_head_dim,
            config.rope_theta,
            config.rope_scaling,
            config.max_position_embeddings,
        )
        self.mscale = mscale
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = self.inv_freq.to(x.device)
        position_ids = position_ids.to(x.device)
        freqs = position_ids[:, :, None].float() * inv_freq[None, None, :]  # (B, S, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.mscale).to(x.dtype)
        sin = (emb.sin() * self.mscale).to(x.dtype)
        return cos, sin


def apply_rotary_pos_emb_interleave(q, k, cos, sin, unsqueeze_dim=1):
    """Interleaved RoPE used by DeepSeek V3.

    q, k: (B, H, S, qk_rope_head_dim); cos/sin: (B, S, qk_rope_head_dim).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSeekV32MLP(nn.Module):
    """SwiGLU MLP (Qwen2-style)."""

    def __init__(self, config: DeepSeekV32Config, hidden_size: Optional[int] = None, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))



class DeepSeekV32TopkRouter(nn.Module):
    """Sigmoid gated, group limited top k router with an FP32 e-score correction bias."""

    def __init__(self, config: DeepSeekV32Config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, config.hidden_size))
        # E-score correction bias is kept in FP32 even under bf16 training.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
            persistent=True,
        )

    @torch.no_grad()
    def _group_limited_topk(self, scores: torch.Tensor) -> torch.Tensor:
        """Return a (n_tokens, top_k) index tensor using group-limited selection."""
        n_tokens = scores.shape[0]
        group_scores = (
            scores.view(n_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )  # (n_tokens, n_group)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(n_tokens, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(n_tokens, -1)
        )
        masked_scores = scores.masked_fill(score_mask == 0, float("-inf"))
        _, topk_idx = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)
        return topk_idx

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n_tokens = hidden_states.shape[0]
        logits = F.linear(hidden_states.float(), self.weight.float())  # (n_tokens, n_experts)
        scores = logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_idx = self._group_limited_topk(scores_for_choice)  # (n_tokens, top_k)

        topk_weight = scores.gather(1, topk_idx)
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight.to(hidden_states.dtype)


class DeepSeekV32MoE(nn.Module):
    """Mixture-of-Experts: routed experts + an always-on shared-expert branch."""

    def __init__(self, config: DeepSeekV32Config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                DeepSeekV32MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepSeekV32TopkRouter(config)
        if config.n_shared_experts > 0:
            self.shared_experts = DeepSeekV32MLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        N = B * S
        flat = hidden_states.view(N, H)
        K = self.num_experts_per_tok
        E = len(self.experts)

        topk_idx, topk_weight = self.gate(flat)  # (N, K), (N, K)

        # Vectorised sorted dispatch: a single argsort replaces E separate torch.where calls,
        # eliminating the per-expert CPU-GPU synchronisation bottleneck.
        expert_ids = topk_idx.reshape(-1)            # (N*K,)
        token_ids  = torch.arange(N, device=flat.device).unsqueeze(1).expand(N, K).reshape(-1)  # (N*K,)
        slot_ids   = torch.arange(K, device=flat.device).unsqueeze(0).expand(N, K).reshape(-1)  # (N*K,)

        # Sort all (token, expert) pairs by expert id once.
        perm            = expert_ids.argsort(stable=True)   # (N*K,)
        s_expert        = expert_ids[perm]                  # (N*K,) sorted
        s_token         = token_ids[perm]                   # (N*K,)
        s_slot          = slot_ids[perm]                    # (N*K,)
        expert_counts   = s_expert.bincount(minlength=E)    # (E,) one CPU sync

        out = torch.zeros_like(flat)
        offset = 0
        for eid, cnt in enumerate(expert_counts.tolist()):  # .tolist(): single CPU sync
            if cnt == 0:
                offset += cnt
                continue
            sl = slice(offset, offset + cnt)
            tok = s_token[sl]
            wgt = topk_weight[tok, s_slot[sl]].unsqueeze(-1)
            out.index_add_(0, tok, self.experts[eid](flat[tok]) * wgt)
            offset += cnt

        out = out.view(B, S, H)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out


class DeepSeekV32Attention(nn.Module):
    """Multi-head Latent Attention with optional DeepSeek Sparse Attention."""

    def __init__(self, config: DeepSeekV32Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.attention_dropout = config.attention_dropout
        self.rope_interleave = config.rope_interleave

        self.use_dsa = config.dsa_active_for_layer(layer_idx)
        self.dsa_top_k = config.dsa_top_k

        # Query proj (optionally low rank).
        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepSeekV32RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)

        # KV joint compression with a separate RoPE key (MQA on the RoPE part)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = DeepSeekV32RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)

        # Softmax scale (matches DeepSeek V3, with optional YaRN mscale adjustment)
        self.softmax_scale = self.q_head_dim ** -0.5
        if config.rope_scaling is not None:
            scaling = config.rope_scaling
            if scaling.get("type", scaling.get("rope_type")) == "yarn":
                mscale = yarn_get_mscale(scaling["factor"], scaling.get("mscale_all_dim", 1.0))
                self.softmax_scale = self.softmax_scale * mscale * mscale

        if self.use_dsa:
            self.indexer = LightningIndexer(
                hidden_size=self.hidden_size,
                n_indexer_heads=config.dsa_n_indexer_heads,
                indexer_dim=config.dsa_indexer_dim,
                rope_head_dim=config.qk_rope_head_dim,
                rope_theta=config.rope_theta,
                rope_scaling=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
            )
        else:
            self.indexer = None

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is not None:
            return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        return self.q_proj(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        dsa_mode: str = "sparse",
        return_index_scores: bool = False,
    ):
        B, S_q, _ = hidden_states.shape

        # Query
        q = self._project_q(hidden_states)
        q = q.view(B, S_q, self.num_heads, self.q_head_dim).transpose(1, 2)  # (B, H, S_q, q_head_dim)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV latent
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # (B, S_q, kv_lora_rank + rope)
        kv_latent, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_latent = self.kv_a_layernorm(kv_latent)  # c^KV normed, (B, S_q, kv_lora_rank)
        k_rope = k_rope.view(B, S_q, 1, self.qk_rope_head_dim).transpose(1, 2)  # (B, 1, S_q, rope)

        cos, sin = position_embeddings
        if self.rope_interleave:
            q_rope, k_rope = apply_rotary_pos_emb_interleave(q_rope, k_rope, cos, sin)
        else:
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)

        # Lightning Indexer key (paper Eq. 1: k^I projected from the hidden state h_s)
        # The indexer input is DETACHED from the autograd graph: per the paper, the indexer
        # is optimised only by the KL alignment loss, while the main model is optimised
        # only by the language-modeling loss. Detaching here isolates the two signals.
        k_index = None
        if self.use_dsa:
            k_index = self.indexer.project_key(hidden_states.detach(), position_ids)  # (B, S_q, D)

        # KV cache: store the compressed latent + rope key + indexer key (not expanded K/V). ---
        if past_key_value is not None:
            past_latent, past_krope, past_kindex = past_key_value
            kv_latent = torch.cat([past_latent, kv_latent], dim=1)
            k_rope = torch.cat([past_krope, k_rope], dim=2)
            if self.use_dsa and past_kindex is not None:
                k_index = torch.cat([past_kindex, k_index], dim=1)
        present = (kv_latent, k_rope, k_index) if use_cache else None

        S_kv = kv_latent.shape[1]

        # Expand KV from the (full) latent
        kv = self.kv_b_proj(kv_latent)  # (B, S_kv, H*(nope+v))
        kv = kv.view(B, S_kv, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Assemble query/key heads: [nope | rope].
        query_states = torch.cat([q_nope, q_rope], dim=-1)  # (B, H, S_q, q_head_dim)
        k_rope_expanded = k_rope.expand(B, self.num_heads, S_kv, self.qk_rope_head_dim)
        key_states = torch.cat([k_nope, k_rope_expanded], dim=-1)  # (B, H, S_kv, q_head_dim)

        # Causal mask (additive)
        causal_mask = self._build_causal_mask(attention_mask, S_q, S_kv, hidden_states.dtype, hidden_states.device)

        # DSA selection 
        index_scores = None
        selected_mask = None
        do_sparse = self.use_dsa and dsa_mode == "sparse" and S_kv > self.dsa_top_k
        if self.use_dsa and (return_index_scores or do_sparse):
            # Indexer input detached (see note above): KL grad updates only the indexer.
            index_scores = self.indexer(
                hidden_states.detach(), position_ids, causal_mask=causal_mask, k_index=k_index
            )

        if do_sparse:
            _, selected_mask = self.indexer.select_top_k(index_scores, self.dsa_top_k)
            # Build additive sparse mask: keep selected & causal, mask the rest.
            sparse_add = torch.where(
                selected_mask.unsqueeze(1),
                torch.zeros((), dtype=hidden_states.dtype, device=hidden_states.device),
                torch.finfo(hidden_states.dtype).min,
            )
            attn_bias = causal_mask + sparse_add
        else:
            attn_bias = causal_mask

        # Scaled dot-product attention (manual, returns weights for KL loss) ---
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)  # (B, H, S_q, v_head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_q, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return {
            "attn_output": attn_output,
            "attn_weights": attn_weights if (output_attentions or return_index_scores) else None,
            "index_scores": index_scores,
            "selected_mask": selected_mask,
            "past_key_value": present,
        }

    @staticmethod
    def _build_causal_mask(attention_mask, S_q, S_kv, dtype, device):
        min_val = torch.finfo(dtype).min
        # Allow each query position t (offset by cached prefix) to attend up to itself.
        offset = S_kv - S_q
        q_idx = torch.arange(S_q, device=device).view(S_q, 1) + offset
        k_idx = torch.arange(S_kv, device=device).view(1, S_kv)
        causal = (k_idx <= q_idx)  # (S_q, S_kv) True where allowed
        mask = torch.where(causal, 0.0, min_val).to(dtype)
        mask = mask.view(1, 1, S_q, S_kv)
        if attention_mask is not None:
            # attention_mask: (B, S_kv) 1 = keep, 0 = pad
            pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_val
            mask = mask + pad
        return mask



class DeepSeekV32DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV32Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepSeekV32Attention(config, layer_idx)

        is_moe = (
            layer_idx >= config.first_k_dense_replace
            and (layer_idx - config.first_k_dense_replace) % config.moe_layer_freq == 0
        )
        self.mlp = DeepSeekV32MoE(config) if is_moe else DeepSeekV32MLP(config)
        self.is_moe = is_moe

        self.input_layernorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        dsa_mode: str = "sparse",
        return_index_scores: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            dsa_mode=dsa_mode,
            return_index_scores=return_index_scores,
        )
        hidden_states = residual + attn["attn_output"]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return {
            "hidden_states": hidden_states,
            "attn_weights": attn["attn_weights"],
            "index_scores": attn["index_scores"],
            "selected_mask": attn["selected_mask"],
            "past_key_value": attn["past_key_value"],
        }



class DeepSeekV32PreTrainedModel(PreTrainedModel):
    config_class = DeepSeekV32Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepSeekV32DecoderLayer"]
    _supports_cache_class = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, DeepSeekV32RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, DeepSeekV32TopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)



@dataclass
class DeepSeekV32ModelOutput(BaseModelOutputWithPast):
    dsa_kl_losses: Optional[List[torch.Tensor]] = None


class DeepSeekV32Model(DeepSeekV32PreTrainedModel):
    def __init__(self, config: DeepSeekV32Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepSeekV32DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = DeepSeekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepSeekV32RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dsa_mode: str = "sparse",
        collect_dsa_losses: bool = False,
    ) -> DeepSeekV32ModelOutput:
        output_attentions = output_attentions if output_attentions is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B, S = inputs_embeds.shape[:2]

        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].shape[1]

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + S, device=inputs_embeds.device).unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        return_index_scores = collect_dsa_losses
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = [] if use_cache else None
        dsa_kl_losses = [] if collect_dsa_losses else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_past = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                out = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    position_ids,
                    position_embeddings,
                    attention_mask,
                    layer_past,
                    output_attentions,
                    use_cache,
                    dsa_mode,
                    return_index_scores,
                )
            else:
                out = layer(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    dsa_mode=dsa_mode,
                    return_index_scores=return_index_scores,
                )

            hidden_states = out["hidden_states"]
            if use_cache:
                next_cache.append(out["past_key_value"])
            if output_attentions:
                all_attentions += (out["attn_weights"],)
            if collect_dsa_losses and layer.self_attn.use_dsa and out["index_scores"] is not None:
                kl = layer.self_attn.indexer.compute_kl_loss(
                    out["index_scores"], out["attn_weights"], out["selected_mask"]
                )
                dsa_kl_losses.append(kl)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return DeepSeekV32ModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            dsa_kl_losses=dsa_kl_losses,
        )



@dataclass
class DeepSeekV32CausalLMOutput(CausalLMOutputWithPast):
    dsa_kl_loss: Optional[torch.Tensor] = None


class DeepSeekV32ForCausalLM(DeepSeekV32PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DeepSeekV32Config):
        super().__init__(config)
        self.model = DeepSeekV32Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dsa_mode: str = "sparse",
        collect_dsa_losses: bool = False,
    ) -> DeepSeekV32CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            dsa_mode=dsa_mode,
            collect_dsa_losses=collect_dsa_losses,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1).to(shift_logits.device),
                ignore_index=-100,
            )

        dsa_kl_loss = None
        if outputs.dsa_kl_losses:
            dsa_kl_loss = torch.stack(outputs.dsa_kl_losses).mean()

        return DeepSeekV32CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            dsa_kl_loss=dsa_kl_loss,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
            "dsa_mode": kwargs.get("dsa_mode", "sparse"),
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int = 20, dsa_mode: str = "sparse", **kwargs):
        """Minimal greedy generation with the compressed-latent KV cache."""
        past = None
        generated = input_ids
        cur = input_ids
        attention_mask = kwargs.get("attention_mask", None)
        for _ in range(max_new_tokens):
            out = self.forward(
                input_ids=cur,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                dsa_mode=dsa_mode,
            )
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            cur = next_token
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=1
                )
        return generated
