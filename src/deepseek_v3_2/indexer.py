# coding=utf-8
"""DeepSeek Sparse Attention: the Lightning Indexer plus the rotary helpers it
shares with the main model.

The rotary helpers live here (the leaf module) so that both the Lightning Indexer
and the main model's rotary embedding compute the exact same YaRN frequencies. This
guarantees the indexer is position-aware in the same way the attention it imitates
is, which is the whole point of the indexer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def compute_rope_inv_freq(
    dim: int,
    base: float,
    rope_scaling: Optional[dict],
    max_position_embeddings: int,
) -> Tuple[torch.Tensor, float]:
    """Return (inv_freq, mscale) for default or YaRN rotary embeddings.

    Both the main attention rotary and the Lightning Indexer call this with the same
    ``rope_scaling`` so their frequencies match exactly. ``mscale`` is the YaRN
    attention temperature; it is applied to the main attention only (it scales the
    softmax, not the rotary vectors), so the indexer ignores the returned value.
    """
    is_yarn = (
        rope_scaling is not None
        and rope_scaling.get("type", rope_scaling.get("rope_type")) == "yarn"
    )
    if is_yarn:
        factor = rope_scaling["factor"]
        orig_max = rope_scaling.get("original_max_position_embeddings", max_position_embeddings)
        beta_fast = rope_scaling.get("beta_fast", 32)
        beta_slow = rope_scaling.get("beta_slow", 1)
        mscale_arg = rope_scaling.get("mscale", 1.0)

        freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        low, high = yarn_find_correction_range(beta_fast, beta_slow, dim, base, orig_max)
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        mscale = yarn_get_mscale(factor, mscale_arg)
    else:
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        mscale = 1.0
    return inv_freq, mscale


@dataclass
class DSAAttentionOutput:
    """Container for the output of a DSA attention layer."""

    attn_output: torch.Tensor
    attn_weights: Optional[torch.Tensor] = None  # (B, H, S_q, S_kv) for the KL loss
    index_scores: Optional[torch.Tensor] = None  # (B, S_q, S_kv) raw indexer scores
    selected_mask: Optional[torch.Tensor] = None  # (B, S_q, S_kv) boolean top-k mask
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None


class LightningIndexer(nn.Module):
    """
    Lightning Indexer (DeepSeek V3.2, Eq. 1).

    Computes a cheap per (query, key) relevance score:

        I_{t,s} = sum_{j=1}^{H_I} w^I_{t,j} * ReLU(q^I_{t,j} * k^I_s)

    where:::

        - q^I_{t,j} in R^{D_I} are H_I indexer query heads projected from the query
          token's hidden state h_t,
        - k^I_s in R^{D_I} is a single (MQA) indexer key projected from the hidden
          state h_s, shared across all indexer heads,
        - w^I_{t,j} is a per head scalar weight, the raw output of a linear projection
          scaled by H_I^{-1/2} (matching the official implementation, not a softmax).

    Rotary position embeddings make the indexer position aware. Following the official
    DeepSeek V3.2 inference code, the indexer RoPE is faithful in three ways that a
    naive port gets wrong:

        1. YaRN: the indexer reuses the same YaRN adjusted frequencies as the main
           attention (via compute_rope_inv_freq), so its sense of position matches the
           attention it imitates, including in the long context regime.
        2. Non interleaved layout: the indexer always uses the half split (GPT NeoX)
           rotary layout, even when the main MLA attention uses the interleaved layout.
           This matches the official code (apply_rotary_emb(..., interleaved=False)).
        3. rope / nope split: only the first rope_head_dim (= qk_rope_head_dim) channels
           of each indexer head carry rotary position; the remaining channels are left
           unrotated, exactly as the official 128 dim head splits into 64 rope + 64 nope.

    The YaRN mscale is an attention softmax temperature, not part of the rotary vectors,
    so it is deliberately not applied here (the indexer has its own head_dim**-0.5 scale).

    Notes vs the 671B paper model (best at reduced scale):
        * The paper uses H_I=64, D_I=128 and FP8 kernels; this is a pure PyTorch,
          configurable width re implementation in fp32 for stability.
    """

    def __init__(
        self,
        hidden_size: int,
        n_indexer_heads: int = 64,
        indexer_dim: int = 128,
        rope_head_dim: int = 64,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 4096,
    ):
        super().__init__()
        if rope_head_dim > indexer_dim:
            raise ValueError(
                f"rope_head_dim ({rope_head_dim}) must be <= indexer_dim ({indexer_dim})"
            )
        if rope_head_dim % 2 != 0:
            raise ValueError(f"rope_head_dim must be even for rotary embeddings got {rope_head_dim}")
        self.hidden_size = hidden_size
        self.n_indexer_heads = n_indexer_heads
        self.indexer_dim = indexer_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = indexer_dim - rope_head_dim
        self.scale = indexer_dim ** -0.5
        self.head_weight_scale = n_indexer_heads ** -0.5

        # q^I: H_I heads; k^I: a single shared (MQA) key projected from the hidden state.
        self.q_proj = nn.Linear(hidden_size, n_indexer_heads * indexer_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, indexer_dim, bias=False)
        self.w_proj = nn.Linear(hidden_size, n_indexer_heads, bias=False)

        # Same YaRN frequencies as the main attention. mscale is unused here on purpose.
        inv_freq, _ = compute_rope_inv_freq(
            rope_head_dim, rope_theta, rope_scaling, max_position_embeddings
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotary(self, position_ids: torch.Tensor, device, dtype):
        inv_freq = self.inv_freq.to(device)
        position_ids = position_ids.to(device)
        freqs = position_ids[:, :, None].float() * inv_freq[None, None, :]  # (B, S, rope/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (B, S, rope)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Non interleaved (half split) RoPE on the rope channels only.

        x: (B, H, S, indexer_dim); cos/sin: (B, S, rope_head_dim).
        """
        if self.nope_head_dim == 0:
            x_pe, x_nope = x, None
        else:
            x_pe, x_nope = torch.split(x, [self.rope_head_dim, self.nope_head_dim], dim=-1)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x_pe = (x_pe * cos) + (rotate_half(x_pe) * sin)
        if x_nope is None:
            return x_pe
        return torch.cat([x_pe, x_nope], dim=-1)

    def project_key(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Project and RoPE the indexer key from the hidden state h_s. Returns (B, S, D).

        Computed separately from scoring so it can be cached across decoding steps
        (the raw hidden states are not retained in the KV cache).
        """
        B, S, _ = hidden_states.shape
        k = self.k_proj(hidden_states).view(B, 1, S, self.indexer_dim)  # (B, 1, S, D)
        cos, sin = self._rotary(position_ids, hidden_states.device, k.dtype)
        k = self._apply_rope(k, cos, sin)
        return k.squeeze(1)  # (B, S, D)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, S_q, hidden_size) query tokens
        position_ids: torch.Tensor,  # (B, S_q)
        causal_mask: Optional[torch.Tensor] = None,  # (B, 1, S_q, S_kv) additive
        k_index: Optional[torch.Tensor] = None,  # (B, S_kv, D) precomputed/cached keys
    ) -> torch.Tensor:  # (B, S_q, S_kv)
        B, S_q, _ = hidden_states.shape
        H, D = self.n_indexer_heads, self.indexer_dim

        if k_index is None:
            k_index = self.project_key(hidden_states, position_ids)

        q = self.q_proj(hidden_states).view(B, S_q, H, D).transpose(1, 2)  # (B, H, S_q, D)
        cos, sin = self._rotary(position_ids, hidden_states.device, q.dtype)
        q = self._apply_rope(q, cos, sin)  # (B, H, S_q, D)

        # Scoring is promoted to fp32 whiichh matches transformers `DeepseekV4IndexerScorer` for
        # bf16/fp16 stability. per head ReLU(q.k) scaled then a raw H_I^{-1/2} scaled
        # head weighting (no softmax which matches the reference implementation).
        scores = torch.einsum("bhqd,bsd->bqhs", q.float(), k_index.float())  
        scores = F.relu(scores) * self.scale
        w = self.w_proj(hidden_states).float() * self.head_weight_scale

        # Weighted sum over heads -> (B, S_q, S_kv)
        index_scores = (w.unsqueeze(-1) * scores).sum(dim=2)

        if causal_mask is not None:
            index_scores = index_scores + causal_mask.squeeze(1).float()

        return index_scores

    @torch.no_grad()
    def select_top_k(
        self,
        index_scores: torch.Tensor,  # (B, S_q, S_kv)
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the top-k KV positions per query.

        Returns:
            indices: (B, S_q, k) long tensor of selected key positions (unsorted).
            mask:    (B, S_q, S_kv) boolean tensor; True where selected.
        """
        B, S_q, S_kv = index_scores.shape
        k = min(top_k, S_kv)
        _, indices = torch.topk(index_scores, k, dim=-1, sorted=False)
        mask = torch.zeros(B, S_q, S_kv, dtype=torch.bool, device=index_scores.device)
        mask.scatter_(-1, indices, True)
        return indices, mask

    def compute_kl_loss(
        self,
        index_scores: torch.Tensor,  # (B, S_q, S_kv) raw indexer scores
        main_attn_weights: torch.Tensor,  # (B, H, S_q, S_kv) main attention probs
        selected_mask: Optional[torch.Tensor] = None,  # (B, S_q, S_kv) sparse-stage mask
    ) -> torch.Tensor:
        """KL alignment loss between the indexer distribution and the (detached) main
        attention distribution aggregated over heads (DeepSeek V3.2 Eq. 2 and Eq. 3).

        The target distribution p is the L1 normalised sum of attention weights over
        heads. The indexer distribution is q = softmax(I_{t,.}). We minimise KL(p || q).

        All computation is promoted to float32 to avoid 0 * (-inf) = NaN artefacts that
        arise when the model runs in bf16/fp16 and causal positions carry -inf scores.
        """
        # Always work in float32 to dodge 0 * (-inf) = NaN in bf16/fp16.
        scores_f = index_scores.float()
        weights_f = main_attn_weights.float()

        # Aggregate over attention heads then L1 normalise => (B, S_q, S_kv)
        p_agg = weights_f.sum(dim=1)
        p_agg = p_agg / (p_agg.sum(dim=-1, keepdim=True).clamp(min=1e-9))
        # clamp away any tiny negatives from bf16 round trip before computing log.
        p_agg = p_agg.clamp(min=0.0)

        B, S_q, S_kv = scores_f.shape

        if selected_mask is not None:
            # Sparse stage (Eq. 3): restrict alignment to the top k selected positions.
            scores_masked = scores_f.masked_fill(~selected_mask, float("-inf"))
            p_masked = p_agg.masked_fill(~selected_mask, 0.0)
            p_masked = p_masked / (p_masked.sum(dim=-1, keepdim=True).clamp(min=1e-9))
            log_q = F.log_softmax(scores_masked, dim=-1)
            log_q = torch.nan_to_num(log_q, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Dense warmup stage (Eq. 2): full KL over all positions.
            p_masked = p_agg
            log_q = F.log_softmax(scores_f, dim=-1)
            log_q = torch.nan_to_num(log_q, nan=0.0, posinf=0.0, neginf=0.0)


        return F.kl_div(
            log_q.reshape(B * S_q, S_kv),
            p_masked.detach().reshape(B * S_q, S_kv),
            reduction="batchmean",
            log_target=False,
        )

    def num_parameters(self) -> int:
        """Analytic parameter count (matches the test in tests/test_dsa_indexer.py)"""
        return (
            self.n_indexer_heads * self.indexer_dim * self.hidden_size  # q_proj
            + self.indexer_dim * self.hidden_size  # k_proj (from hidden state h_s)
            + self.n_indexer_heads * self.hidden_size  # w_proj
        )
