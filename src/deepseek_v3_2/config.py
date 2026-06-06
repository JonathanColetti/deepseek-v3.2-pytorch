"""DeepSeek V3.2 model configuration."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class DeepSeekV32Config(PretrainedConfig):
    r"""
    Configuration class for [`DeepSeekV32Model`]. It is used to instantiate a DeepSeek V3.2 model
    according to the specified arguments, defining the model architecture.

    DeepSeek V3.2 augments DeepSeek V3 with **DSA (DeepSeek Sparse Attention)**: a Lightning Indexer
    selects a small top-k subset of key/value positions per query token, which the Multi-head Latent
    Attention (MLA) then attends to. This yields near-linear attention cost at long context lengths.

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the (dense) MLP intermediate representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of each MoE expert's intermediate representation.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention (query) heads.
        num_key_value_heads (`int`, *optional*, defaults to 128):
            Number of key/value heads (kept for API parity; MLA uses a shared latent).
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of always-active shared experts.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor applied to routed-expert outputs.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the KV LoRA / compressed latent dimension.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the query LoRA compression. If `None`, the query projection is not low-rank.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Per-head dimension carrying rotary position information.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Per-head dimension *without* rotary position information.
        v_head_dim (`int`, *optional*, defaults to 128):
            Per-head value dimension.
        n_group (`int`, *optional*, defaults to 8):
            Number of expert groups for grouped top-k routing.
        topk_group (`int`, *optional*, defaults to 4):
            Number of groups selected per token.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of routed experts selected per token.
        first_k_dense_replace (`int`, *optional*, defaults to 3):
            Number of leading layers that use a dense MLP instead of MoE.
        moe_layer_freq (`int`, *optional*, defaults to 1):
            Frequency of MoE layers (1 = every layer after `first_k_dense_replace`).
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the top-k routing weights.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            Routing score function. Only `"sigmoid"` is supported.
        aux_loss_alpha (`float`, *optional*, defaults to 0.001):
            Auxiliary load-balancing loss coefficient.
        seq_aux (`bool`, *optional*, defaults to `True`):
            Whether to compute the auxiliary loss at the sequence level.
        max_position_embeddings (`int`, *optional*, defaults to 163840):
            Maximum sequence length the model supports.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated-normal initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMSNorm.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return / accept past key-values.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of the rotary embeddings.
        rope_scaling (`dict`, *optional*):
            RoPE scaling configuration (e.g. YaRN).
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether RoPE is applied in interleaved layout.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether attention projections use a bias.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Attention dropout probability.
        use_dsa (`bool`, *optional*, defaults to `True`):
            Whether to enable DeepSeek Sparse Attention.
        dsa_top_k (`int`, *optional*, defaults to 2048):
            Number of KV positions selected per query token by the Lightning Indexer.
        dsa_n_indexer_heads (`int`, *optional*, defaults to 64):
            Number of Lightning Indexer heads (H_I). The 671B paper model uses 64.
        dsa_indexer_dim (`int`, *optional*, defaults to 128):
            Per-head Lightning Indexer dimension (D_I). The 671B paper model uses 128.
        dsa_start_layer (`int`, *optional*, defaults to 0):
            First layer index (inclusive) at which DSA is active.
        dsa_end_layer (`int`, *optional*):
            Last layer index (exclusive) at which DSA is active. `None` = all layers.
        dsa_warmup_steps (`int`, *optional*, defaults to 1000):
            Number of dense indexer warm-up steps.
        dsa_sparse_steps (`int`, *optional*, defaults to 4000):
            Number of sparse-training steps.
        dsa_kl_loss_weight (`float`, *optional*, defaults to 0.1):
            Weight of the indexer KL alignment loss during sparse training.
    """

    model_type = "deepseek_v3_2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 61,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 128,
        n_shared_experts: int = 1,
        n_routed_experts: int = 256,
        routed_scaling_factor: float = 2.5,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 128,
        v_head_dim: int = 128,
        n_group: int = 8,
        topk_group: int = 4,
        num_experts_per_tok: int = 8,
        first_k_dense_replace: int = 3,
        moe_layer_freq: int = 1,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        aux_loss_alpha: float = 0.001,
        seq_aux: bool = True,
        max_position_embeddings: int = 163840,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        rope_interleave: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # DSA (DeepSeek V3.2)
        use_dsa: bool = True,
        dsa_top_k: int = 2048,
        dsa_n_indexer_heads: int = 64,
        dsa_indexer_dim: int = 128,
        dsa_start_layer: int = 0,
        dsa_end_layer: int | None = None,
        dsa_warmup_steps: int = 1000,
        dsa_sparse_steps: int = 15000,
        dsa_kl_loss_weight: float = 0.1,
        pad_token_id: int | None = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleave = rope_interleave
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # DSA
        self.use_dsa = use_dsa
        self.dsa_top_k = dsa_top_k
        self.dsa_n_indexer_heads = dsa_n_indexer_heads
        self.dsa_indexer_dim = dsa_indexer_dim
        self.dsa_start_layer = dsa_start_layer
        self.dsa_end_layer = dsa_end_layer
        self.dsa_warmup_steps = dsa_warmup_steps
        self.dsa_sparse_steps = dsa_sparse_steps
        self.dsa_kl_loss_weight = dsa_kl_loss_weight

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def dsa_active_for_layer(self, layer_idx: int) -> bool:
        """Whether DSA should be active for a given decoder layer index."""
        if not self.use_dsa:
            return False
        end = self.dsa_end_layer if self.dsa_end_layer is not None else self.num_hidden_layers
        return self.dsa_start_layer <= layer_idx < end
