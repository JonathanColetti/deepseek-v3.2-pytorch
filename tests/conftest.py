
"""Shared pytest fixtures: a tiny but fully functional DeepSeek V3.2 model."""

import pytest
import torch

from deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForCausalLM


@pytest.fixture
def tiny_config():
    """Minimal config for fast unit tests."""
    return DeepSeekV32Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=64,
        first_k_dense_replace=1,
        use_dsa=True,
        dsa_top_k=16,
        dsa_n_indexer_heads=2,
        dsa_indexer_dim=8,
    )


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    return DeepSeekV32ForCausalLM(tiny_config).eval()


@pytest.fixture
def sample_inputs():
    torch.manual_seed(1)
    return {
        "input_ids": torch.randint(0, 1000, (2, 32)),
        "attention_mask": torch.ones(2, 32, dtype=torch.long),
    }
