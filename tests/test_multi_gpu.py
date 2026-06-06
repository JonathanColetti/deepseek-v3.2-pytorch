# coding=utf-8
"""Multi-GPU / distributed tests. Skipped automatically when <2 GPUs are present."""

import pytest
import torch

from deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForCausalLM
from deepseek_v3_2.model import DeepSeekV32DecoderLayer
from deepseek_v3_2.training.distributed import run_two_gpu_training

requires_2_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires 2 GPUs",
)


def _train_fn(rank, world_size, config, dataset):
    from deepseek_v3_2.training.distributed import (
        cleanup_distributed,
        get_fsdp_model,
        init_distributed,
    )

    init_distributed(rank, world_size)
    model = DeepSeekV32ForCausalLM(config).cuda(rank)
    fsdp_model = get_fsdp_model(model, DeepSeekV32DecoderLayer, rank, use_activation_checkpointing=True)
    x = dataset["input_ids"].cuda(rank)
    out = fsdp_model(input_ids=x, labels=x)
    out.loss.backward()
    cleanup_distributed()


@requires_2_gpus
@pytest.mark.multi_gpu
def test_fsdp_two_gpu_training():
    """FSDP-wrapped model performs a forward/backward step across 2 GPUs."""
    cfg = DeepSeekV32Config(
        vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32, kv_lora_rank=16,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, n_routed_experts=8,
        num_experts_per_tok=2, n_group=2, topk_group=1, moe_intermediate_size=64,
        first_k_dense_replace=1, use_dsa=True, dsa_top_k=16, dsa_n_indexer_heads=2,
        dsa_indexer_dim=8, max_position_embeddings=512,
    )
    dataset = {"input_ids": torch.randint(0, 1000, (2, 32))}
    run_two_gpu_training(_train_fn, cfg, dataset)


def test_run_two_gpu_training_cpu_fallback():
    """run_two_gpu_training falls back to a single in-process call without GPUs."""
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        pytest.skip("GPUs present; covered by FSDP test")
    called = {}

    def fn(rank, world_size, config, dataset):
        called["rank"] = rank
        called["world_size"] = world_size

    run_two_gpu_training(fn, config=None, dataset=None)
    assert called == {"rank": 0, "world_size": 1}
