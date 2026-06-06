# coding=utf-8
"""Distributed (FSDP) helpers for 2-GPU DeepSeek V3.2 training.

These utilities degrade gracefully on CPU-only machines so that the rest of the
package (and its unit tests) can be exercised without GPUs.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import torch.multiprocessing as mp


def init_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """Initialize the default process group and bind a CUDA device.

    Falls back to the ``gloo`` backend automatically when CUDA is unavailable.
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    if not torch.cuda.is_available() and backend == "nccl":
        backend = "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def get_fsdp_model(model, DecoderLayerClass, rank: int, use_activation_checkpointing: bool = True):
    """Wrap ``model`` with FSDP, sharding per-decoder-layer, in bf16 mixed precision.

    Args:
        model: the (un-sharded) module to wrap.
        DecoderLayerClass: the transformer block class to use as the wrap boundary.
        rank: local device rank.
        use_activation_checkpointing: whether to apply activation checkpointing to blocks.
    """


    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={DecoderLayerClass},
    )
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=wrap_policy,
        mixed_precision=bf16_policy,
        device_id=rank if torch.cuda.is_available() else None,
        use_orig_params=True,
    )
    if use_activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        check_fn = lambda m: isinstance(m, DecoderLayerClass)  # noqa: E731
        apply_activation_checkpointing(
            fsdp_model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
        )
    return fsdp_model


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def run_two_gpu_training(train_fn, config, dataset) -> None:
    """Launch ``train_fn(rank, world_size, config, dataset)`` across (up to) 2 GPUs."""

    world_size = min(2, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    if world_size <= 1:
        # Single process fallback (CPU or single GPU).
        train_fn(0, 1, config, dataset)
        return
    mp.spawn(train_fn, args=(world_size, config, dataset), nprocs=world_size, join=True)
