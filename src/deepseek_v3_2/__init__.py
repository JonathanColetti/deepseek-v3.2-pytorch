# coding=utf-8
"""DeepSeek V3.2 with DeepSeek Sparse Attention (DSA)."""

from .config import DeepSeekV32Config
from .indexer import DSAAttentionOutput, LightningIndexer
from .model import (
    DeepSeekV32Attention,
    DeepSeekV32DecoderLayer,
    DeepSeekV32ForCausalLM,
    DeepSeekV32MLP,
    DeepSeekV32Model,
    DeepSeekV32MoE,
    DeepSeekV32PreTrainedModel,
    DeepSeekV32RMSNorm,
    DeepSeekV32RotaryEmbedding,
    DeepSeekV32TopkRouter,
)

__version__ = "0.1.0"

__all__ = [
    "DeepSeekV32Config",
    "LightningIndexer",
    "DSAAttentionOutput",
    "DeepSeekV32Attention",
    "DeepSeekV32DecoderLayer",
    "DeepSeekV32ForCausalLM",
    "DeepSeekV32MLP",
    "DeepSeekV32Model",
    "DeepSeekV32MoE",
    "DeepSeekV32PreTrainedModel",
    "DeepSeekV32RMSNorm",
    "DeepSeekV32RotaryEmbedding",
    "DeepSeekV32TopkRouter",
]
