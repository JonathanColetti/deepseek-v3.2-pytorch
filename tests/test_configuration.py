# coding=utf-8
"""Tests for DeepSeekV32Config."""

import json
import os

from deepseek_v3_2 import DeepSeekV32Config


def test_default_config_values():
    """Default config reflects the 671B paper hyperparameters."""
    cfg = DeepSeekV32Config()
    assert cfg.hidden_size == 7168
    assert cfg.num_hidden_layers == 61
    assert cfg.num_attention_heads == 128
    assert cfg.kv_lora_rank == 512
    assert cfg.q_lora_rank == 1536
    assert cfg.use_dsa is True
    assert cfg.dsa_top_k == 2048
    assert cfg.dsa_n_indexer_heads == 64
    assert cfg.dsa_indexer_dim == 128


def test_q_head_dim_derived():
    """q_head_dim is the sum of nope and rope head dims."""
    cfg = DeepSeekV32Config(qk_nope_head_dim=128, qk_rope_head_dim=64)
    assert cfg.q_head_dim == 192


def test_dsa_active_for_layer_range():
    """dsa_active_for_layer respects the start/end layer window."""
    cfg = DeepSeekV32Config(num_hidden_layers=10, dsa_start_layer=2, dsa_end_layer=5)
    assert not cfg.dsa_active_for_layer(1)
    assert cfg.dsa_active_for_layer(2)
    assert cfg.dsa_active_for_layer(4)
    assert not cfg.dsa_active_for_layer(5)


def test_dsa_disabled_globally():
    """use_dsa=False disables DSA for every layer."""
    cfg = DeepSeekV32Config(use_dsa=False)
    assert not cfg.dsa_active_for_layer(0)


def test_roundtrip_serialization():
    """Config survives a to_dict/from_dict round-trip preserving DSA fields."""
    cfg = DeepSeekV32Config(dsa_top_k=777, dsa_n_indexer_heads=3)
    d = cfg.to_dict()
    cfg2 = DeepSeekV32Config.from_dict(d)
    assert cfg2.dsa_top_k == 777
    assert cfg2.dsa_n_indexer_heads == 3


def test_shipped_json_configs_load():
    """All shipped JSON configs parse into a valid DeepSeekV32Config."""
    root = os.path.join(os.path.dirname(__file__), "..", "configs")
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if fname.endswith(".json"):
                with open(os.path.join(dirpath, fname)) as f:
                    data = json.load(f)
                cfg = DeepSeekV32Config(**data)
                assert cfg.hidden_size > 0
