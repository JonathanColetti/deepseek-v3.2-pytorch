"""Tests for the MLA + DSA attention module."""

import torch

from deepseek_v3_2 import DeepSeekV32Config
from deepseek_v3_2.model import DeepSeekV32Attention, DeepSeekV32RotaryEmbedding


def _attn_and_rope(use_dsa=True, top_k=16):
    cfg = DeepSeekV32Config(
        hidden_size=64, num_attention_heads=4, q_lora_rank=32, kv_lora_rank=16,
        qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8, max_position_embeddings=512,
        use_dsa=use_dsa, dsa_top_k=top_k, dsa_n_indexer_heads=2, dsa_indexer_dim=8,
        num_hidden_layers=1,
    )
    torch.manual_seed(0)
    attn = DeepSeekV32Attention(cfg, layer_idx=0).eval()
    rope = DeepSeekV32RotaryEmbedding(cfg)
    return cfg, attn, rope


def _run(attn, rope, h, dsa_mode="sparse", return_index_scores=False):
    B, S, _ = h.shape
    pos = torch.arange(S).unsqueeze(0)
    pe = rope(h, pos)
    return attn(h, pos, pe, dsa_mode=dsa_mode, output_attentions=True, return_index_scores=return_index_scores)


def test_output_shape():
    """Attention output preserves (B, S, hidden_size)."""
    cfg, attn, rope = _attn_and_rope()
    h = torch.randn(2, 20, 64)
    out = _run(attn, rope, h)
    assert out["attn_output"].shape == (2, 20, 64)


def test_dense_mode_no_selection():
    """In dense mode no top-k mask is produced."""
    cfg, attn, rope = _attn_and_rope()
    h = torch.randn(1, 40, 64)
    out = _run(attn, rope, h, dsa_mode="dense")
    assert out["selected_mask"] is None


def test_sparse_selection_when_long():
    """In sparse mode with seq_len > top_k, exactly top_k keys are selected per query."""
    cfg, attn, rope = _attn_and_rope(top_k=8)
    h = torch.randn(1, 32, 64)
    out = _run(attn, rope, h, dsa_mode="sparse")
    assert out["selected_mask"] is not None
    # Each query selects exactly top_k positions (causal constraints handled in scoring).
    assert torch.all(out["selected_mask"].sum(-1) == 8)


def test_sparse_falls_back_when_short():
    """When seq_len <= top_k, sparse mode falls back to dense (no mask)."""
    cfg, attn, rope = _attn_and_rope(top_k=64)
    h = torch.randn(1, 16, 64)
    out = _run(attn, rope, h, dsa_mode="sparse")
    assert out["selected_mask"] is None


def test_index_scores_returned():
    """return_index_scores yields a (B, S_q, S_kv) score tensor."""
    cfg, attn, rope = _attn_and_rope()
    h = torch.randn(2, 12, 64)
    out = _run(attn, rope, h, return_index_scores=True)
    assert out["index_scores"].shape == (2, 12, 12)


def test_baseline_without_dsa():
    """With use_dsa=False the module has no indexer and still runs."""
    cfg, attn, rope = _attn_and_rope(use_dsa=False)
    assert attn.indexer is None
    h = torch.randn(1, 30, 64)
    out = _run(attn, rope, h)
    assert out["attn_output"].shape == (1, 30, 64)
    assert out["selected_mask"] is None
