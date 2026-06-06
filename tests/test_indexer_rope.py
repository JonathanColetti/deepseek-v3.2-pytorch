"""Tests that the Lightning Indexer RoPE is faithful to the official DeepSeek V3.2
inference code: YaRN frequencies shared with the main attention, a non interleaved
(half split) layout, a rope/nope channel split, and no mscale on the rotary vectors.
"""

import torch

from deepseek_v3_2.config import DeepSeekV32Config
from deepseek_v3_2.indexer import LightningIndexer, compute_rope_inv_freq, rotate_half
from deepseek_v3_2.model import DeepSeekV32RotaryEmbedding


YARN = {
    "type": "yarn",
    "factor": 40,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
}


def _pos(B, S):
    return torch.arange(S).unsqueeze(0).expand(B, -1)


def _indexer(rope_head_dim, dim, rope_scaling=YARN, theta=10000.0, max_pos=163840):
    return LightningIndexer(
        hidden_size=8,
        n_indexer_heads=1,
        indexer_dim=dim,
        rope_head_dim=rope_head_dim,
        rope_theta=theta,
        rope_scaling=rope_scaling,
        max_position_embeddings=max_pos,
    )


def test_indexer_uses_yarn_frequencies():
    """With a YaRN config the indexer inv_freq equals the YaRN inv_freq and differs
    from plain rope_theta frequencies (so YaRN really is applied to the indexer)."""
    rope_dim, theta, max_pos = 64, 10000.0, 163840
    idx = _indexer(rope_dim, 128, YARN, theta, max_pos)
    yarn_inv, _ = compute_rope_inv_freq(rope_dim, theta, YARN, max_pos)
    plain_inv, _ = compute_rope_inv_freq(rope_dim, theta, None, max_pos)
    assert torch.allclose(idx.inv_freq, yarn_inv)
    assert not torch.allclose(yarn_inv, plain_inv)


def test_indexer_freqs_match_main_attention():
    """The indexer and the main rotary share identical frequencies when the indexer
    rope_head_dim equals qk_rope_head_dim (the official design)."""
    cfg = DeepSeekV32Config(
        hidden_size=64,
        qk_rope_head_dim=64,
        rope_theta=10000.0,
        rope_scaling=YARN,
        max_position_embeddings=163840,
        dsa_indexer_dim=128,
        dsa_n_indexer_heads=2,
    )
    main = DeepSeekV32RotaryEmbedding(cfg)
    idx = _indexer(cfg.qk_rope_head_dim, cfg.dsa_indexer_dim, cfg.rope_scaling)
    assert torch.allclose(main.inv_freq, idx.inv_freq)


def test_indexer_rope_has_no_mscale():
    """The YaRN mscale is a softmax temperature, not part of the rotary vectors. The
    indexer cos/sin keep unit magnitude even with a YaRN config, whereas the main
    rotary scales them by mscale > 1."""
    rope_dim = 64
    idx = _indexer(rope_dim, 128, YARN)
    cos, sin = idx._rotary(_pos(1, 8), torch.device("cpu"), torch.float32)
    mag = cos**2 + sin**2
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)

    cfg = DeepSeekV32Config(
        hidden_size=64, qk_rope_head_dim=rope_dim, rope_theta=10000.0,
        rope_scaling=YARN, max_position_embeddings=163840,
    )
    assert DeepSeekV32RotaryEmbedding(cfg).mscale > 1.0


def test_rope_nope_split_leaves_nope_channels_unrotated():
    """Only the first rope_head_dim channels carry position. The remaining nope channels
    are constant across positions when the input vector is constant."""
    rope_dim, dim = 32, 64
    idx = _indexer(rope_dim, dim)
    v = torch.randn(1, 1, 1, dim)
    x = v.expand(1, 1, 5, dim).clone()  # identical vector at every position
    cos, sin = idx._rotary(_pos(1, 5), x.device, x.dtype)
    out = idx._apply_rope(x, cos, sin)
    nope = out[..., rope_dim:]
    assert torch.allclose(nope[:, :, 0], nope[:, :, 4], atol=1e-6)  # nope unchanged
    rope = out[..., :rope_dim]
    assert not torch.allclose(rope[:, :, 0], rope[:, :, 4], atol=1e-4)  # rope rotates


def test_indexer_rope_is_non_interleaved():
    """The indexer uses the half-split (non-interleaved) layout: _apply_rope equals the
    reference x*cos + rotate_half(x)*sin and differs from the interleaved layout the
    main MLA attention uses (the gotcha the official code fixed)."""
    dim = 32
    idx = _indexer(dim, dim)  # all-rope head for a clean comparison
    torch.manual_seed(0)
    x = torch.randn(1, 1, 6, dim)
    cos, sin = idx._rotary(_pos(1, 6), x.device, x.dtype)
    out = idx._apply_rope(x, cos, sin)

    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    ref_noninterleave = x * c + rotate_half(x) * s
    assert torch.allclose(out, ref_noninterleave, atol=1e-6)

    b, h, sq, d = x.shape
    xi = x.view(b, h, sq, d // 2, 2).transpose(4, 3).reshape(b, h, sq, d)
    ref_interleave = xi * c + rotate_half(xi) * s
    assert not torch.allclose(out, ref_interleave, atol=1e-4)


def test_rope_dot_product_depends_on_relative_position():
    """The defining RoPE property: a rotated q.k depends only on the relative offset
    between the two positions, not their absolute values."""
    dim = 32
    idx = _indexer(dim, dim)
    torch.manual_seed(0)
    qv = torch.randn(1, 1, 1, dim)
    kv = torch.randn(1, 1, 1, dim)

    def rotated_dot(qpos, kpos):
        cq, sq = idx._rotary(torch.tensor([[qpos]]), qv.device, qv.dtype)
        ck, sk = idx._rotary(torch.tensor([[kpos]]), kv.device, kv.dtype)
        q = idx._apply_rope(qv, cq, sq)
        k = idx._apply_rope(kv, ck, sk)
        return (q * k).sum().item()

    assert abs(rotated_dot(10, 5) - rotated_dot(20, 15)) < 1e-3   # same offset 5
    assert abs(rotated_dot(10, 5) - rotated_dot(10, 8)) > 1e-3    # offset 5 vs 2
