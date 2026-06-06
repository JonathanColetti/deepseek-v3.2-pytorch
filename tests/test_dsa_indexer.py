"""Unit tests for the Lightning Indexer."""

import torch
import torch.nn.functional as F

from deepseek_v3_2.indexer import LightningIndexer


def _make_indexer(hidden=64, heads=4, dim=8, rope_head_dim=4):
    torch.manual_seed(0)
    return LightningIndexer(
        hidden_size=hidden,
        n_indexer_heads=heads,
        indexer_dim=dim,
        rope_head_dim=rope_head_dim,
    )


def _pos(B, S):
    return torch.arange(S).unsqueeze(0).expand(B, -1)


def test_output_shape():
    """Indexer scores have shape (B, S_q, S_kv)."""
    idx = _make_indexer()
    h = torch.randn(2, 10, 64)
    scores = idx(h, _pos(2, 10))
    assert scores.shape == (2, 10, 10)


def test_scoring_matches_formula():
    """forward() implements I_{t,s} = sum_h w_{t,h} * ReLU(q_{t,h}.k_s) * scale, with k
    projected from the hidden state h_s, RoPE on q/k, and raw H^{-1/2} scaled weights
    (mirrors transformers' DeepseekV4IndexerScorer)."""
    idx = _make_indexer()
    torch.manual_seed(1)
    h = torch.randn(2, 7, 64)
    pos = _pos(2, 7)
    out = idx(h, pos)

    # Manual recomputation via the module's own projections.
    B, S, _ = h.shape
    H, D = idx.n_indexer_heads, idx.indexer_dim
    k = idx.project_key(h, pos)  # (B, S, D), rope'd
    q = idx.q_proj(h).view(B, S, H, D).transpose(1, 2)
    cos, sin = idx._rotary(pos, h.device, q.dtype)
    q = idx._apply_rope(q, cos, sin)
    scores = F.relu(torch.einsum("bhqd,bsd->bqhs", q.float(), k.float())) * idx.scale
    w = idx.w_proj(h).float() * idx.head_weight_scale
    expected = (w.unsqueeze(-1) * scores).sum(dim=2)
    assert torch.allclose(out, expected, atol=1e-5)


def test_head_weights_are_raw_not_softmax():
    """Per-head weights are the raw projection scaled by H^{-1/2} (no softmax), so
    they are not constrained to be non-negative or to sum to 1."""
    idx = _make_indexer()
    h = torch.randn(4, 6, 64)
    w = idx.w_proj(h) * idx.head_weight_scale
    sums = w.sum(-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
    assert (w < 0).any()  # raw weights take negative values


def test_relu_clips_negative_dot_products():
    """The per-head activation is ReLU(q.k): negating a key (flipping the dot product
    sign for every head) cannot make the head term more negative, it floors at 0."""
    idx = _make_indexer()
    # Single-head, identity-ish: make head weight positive so index sign == ReLU sign.
    torch.manual_seed(2)
    h = torch.randn(1, 4, 64)
    pos = _pos(1, 4)
    # Force positive head weights so the weighted sum exposes the ReLU floor.
    with torch.no_grad():
        idx.w_proj.weight.copy_(idx.w_proj.weight.abs())
    h = h.abs()  # positive hidden gives positive w
    scores = idx(h, pos)
    assert torch.all(scores >= 0)


def test_top_k_count():
    """select_top_k returns exactly k indices per query and a matching mask."""
    idx = _make_indexer()
    scores = torch.randn(2, 12, 12)
    indices, mask = idx.select_top_k(scores, top_k=4)
    assert indices.shape == (2, 12, 4)
    assert torch.all(mask.sum(-1) == 4)


def test_causal_mask_applied():
    """Future positions receive a very negative score after the causal mask."""
    idx = _make_indexer()
    S = 5
    h = torch.randn(1, S, 64)
    min_val = torch.finfo(torch.float32).min
    q = torch.arange(S).view(S, 1)
    k = torch.arange(S).view(1, S)
    causal = torch.where(k <= q, 0.0, min_val).view(1, 1, S, S)
    scores = idx(h, _pos(1, S), causal)
    # Query 0 can only see key 0; key 1.. must be strongly negative.
    assert scores[0, 0, 1] <= min_val / 2


def test_kl_loss_shape():
    """compute_kl_loss returns a scalar."""
    idx = _make_indexer()
    scores = torch.randn(2, 8, 8)
    attn = torch.rand(2, 4, 8, 8)
    attn = attn / attn.sum(-1, keepdim=True)
    loss = idx.compute_kl_loss(scores, attn)
    assert loss.dim() == 0


def test_parameter_count():
    """Total indexer parameters match H_I*D_I*hidden + D_I*hidden + H_I*hidden
    (q_proj + k_proj-from-hidden + w_proj). The rope/nope split adds no parameters."""
    hidden, heads, dim = 64, 4, 8
    idx = _make_indexer(hidden, heads, dim)
    expected = heads * dim * hidden + dim * hidden + heads * hidden
    actual = sum(p.numel() for p in idx.parameters())
    assert actual == expected == idx.num_parameters()
