# coding=utf-8
"""Deep behavioral tests for DeepSeek V3.2 DSA."""

import copy

import torch

from deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForCausalLM


def test_gradient_flow_indexer(tiny_model, sample_inputs):
    """All Lightning Indexer parameters receive non-zero gradients from the KL loss."""
    tiny_model.train()
    out = tiny_model(**sample_inputs, collect_dsa_losses=True)
    out.dsa_kl_loss.backward()
    indexer_params = [(n, p) for n, p in tiny_model.named_parameters() if "indexer" in n]
    assert len(indexer_params) > 0
    for name, p in indexer_params:
        assert p.grad is not None, f"{name} has no grad"
        assert p.grad.abs().sum() > 0, f"{name} has zero grad"


def test_gradient_flow_main_model(tiny_model, sample_inputs):
    """Non-indexer parameters receive non-zero gradients from the LM loss."""
    tiny_model.train()
    out = tiny_model(input_ids=sample_inputs["input_ids"], labels=sample_inputs["input_ids"])
    out.loss.backward()
    # lm_head is a clear non-indexer parameter that must always receive gradient.
    g = tiny_model.lm_head.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_gradient_isolation_kl_does_not_touch_main_model(tiny_model, sample_inputs):
    """Paper Sec. 2.1: the indexer input is detached so the KL alignment loss updates ONLY
    the indexer: never the main model. Conversely the LM loss never updates the indexer
    (top-k selection is non-differentiable)."""
    tiny_model.train()

    # (a) KL loss -> gradients only on indexer params.
    tiny_model.zero_grad(set_to_none=True)
    out = tiny_model(**sample_inputs, dsa_mode="sparse", collect_dsa_losses=True)
    out.dsa_kl_loss.backward()
    for name, p in tiny_model.named_parameters():
        if p.grad is None:
            continue
        if "indexer" in name:
            assert p.grad.abs().sum() > 0, f"indexer param {name} got no KL grad"
        else:
            assert p.grad.abs().sum() == 0, f"main-model param {name} leaked KL grad"

    # (b) LM loss -> no gradient reaches the indexer.
    tiny_model.zero_grad(set_to_none=True)
    out = tiny_model(input_ids=sample_inputs["input_ids"], labels=sample_inputs["input_ids"],
                     dsa_mode="sparse")
    out.loss.backward()
    for name, p in tiny_model.named_parameters():
        if "indexer" in name and p.grad is not None:
            assert p.grad.abs().sum() == 0, f"indexer param {name} leaked LM grad"


def test_kv_cache_consistency_sparse(tiny_config):
    """Incremental decoding with the cache matches the full forward in SPARSE mode,
    including the cached Lightning-Indexer key. top_k >= seq_len here so selection is a
    no-op, isolating the indexer-key caching path."""
    model = DeepSeekV32ForCausalLM(tiny_config).eval()
    torch.manual_seed(7)
    x = torch.randint(0, 1000, (1, 10))
    full = model(input_ids=x, use_cache=False, dsa_mode="sparse").logits
    past, chunks = None, []
    for t in range(10):
        o = model(input_ids=x[:, t:t + 1], past_key_values=past, use_cache=True, dsa_mode="sparse")
        past = o.past_key_values
        chunks.append(o.logits)
    inc = torch.cat(chunks, dim=1)
    assert torch.allclose(full, inc, atol=1e-3)


def test_indexer_score_monotonicity(tiny_config):
    """After KL alignment, the indexer distribution correlates positively with the
    aggregated attention distribution.

    The KL warm-up loss explicitly drives softmax(I_{t,.}) toward the L1-normalized,
    head-aggregated attention weights, so we measure the correlation between those two
    *distributions* over the valid (causal) entries.
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    model = DeepSeekV32ForCausalLM(tiny_config)
    x = torch.randint(0, 1000, (1, 24))
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if "indexer" in n], lr=2e-2)
    for _ in range(200):
        out = model(input_ids=x, collect_dsa_losses=True, dsa_mode="dense")
        opt.zero_grad()
        out.dsa_kl_loss.backward()
        opt.step()

    model.eval()
    layer = model.model.layers[0]
    h = model.model.embed_tokens(x)
    h = layer.input_layernorm(h)
    pos = torch.arange(24).unsqueeze(0)
    pe = model.model.rotary_emb(h, pos)
    attn = layer.self_attn(h, pos, pe, output_attentions=True, return_index_scores=True)

    q = F.softmax(attn["index_scores"][0], dim=-1)  # indexer distribution (S, S)
    p = attn["attn_weights"][0].sum(0)  # aggregated over heads (S, S)
    p = p / (p.sum(-1, keepdim=True) + 1e-9)

    mask = torch.tril(torch.ones(24, 24, dtype=torch.bool))
    s = q[mask]
    t = p[mask]
    s = (s - s.mean()) / (s.std() + 1e-6)
    t = (t - t.mean()) / (t.std() + 1e-6)
    corr = (s * t).mean()
    assert corr > 0.3, f"expected strong positive correlation, got {corr}"


def test_sparsity_exactly_top_k(tiny_config):
    """In sparse mode each query attends to exactly min(top_k, seq_len) keys."""
    cfg = copy.deepcopy(tiny_config)
    cfg.dsa_top_k = 8
    model = DeepSeekV32ForCausalLM(cfg).eval()
    x = torch.randint(0, 1000, (1, 32))
    h = model.model.embed_tokens(x)
    layer = model.model.layers[0]
    h = layer.input_layernorm(h)
    pos = torch.arange(32).unsqueeze(0)
    pe = model.model.rotary_emb(h, pos)
    out = layer.self_attn(h, pos, pe, dsa_mode="sparse")
    assert torch.all(out["selected_mask"].sum(-1) == 8)


def test_no_future_leakage(tiny_model):
    """Outputs at positions 0..t-1 are unaffected by changing tokens at t..T (causality)."""
    torch.manual_seed(3)
    x = torch.randint(0, 1000, (1, 16))
    x2 = x.clone()
    x2[:, 8:] = torch.randint(0, 1000, (1, 8))
    a = tiny_model(input_ids=x, dsa_mode="dense").logits[:, :8]
    b = tiny_model(input_ids=x2, dsa_mode="dense").logits[:, :8]
    assert torch.allclose(a, b, atol=1e-4)


def test_kl_loss_decreases(tiny_config):
    """After 10 warm-up steps the indexer KL loss decreases."""
    torch.manual_seed(0)
    model = DeepSeekV32ForCausalLM(tiny_config)
    x = torch.randint(0, 1000, (2, 24))
    opt = torch.optim.Adam([p for n, p in model.named_parameters() if "indexer" in n], lr=1e-2)
    first, last = None, None
    for i in range(10):
        out = model(input_ids=x, collect_dsa_losses=True, dsa_mode="dense")
        loss = out.dsa_kl_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.detach().item()
        last = loss.detach().item()
    assert last < first


def test_numerical_stability_fp16(tiny_config):
    """Forward pass in fp16 produces no NaN/Inf."""
    model = DeepSeekV32ForCausalLM(tiny_config).half().eval()
    x = torch.randint(0, 1000, (2, 32))
    out = model(input_ids=x)
    assert torch.isfinite(out.logits).all()


def test_numerical_stability_bf16(tiny_config):
    """Forward pass in bf16 produces no NaN/Inf."""
    model = DeepSeekV32ForCausalLM(tiny_config).to(torch.bfloat16).eval()
    x = torch.randint(0, 1000, (2, 32))
    out = model(input_ids=x)
    assert torch.isfinite(out.logits).all()


def test_kv_cache_consistency(tiny_model):
    """Logits computed incrementally with a KV cache match the full forward pass."""
    torch.manual_seed(5)
    x = torch.randint(0, 1000, (1, 12))
    full = tiny_model(input_ids=x, use_cache=False, dsa_mode="dense").logits
    past = None
    chunks = []
    for t in range(12):
        o = tiny_model(input_ids=x[:, t:t + 1], past_key_values=past, use_cache=True, dsa_mode="dense")
        past = o.past_key_values
        chunks.append(o.logits)
    inc = torch.cat(chunks, dim=1)
    assert torch.allclose(full, inc, atol=1e-3)


def test_top_k_selection_deterministic(tiny_config):
    """Identical inputs produce identical top-k selections."""
    cfg = copy.deepcopy(tiny_config)
    cfg.dsa_top_k = 8
    model = DeepSeekV32ForCausalLM(cfg).eval()
    x = torch.randint(0, 1000, (1, 32))
    h = model.model.embed_tokens(x)
    layer = model.model.layers[0]
    hh = layer.input_layernorm(h)
    pos = torch.arange(32).unsqueeze(0)
    pe = model.model.rotary_emb(hh, pos)
    m1 = layer.self_attn(hh, pos, pe, dsa_mode="sparse")["selected_mask"]
    m2 = layer.self_attn(hh, pos, pe, dsa_mode="sparse")["selected_mask"]
    assert torch.equal(m1, m2)


def test_moe_routing_with_dsa(tiny_model, sample_inputs):
    """MoE routing functions correctly while DSA is active (finite output, routed mix)."""
    out = tiny_model(**sample_inputs, dsa_mode="sparse")
    assert torch.isfinite(out.logits).all()
    # Confirm a MoE layer exists and its router produces valid top k indices.
    moe = tiny_model.model.layers[1].mlp
    flat = torch.randn(10, tiny_model.config.hidden_size)
    idx, w = moe.gate(flat)
    assert idx.shape == (10, tiny_model.config.num_experts_per_tok)
    assert torch.all(idx >= 0) and torch.all(idx < tiny_model.config.n_routed_experts)


def test_dsa_disabled_matches_baseline(tiny_config):
    """With use_dsa=False the model runs as plain MLA and emits no DSA losses."""
    cfg = copy.deepcopy(tiny_config)
    cfg.use_dsa = False
    model = DeepSeekV32ForCausalLM(cfg).eval()
    x = torch.randint(0, 1000, (2, 32))
    out = model(input_ids=x, collect_dsa_losses=True)
    assert out.logits.shape == (2, 32, 1000)
    assert out.dsa_kl_loss is None
    for n, _ in model.named_parameters():
        assert "indexer" not in n
