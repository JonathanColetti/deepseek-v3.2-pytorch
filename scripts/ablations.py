#!/usr/bin/env python3
# coding=utf-8
"""
scripts/ablations.py: paper-faithful DSA ablations at small scale, on REAL data.

Every experiment uses WikiText-2 (never random tokens) and the corrected Lightning
Indexer, started from a pre-trained dense checkpoint so perplexities live in a real
regime instead of the ~vocab-size noise floor.

  A1  Indexer alignment / recall  : does the indexer's top-k recover the tokens that
                                      dense attention actually attends to? (paper Eq. 2/3)
                                      This is the most direct proof the indexer works:
                                      it is measured, not trained-and-hoped.
  A2  Learned vs Random vs Stride : does the *learned* indexer beat naive sparse
                                      selection on end-task PPL? (compute-matched)
  A3  top-k sensitivity           : PPL vs sparsity tradeoff (graceful degradation).
  A4  Attention FLOP scaling      : O(L^2) -> O(L.k).

Usage:
    python3 scripts/ablations.py                      # all, from the dense checkpoint
    python3 scripts/ablations.py --only a1            # just the recall proof
    python3 scripts/ablations.py --checkpoint <path>  # custom dense checkpoint
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

# again hacky but eh not planning to scale and its because of the src problem in pyproject
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForCausalLM


def load_wikitext2(seq_len: int, vocab_size: int, split: str) -> torch.Tensor:
    cache = Path(f"/tmp/wikitext2_{split}_{seq_len}.pt")
    if cache.exists():
        return torch.load(cache, weights_only=True)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.model_max_length = 1_000_000
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [t for t in raw["text"] if t.strip()]
    ids = torch.tensor(tok.encode(tok.eos_token.join(texts), add_special_tokens=False),
                       dtype=torch.long).clamp(max=vocab_size - 1)
    torch.save(ids, cache)
    return ids


def batches(ids: torch.Tensor, seq_len: int, batch: int, shuffle: bool):
    n = (len(ids) // seq_len) * seq_len
    chunks = ids[:n].view(-1, seq_len)
    order = torch.randperm(len(chunks)) if shuffle else torch.arange(len(chunks))
    for i in range(0, len(order) - batch + 1, batch):
        yield chunks[order[i:i + batch]]


def cycle(ids, seq_len, batch):
    while True:
        yield from batches(ids, seq_len, batch, shuffle=True)


def load_cfg(path: str, **overrides) -> DeepSeekV32Config:
    with open(path) as f:
        raw = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    raw.update(overrides)
    return DeepSeekV32Config(**raw)


def make_model(cfg, checkpoint: str | None, device) -> DeepSeekV32ForCausalLM:
    model = DeepSeekV32ForCausalLM(cfg).to(device)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)  # indexer keys start fresh
    return model


def freeze_except_indexer(model):
    for name, p in model.named_parameters():
        p.requires_grad = "indexer" in name


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def warmup_indexer(model, train_ids, cfg, steps, seq_len, batch, lr, device):
    """Stage-1 warm-up on REAL data: freeze model, align indexer via KL (dense attn)."""
    freeze_except_indexer(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    it = cycle(train_ids, seq_len, batch)
    model.train()
    last = 0.0
    for _ in range(steps):
        x = next(it).to(device)
        out = model(input_ids=x, dsa_mode="dense", collect_dsa_losses=True, use_cache=False)
        opt.zero_grad(set_to_none=True)
        out.dsa_kl_loss.backward()
        opt.step()
        last = out.dsa_kl_loss.item()
    unfreeze_all(model)
    return last


@torch.no_grad()
def eval_ppl(model, val_ids, seq_len, batch, device, dsa_mode):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for x in batches(val_ids, seq_len, batch, shuffle=False):
        x = x.to(device)
        out = model(input_ids=x, labels=x, dsa_mode=dsa_mode, use_cache=False)
        n = x.numel()
        tot_loss += out.loss.item() * n
        tot_tok += n
    return math.exp(min(tot_loss / tot_tok, 20.0))


@torch.no_grad()
def _capture_dense_scores(model, x, device):
    """Run a DENSE forward and capture, per DSA layer, the (index_scores, attn_weights)."""
    captured: List = []

    def hook(_mod, _inp, out):
        if isinstance(out, dict) and out.get("index_scores") is not None and out.get("attn_weights") is not None:
            captured.append((out["index_scores"].detach().float(), out["attn_weights"].detach().float()))

    handles = [layer.self_attn.register_forward_hook(hook)
               for layer in model.model.layers if layer.self_attn.use_dsa]
    model.eval()
    model(input_ids=x.to(device), dsa_mode="dense", collect_dsa_losses=True,
          output_attentions=True, use_cache=False)
    for h in handles:
        h.remove()
    return captured


def _selector_scores(kind, index_scores, causal, generator=None):
    """Per-query selection scores for a given selector, with future positions = -inf."""
    neg = float("-inf")
    if kind == "learned":
        s = index_scores.clone()
    elif kind == "random":
        s = torch.rand(index_scores.shape, device=index_scores.device, generator=generator)
    elif kind == "stride":
        # Prefer every (stride)-th key: keys with smaller (j % S) rank higher. The exact
        # stride is folded into a monotone score so top-k spreads roughly uniformly.
        L = index_scores.shape[-1]
        j = torch.arange(L, device=index_scores.device)
        base = (j % 7 == 0).float() + (j % 3 == 0).float() * 0.3  # deterministic, spread out
        s = base.view(1, 1, L).expand_as(index_scores).clone()
    else:
        raise ValueError(kind)
    return s.masked_fill(~causal, neg)


def selection_quality(captured, k, device, seed=0):
    """For each selector, average over (layers, batch, queries-with->k-candidates):
       recall@k   = |selected  intersect  dense-top-k| / k
       mass@k     = sum attention prob over selected / sum over all valid keys
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    agg = {sel: {"recall": 0.0, "mass": 0.0, "n": 0} for sel in ("learned", "random", "stride")}

    for index_scores, attn in captured:
        B, L, _ = index_scores.shape
        p = attn.sum(1)  # (B,L,L) aggregate over heads
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=index_scores.device))
        p_causal = p.masked_fill(~causal, float("-inf"))
        dense_idx = p_causal.topk(k, dim=-1).indices                 # (B,L,k)
        dense_mask = torch.zeros(B, L, L, dtype=torch.bool, device=index_scores.device)
        dense_mask.scatter_(-1, dense_idx, True)

        p_mass = p.masked_fill(~causal, 0.0)
        total_mass = p_mass.sum(-1).clamp(min=1e-9)                  # (B,L)

        # Only score queries that have strictly MORE than k candidates (non-trivial).
        valid_q = (torch.arange(L, device=index_scores.device) + 1) > k  # (L,)
        nq = valid_q.sum().item()
        if nq == 0:
            continue

        for sel in agg:
            s = _selector_scores(sel, index_scores, causal, generator=gen)
            sel_idx = s.topk(k, dim=-1).indices
            sel_mask = torch.zeros(B, L, L, dtype=torch.bool, device=index_scores.device)
            sel_mask.scatter_(-1, sel_idx, True)
            recall = (dense_mask & sel_mask).sum(-1).float() / k     # (B,L)
            mass = (p_mass * sel_mask).sum(-1) / total_mass          # (B,L)
            agg[sel]["recall"] += recall[:, valid_q].sum().item()
            agg[sel]["mass"] += mass[:, valid_q].sum().item()
            agg[sel]["n"] += B * nq

    return {sel: {"recall": d["recall"] / max(1, d["n"]),
                  "mass": d["mass"] / max(1, d["n"])} for sel, d in agg.items()}


def experiment_a1_recall(args, cfg, train_ids, val_ids, device) -> Dict:
    print(f"\n{'='*64}\nA1: Indexer recall vs dense attention (k={args.top_k}, real data)\n{'='*64}")
    k = args.top_k
    val_batch = next(batches(val_ids, args.seq_len, args.batch_size, shuffle=False))

    # Before warm-up: indexer is random.
    model = make_model(cfg, args.checkpoint, device)
    before = selection_quality(_capture_dense_scores(model, val_batch, device), k, device)
    print(f"  random indexer (pre-warmup):  recall@{k}={before['learned']['recall']:.3f}  "
          f"mass@{k}={before['learned']['mass']:.3f}")

    # After warm-up on real data.
    kl = warmup_indexer(model, train_ids, cfg, args.warmup_steps, args.seq_len,
                        args.batch_size, args.warmup_lr, device)
    after = selection_quality(_capture_dense_scores(model, val_batch, device), k, device)
    print(f"  warm-up KL final: {kl:.4f}")
    print(f"  {'selector':<10}  recall@{k}   attn-mass@{k}")
    print(f"  {'-'*36}")
    for sel in ("learned", "random", "stride"):
        tag = "learned[OK]" if sel == "learned" else sel
        print(f"  {tag:<10}  {after[sel]['recall']:.3f}      {after[sel]['mass']:.3f}")
    expected_random = k / (args.seq_len / 2)  # rough chance level
    print(f"  (chance-level recall ~ k / (mean #candidates) ~ {expected_random:.3f})")
    return {"k": k, "pre_warmup": before, "post_warmup": after, "warmup_kl": kl}


from contextlib import contextmanager


@contextmanager
def _patched_selector(model, kind, device, seed=0):
    """Override the indexer's scoring with a naive selector (random / stride)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    originals = {}

    def make_fwd():
        def fwd(hidden_states, position_ids, causal_mask=None, k_index=None):
            B, S_q, _ = hidden_states.shape
            S_kv = k_index.shape[1] if k_index is not None else S_q
            if kind == "random":
                scores = torch.rand(B, S_q, S_kv, device=device, generator=gen)
            else:  # stride
                j = torch.arange(S_kv, device=device)
                scores = ((j % 7 == 0).float() + (j % 3 == 0).float() * 0.3).view(1, 1, S_kv).expand(B, S_q, S_kv).clone()
            if causal_mask is not None:
                scores = scores + causal_mask.squeeze(1).float()
            return scores
        return fwd

    for layer in model.model.layers:
        if layer.self_attn.indexer is not None:
            originals[layer] = layer.self_attn.indexer.forward
            layer.self_attn.indexer.forward = make_fwd()
    try:
        yield
    finally:
        for layer, fn in originals.items():
            layer.self_attn.indexer.forward = fn


def _finetune(model, train_ids, cfg, steps, seq_len, batch, lr, device, dsa_mode, selector=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    it = cycle(train_ids, seq_len, batch)
    model.train()
    for _ in range(steps):
        x = next(it).to(device)
        ctx = _patched_selector(model, selector, device) if selector else _nullctx()
        with ctx:
            out = model(input_ids=x, labels=x, dsa_mode=dsa_mode,
                        collect_dsa_losses=(selector is None and dsa_mode == "sparse"), use_cache=False)
            kl = out.dsa_kl_loss if out.dsa_kl_loss is not None else 0.0
            loss = out.loss + (cfg.dsa_kl_loss_weight * kl if kl else 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()


@contextmanager
def _nullctx():
    yield


def experiment_a2_learned_vs_naive(args, cfg, train_ids, val_ids, device) -> Dict:
    print(f"\n{'='*64}\nA2: Learned vs Random vs Stride vs Dense (PPL, real data, {args.sparse_steps} steps)\n{'='*64}")
    res = {}

    # Dense (compute-matched baseline).
    dense_cfg = load_cfg(args.config, use_dsa=False)
    dense = make_model(dense_cfg, args.checkpoint, device)
    _finetune(dense, train_ids, dense_cfg, args.sparse_steps, args.seq_len, args.batch_size,
              args.sparse_lr, device, dsa_mode="dense")
    res["Dense"] = eval_ppl(dense, val_ids, args.seq_len, args.batch_size, device, "dense")
    del dense

    # DSA-Learned: warm up indexer, then fine-tune all params with the real indexer.
    learned = make_model(cfg, args.checkpoint, device)
    warmup_indexer(learned, train_ids, cfg, args.warmup_steps, args.seq_len, args.batch_size,
                   args.warmup_lr, device)
    _finetune(learned, train_ids, cfg, args.sparse_steps, args.seq_len, args.batch_size,
              args.sparse_lr, device, dsa_mode="sparse", selector=None)
    res["DSA-Learned"] = eval_ppl(learned, val_ids, args.seq_len, args.batch_size, device, "sparse")
    del learned

    # Naive selectors: fine-tune with random / stride selection (no indexer learning).
    for sel in ("random", "stride"):
        m = make_model(cfg, args.checkpoint, device)
        _finetune(m, train_ids, cfg, args.sparse_steps, args.seq_len, args.batch_size,
                  args.sparse_lr, device, dsa_mode="sparse", selector=sel)
        with _patched_selector(m, sel, device):
            res[f"DSA-{sel.capitalize()}"] = eval_ppl(m, val_ids, args.seq_len, args.batch_size, device, "sparse")
        del m

    print(f"  {'variant':<14} val PPL   Delta vs Dense")
    print(f"  {'-'*40}")
    for label in ("Dense", "DSA-Learned", "DSA-Random", "DSA-Stride"):
        d = res[label] - res["Dense"]
        print(f"  {label:<14} {res[label]:7.2f}   {d:+.2f} ({d/res['Dense']*100:+.1f}%)")
    ok = res["DSA-Learned"] <= res["DSA-Random"] <= res["DSA-Stride"]
    print(f"  Learned <= Random <= Stride: {'[OK] YES: indexer adds value' if ok else '[!] ordering not clean (try more steps)'}")
    return res


def experiment_a3_topk_sweep(args, cfg, train_ids, val_ids, device) -> Dict:
    print(f"\n{'='*64}\nA3: top-k sensitivity (warm-up once, eval at each k; real data)\n{'='*64}")
    model = make_model(cfg, args.checkpoint, device)
    warmup_indexer(model, train_ids, cfg, args.warmup_steps, args.seq_len, args.batch_size,
                   args.warmup_lr, device)
    res = {}
    ks = [kk for kk in (32, 64, 128, 256) if kk < args.seq_len]
    for kk in ks:
        for layer in model.model.layers:
            layer.self_attn.dsa_top_k = kk
        res[str(kk)] = eval_ppl(model, val_ids, args.seq_len, args.batch_size, device, "sparse")
    res["dense"] = eval_ppl(model, val_ids, args.seq_len, args.batch_size, device, "dense")
    print(f"  {'top-k':<8} {'%ctx':<6} val PPL")
    print(f"  {'-'*26}")
    for kk in ks:
        print(f"  {kk:<8} {kk/args.seq_len*100:>4.0f}%  {res[str(kk)]:7.2f}")
    print(f"  {'dense':<8} {'100%':<6} {res['dense']:7.2f}")
    return res


def experiment_a4_flops(args, cfg) -> Dict:
    print(f"\n{'='*64}\nA4: Attention FLOP scaling: O(L^2) dense vs O(L.k) DSA (k={args.top_k})\n{'='*64}")
    H = cfg.num_attention_heads
    d = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    rows = []
    print(f"  {'seq_len':<8} {'dense GFLOP':<12} {'DSA GFLOP':<12} {'ratio':<8}")
    print(f"  {'-'*42}")
    for L in (128, 512, 2048, 8192, 32768, 131072):
        k = min(args.top_k, L)
        dense = 2 * H * L * L * d / 1e9
        dsa = 2 * H * L * k * d / 1e9
        ratio = dense / dsa
        rows.append({"seq_len": L, "dense_gflop": round(dense, 3), "dsa_gflop": round(dsa, 3), "ratio": round(ratio, 1)})
        print(f"  {L:<8} {dense:<12.3f} {dsa:<12.3f} {ratio:<8.1f}x")
    print(f"  -> reduction grows linearly with L once L > k; {args.top_k}/131072 ~ {131072//args.top_k}x at 128K context.")
    return {"top_k": args.top_k, "rows": rows}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = load_cfg(args.config, dsa_top_k=args.top_k)
    if not Path(args.checkpoint).exists():
        print(f"[!]  checkpoint {args.checkpoint} not found: A1/A2/A3 need a dense checkpoint.\n"
              f"   Train one with: python3 train.py --config {args.config} --dataset wikitext2")
    train_ids = load_wikitext2(args.seq_len, cfg.vocab_size, "train")
    val_ids = load_wikitext2(args.seq_len, cfg.vocab_size, "validation")

    report: Dict = {"config": args.config, "checkpoint": args.checkpoint, "top_k": args.top_k}
    run = args.only
    if run in (None, "a1"):
        report["a1_recall"] = experiment_a1_recall(args, cfg, train_ids, val_ids, device)
    if run in (None, "a2"):
        report["a2_learned_vs_naive"] = experiment_a2_learned_vs_naive(args, cfg, train_ids, val_ids, device)
    if run in (None, "a3"):
        report["a3_topk_sweep"] = experiment_a3_topk_sweep(args, cfg, train_ids, val_ids, device)
    if run in (None, "a4"):
        report["a4_flops"] = experiment_a4_flops(args, cfg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/deepseek_v3_2_nano.json")
    p.add_argument("--checkpoint", default="output/wikitext_comparison/dense/checkpoint-final/checkpoint.pt")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--sparse-steps", type=int, default=300)
    p.add_argument("--warmup-lr", type=float, default=3e-4)
    p.add_argument("--sparse-lr", type=float, default=1e-4)
    p.add_argument("--only", choices=["a1", "a2", "a3", "a4"], default=None)
    p.add_argument("--output", default="output/ablations_report.json")
    main(p.parse_args())
