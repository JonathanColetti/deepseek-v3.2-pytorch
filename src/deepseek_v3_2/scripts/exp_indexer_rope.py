#!/usr/bin/env python3
# coding=utf-8
"""
Does the faithful (YaRN, shared frequency) indexer RoPE actually matter?
A controlled long-context ablation on REAL data.

The Lightning Indexer must imitate what dense attention attends to. The official
DeepSeek V3.2 indexer reuses the main attention's YaRN scaled frequencies so its sense
of position matches the attention it imitates, including beyond the original context
window. A naive port instead uses plain rope_theta frequencies on the indexer.

This experiment isolates exactly that one variable:

  1. Pretrain one small model with YaRN active (so dense attention develops a real,
     YaRN shaped positional structure) on WikiText-2 at a context length well past the
     YaRN original window.
  2. Deep copy it into two arms that share identical weights and an identical (random,
     untrained) indexer:
         arm "yarn"  : indexer keeps the YaRN frequencies shared with the attention.
         arm "plain" : indexer frequencies overwritten with plain rope_theta (the bug).
  3. Warm up each arm's indexer with the same data and seed (KL alignment to dense
     attention), everything else frozen.
  4. Measure indexer recall@k against dense attention's own top-k, split by query
     position: inside the original window vs the extended region.

Hypothesis: the YaRN arm recovers more of dense attention's chosen tokens, with the
larger gap in the extended region, where the plain frequencies diverge most from the
attention they are trying to match.

Usage:
    python3 -m deepseek_v3_2.scripts.exp_indexer_rope
    deepseek-exp-rope
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from .ablations import (
    batches,
    cycle,
    freeze_except_indexer,
    load_wikitext2,
    unfreeze_all,
)
from .. import DeepSeekV32Config, DeepSeekV32ForCausalLM
from ..indexer import compute_rope_inv_freq


def build_config(seq_len: int, original_window: int, top_k: int) -> DeepSeekV32Config:
    """Small all-dense (no MoE) DSA model with YaRN, sized for a single GPU."""
    return DeepSeekV32Config(
        vocab_size=50257,
        hidden_size=384,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=6,
        num_key_value_heads=6,
        q_lora_rank=192,
        kv_lora_rank=64,
        qk_nope_head_dim=48,
        qk_rope_head_dim=48,
        v_head_dim=48,
        max_position_embeddings=seq_len,
        rope_theta=10000.0,
        rope_interleave=True,
        rope_scaling={
            "type": "yarn",
            "factor": 4,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": original_window,
        },
        # All layers dense -> no MoE routing, keeps the ablation about RoPE only.
        first_k_dense_replace=6,
        moe_layer_freq=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=256,
        use_dsa=True,
        dsa_top_k=top_k,
        dsa_n_indexer_heads=8,
        dsa_indexer_dim=96,        # rope 48 / nope 48, mirrors the official 64/64 split
        tie_word_embeddings=True,
        bos_token_id=50256,
        eos_token_id=50256,
    )


def pretrain(model, ids, steps, seq_len, batch, lr, device):
    """Plain dense LM pretraining so attention develops real positional structure."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    it = cycle(ids, seq_len, batch)
    model.train()
    for s in range(steps):
        x = next(it).to(device)
        out = model(input_ids=x, labels=x, dsa_mode="dense", use_cache=False)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (s + 1) % 100 == 0:
            print(f"  pretrain {s + 1}/{steps}  loss={out.loss.item():.3f}")


def warmup_indexer(model, ids, steps, seq_len, batch, lr, device, seed):
    """KL align the indexer to dense attention, everything else frozen."""
    torch.manual_seed(seed)  # identical data stream across arms
    freeze_except_indexer(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    it = cycle(ids, seq_len, batch)
    model.train()
    last = 0.0
    for s in range(steps):
        x = next(it).to(device)
        out = model(input_ids=x, dsa_mode="dense", collect_dsa_losses=True, use_cache=False)
        opt.zero_grad(set_to_none=True)
        out.dsa_kl_loss.backward()
        opt.step()
        last = out.dsa_kl_loss.item()
        if (s + 1) % 100 == 0:
            print(f"    warmup {s + 1}/{steps}  kl={last:.4f}")
    unfreeze_all(model)
    return last


@torch.no_grad()
def _capture(model, x, device):
    captured = []

    def hook(_m, _i, out):
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


@torch.no_grad()
def recall_by_position(captured, k, boundary, device) -> Tuple[float, float]:
    """Mean indexer recall@k vs dense attention top-k, split at `boundary`.

    Returns (recall_in_window, recall_extended). Only queries with strictly more than
    k causal candidates are scored.
    """
    in_sum = in_n = ext_sum = ext_n = 0.0
    for index_scores, attn in captured:
        B, L, _ = index_scores.shape
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        p = attn.sum(1).masked_fill(~causal, float("-inf"))
        dmask = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        dmask.scatter_(-1, p.topk(k, dim=-1).indices, True)
        s = index_scores.masked_fill(~causal, float("-inf"))
        smask = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        smask.scatter_(-1, s.topk(k, dim=-1).indices, True)
        recall = (dmask & smask).sum(-1).float() / k  # (B, L)
        pos = torch.arange(L, device=device)
        valid = (pos + 1) > k
        inw = valid & (pos < boundary)
        ext = valid & (pos >= boundary)
        in_sum += recall[:, inw].sum().item()
        in_n += B * int(inw.sum().item())
        ext_sum += recall[:, ext].sum().item()
        ext_n += B * int(ext.sum().item())
    return in_sum / max(1.0, in_n), ext_sum / max(1.0, ext_n)


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--original-window", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pretrain-steps", type=int, default=600)
    ap.add_argument("--warmup-steps", type=int, default=400)
    ap.add_argument("--pretrain-lr", type=float, default=3e-4)
    ap.add_argument("--warmup-lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/exp_indexer_rope.json")

    if args is None:
        parsed = ap.parse_args()
    else:
        parsed = ap.parse_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"YaRN window={parsed.original_window}, seq_len={parsed.seq_len}, "
          f"top_k={parsed.top_k} (boundary between in-window and extended = {parsed.original_window})")

    cfg = build_config(parsed.seq_len, parsed.original_window, parsed.top_k)
    train_ids = load_wikitext2(parsed.seq_len, cfg.vocab_size, "train")
    val_ids = load_wikitext2(parsed.seq_len, cfg.vocab_size, "validation")
    val_batch = next(batches(val_ids, parsed.seq_len, parsed.batch_size, shuffle=False))

    # One shared pretrained model (YaRN attention).
    torch.manual_seed(parsed.seed)
    print(f"\n[1/3] Pretraining shared model ({parsed.pretrain_steps} steps)...")
    model = DeepSeekV32ForCausalLM(cfg).to(device)
    pretrain(model, train_ids, parsed.pretrain_steps, parsed.seq_len, parsed.batch_size,
             parsed.pretrain_lr, device)

    # Two arms with identical weights + identical random indexer.
    model_yarn = copy.deepcopy(model)
    model_plain = copy.deepcopy(model)
    plain_inv, _ = compute_rope_inv_freq(
        cfg.qk_rope_head_dim, cfg.rope_theta, None, cfg.max_position_embeddings
    )
    for layer in model_plain.model.layers:
        idx = layer.self_attn.indexer
        if idx is not None:
            idx.inv_freq = plain_inv.clone().to(device)  # overwrite with plain rope_theta

    # Sanity which the arms really do differ only in indexer frequencies.
    yarn_inv = model_yarn.model.layers[0].self_attn.indexer.inv_freq
    plain_inv0 = model_plain.model.layers[0].self_attn.indexer.inv_freq
    assert not torch.allclose(yarn_inv.cpu(), plain_inv0.cpu()), "arms must differ in inv_freq"

    # Warm up each indexer identically, then measure recall by position.
    results = {}
    for name, m in (("yarn", model_yarn), ("plain", model_plain)):
        print(f"\n[2/3] Warm-up indexer, arm '{name}' ({parsed.warmup_steps} steps)...")
        kl = warmup_indexer(m, train_ids, parsed.warmup_steps, parsed.seq_len,
                            parsed.batch_size, parsed.warmup_lr, device, seed=parsed.seed + 1)
        rin, rext = recall_by_position(_capture(m, val_batch, device),
                                       parsed.top_k, parsed.original_window, device)
        results[name] = {"warmup_kl": kl, "recall_in_window": rin, "recall_extended": rext}
        print(f"    arm '{name}': KL={kl:.4f}  "
              f"recall(in-window)={rin:.3f}  recall(extended)={rext:.3f}")

    print(f"\n[3/3] {'='*60}")
    print("  Indexer recall@{} vs dense attention top-k (higher is better)".format(parsed.top_k))
    print(f"  {'='*60}")
    print(f"  {'arm':<8}{'in-window (<%d)' % parsed.original_window:>18}{'extended (>=%d)' % parsed.original_window:>18}")
    for name in ("yarn", "plain"):
        r = results[name]
        print(f"  {name:<8}{r['recall_in_window']:>18.3f}{r['recall_extended']:>18.3f}")
    g_in = results["yarn"]["recall_in_window"] - results["plain"]["recall_in_window"]
    g_ext = results["yarn"]["recall_extended"] - results["plain"]["recall_extended"]
    print(f"  {'-'*44}")
    print(f"  {'YaRN - plain':<8}{g_in:>18.3f}{g_ext:>18.3f}")
    verdict = (
        "[OK] faithful YaRN indexer recovers more of dense attention, "
        "and the gap is larger in the extended region"
        if g_ext >= g_in and g_ext > 0
        else "result: see numbers above"
    )
    print(f"\n  {verdict}")
    print(f"  {'='*60}")

    out = Path(parsed.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "seq_len": parsed.seq_len, "original_window": parsed.original_window,
            "top_k": parsed.top_k, "pretrain_steps": parsed.pretrain_steps,
            "warmup_steps": parsed.warmup_steps,
        },
        "results": results,
        "gap_yarn_minus_plain": {"in_window": g_in, "extended": g_ext},
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  Report -> {out}")


if __name__ == "__main__":
    main()
