#!/usr/bin/env python3
# coding=utf-8
"""
Important mini proof: DSA from pre-trained dense
=================================================
The actual paper methodology in miniature:

  1. Load a pre-trained DENSE checkpoint (PPL already converged on WikiText-2)
  2. Transplant its weights into a DSA model (Lightning Indexer starts random)
  3. Stage 1: warm-up: freeze main model, train indexer via KL loss
  4. Stage 2: sparse fine-tune: train everything with top-k sparse attention
  5. Evaluate val-PPL at each stage vs the dense baseline

This is the cleanest possible proof: same architecture, same pre-trained weights,
only change is replacing dense attention with DSA.  If final val-PPL ~ dense
val-PPL the implementation is correct.

Usage:
    python3 -m deepseek_v3_2.scripts.proof_dsa_from_pretrained \
        --checkpoint output/wikitext_comparison/dense/checkpoint-final/checkpoint.pt \
        --config     configs/deepseek_v3_2_nano.json \
        --warmup-steps 150 --sparse-steps 300

    deepseek-proof-dsa \
        --checkpoint output/wikitext_comparison/dense/checkpoint-final/checkpoint.pt \
        --config     configs/deepseek_v3_2_nano.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from datasets import load_dataset
from transformers import AutoTokenizer

from .. import DeepSeekV32Config, DeepSeekV32ForCausalLM

def load_wikitext2(seq_len: int, vocab_size: int, split: str = "train") -> torch.Tensor:
    cache = Path(f"/tmp/wikitext2_{split}_{seq_len}.pt")
    if cache.exists():
        return torch.load(cache, weights_only=True)
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


def make_loader(ids: torch.Tensor, seq_len: int, batch: int, shuffle: bool = True):
    n = (len(ids) // seq_len) * seq_len
    chunks = ids[:n].view(-1, seq_len)
    idx = torch.randperm(len(chunks)) if shuffle else torch.arange(len(chunks))
    for i in range(0, len(idx) - batch + 1, batch):
        yield chunks[idx[i:i+batch]]


@torch.no_grad()
def evaluate_ppl(model: nn.Module, val_ids: torch.Tensor,
                 seq_len: int, batch: int, device: torch.device,
                 dsa_mode: str = "sparse") -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x in make_loader(val_ids, seq_len, batch, shuffle=False):
        x = x.to(device)
        out = model(input_ids=x, labels=x, dsa_mode=dsa_mode, use_cache=False)
        n = (x != -100).sum().item()
        total_loss += out.loss.item() * n
        total_tokens += n
    return math.exp(min(total_loss / total_tokens, 20.0))


def load_dense_into_dsa(checkpoint_path: str, cfg: DeepSeekV32Config,
                        device: torch.device) -> DeepSeekV32ForCausalLM:
    """
    Create a DSA model and load dense weights into it.
    The Lightning Indexer parameters are NOT in the dense checkpoint and will
    remain at their random initialisation: exactly as in the paper's recipe.
    """
    model = DeepSeekV32ForCausalLM(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]

    # strict=False: dense checkpoint has no indexer.* keys: that's expected.
    missing, unexpected = model.load_state_dict(state, strict=False)
    indexer_missing = [k for k in missing if "indexer" in k]
    non_indexer_missing = [k for k in missing if "indexer" not in k]

    print(f"  Loaded {len(state) - len(unexpected)} / {len(state)} keys from checkpoint")
    print(f"  Indexer keys initialised fresh: {len(indexer_missing)}")
    if non_indexer_missing:
        print(f"  [!]  Non-indexer keys missing (unexpected): {non_indexer_missing}")

    return model


def freeze_except_indexer(model: nn.Module) -> int:
    n = 0
    for name, p in model.named_parameters():
        p.requires_grad = "indexer" in name
        if p.requires_grad:
            n += p.numel()
    return n


def unfreeze_all(model: nn.Module) -> int:
    n = 0
    for p in model.parameters():
        p.requires_grad = True
        n += p.numel()
    return n


def run_stage(label: str, model: nn.Module, train_ids: torch.Tensor,
              val_ids: torch.Tensor, n_steps: int, seq_len: int, batch: int,
              lr: float, device: torch.device, dsa_mode: str,
              kl_only: bool = False, log_interval: int = 50) -> Dict:
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr/10)

    lm_losses, kl_losses = [], []
    data_iter = _cycle(train_ids, seq_len, batch)

    for step in range(1, n_steps + 1):
        model.train()
        x = next(data_iter).to(device)
        out = model(input_ids=x, labels=None if kl_only else x,
                    dsa_mode=dsa_mode, collect_dsa_losses=True, use_cache=False)

        if kl_only:
            loss = out.dsa_kl_loss
            if loss is None:
                raise RuntimeError("No KL loss: is use_dsa=True?")
            kl_losses.append(loss.detach().item())
        else:
            lm = out.loss
            kl = out.dsa_kl_loss if out.dsa_kl_loss is not None else torch.zeros(1, device=device)
            loss = lm + model.config.dsa_kl_loss_weight * kl
            lm_losses.append(lm.detach().item())
            if out.dsa_kl_loss is not None:
                kl_losses.append(out.dsa_kl_loss.detach().item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        scheduler.step()

        if step % log_interval == 0:
            kl_val = kl_losses[-1] if kl_losses else (loss.detach().item() if kl_only else 0.0)
            lm_val = lm_losses[-1] if lm_losses else 0.0
            ppl_str = f"ppl={math.exp(min(lm_val,20)):.1f}  " if not kl_only else ""
            print(f"    [{label}] step {step:4d}/{n_steps}  {ppl_str}kl={kl_val:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # NOTE: a DSA model is always evaluated in SPARSE mode -
    # evaluating the warmed-up indexer in dense mode would be a tautology (dense attention
    # never invokes the indexer, so it trivially equals the dense baseline regardless of
    # whether the indexer learned anything). Sparse eval measures the aligned indexer.
    val_ppl = evaluate_ppl(model, val_ids, seq_len, batch, device, dsa_mode="sparse")
    return {
        "val_ppl": val_ppl,
        "mean_lm_loss": sum(lm_losses[-20:]) / max(1, len(lm_losses[-20:])),
        "mean_kl": sum(kl_losses[-20:]) / max(1, len(kl_losses[-20:])) if kl_losses else 0.0,
    }


def _cycle(ids, seq_len, batch):
    n = (len(ids) // seq_len) * seq_len
    chunks = ids[:n].view(-1, seq_len)
    while True:
        perm = torch.randperm(len(chunks))
        for i in range(0, len(perm) - batch + 1, batch):
            yield chunks[perm[i:i+batch]]


def main(args=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="output/wikitext_comparison/dense/checkpoint-final/checkpoint.pt")
    p.add_argument("--config",     default="configs/deepseek_v3_2_nano.json")
    p.add_argument("--seq-len",    type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup-steps",  type=int, default=150)
    p.add_argument("--sparse-steps",  type=int, default=300)
    p.add_argument("--warmup-lr",  type=float, default=3e-4)
    p.add_argument("--sparse-lr",  type=float, default=1e-4)
    p.add_argument("--kl-weight",  type=float, default=1.0)
    p.add_argument("--output",     default="output/proof_from_pretrained.json")

    if args is None:
        parsed = p.parse_args()
    else:
        parsed = p.parse_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load config
    with open(parsed.config) as f:
        raw = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    cfg = DeepSeekV32Config(**raw)
    cfg.dsa_kl_loss_weight = parsed.kl_weight

    # Data
    print("Loading WikiText-2 ...")
    train_ids = load_wikitext2(parsed.seq_len, cfg.vocab_size, "train")
    val_ids   = load_wikitext2(parsed.seq_len, cfg.vocab_size, "validation")
    print(f"  train: {len(train_ids):,} tokens ({(len(train_ids)//parsed.seq_len):,} chunks)")
    print(f"  val:   {len(val_ids):,} tokens ({(len(val_ids)//parsed.seq_len):,} chunks)\n")

    # Step 0: baseline: evaluate the DENSE checkpoint as-is --------------
    print("="*55)
    print("Step 0: Dense checkpoint baseline (no DSA)")
    print("="*55)
    dense_cfg = DeepSeekV32Config(**{**raw, "use_dsa": False})
    dense_model = DeepSeekV32ForCausalLM(dense_cfg).to(device)
    ckpt = torch.load(parsed.checkpoint, map_location=device, weights_only=False)
    dense_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    dense_val_ppl = evaluate_ppl(dense_model, val_ids, parsed.seq_len, parsed.batch_size, device, dsa_mode="dense")
    print(f"  Dense val PPL = {dense_val_ppl:.2f}   <- target to match")
    # save mem
    del dense_model

    results = {"dense_val_ppl": dense_val_ppl}

    # Step 1: transplant weights -> DSA model
    print("\n" + "="*55)
    print("Step 1: Load dense weights into DSA model")
    print("="*55)
    model = load_dense_into_dsa(parsed.checkpoint, cfg, device)

    # eval DSA immediately after load (indexer is random -> sparse attn is bad)
    before_warmup_ppl = evaluate_ppl(model, val_ids, parsed.seq_len, parsed.batch_size, device, dsa_mode="sparse")
    print(f"  DSA val PPL (random indexer, before warmup) = {before_warmup_ppl:.2f}")
    results["dsa_before_warmup_ppl"] = before_warmup_ppl

    # warmup: train indexer only
    if parsed.warmup_steps > 0:
        print(f"\n{'='*55}")
        print(f"Step 2: DSA warm-up ({parsed.warmup_steps} steps, indexer only)")
        print(f"  Freeze main model; align indexer to dense attention via KL loss")
        print("="*55)
        n_trainable = freeze_except_indexer(model)
        print(f"  Trainable: {n_trainable/1e3:.1f}K indexer params")
        warmup_result = run_stage(
            "warmup", model, train_ids, val_ids,
            n_steps=parsed.warmup_steps, seq_len=parsed.seq_len, batch=parsed.batch_size,
            lr=parsed.warmup_lr, device=device, dsa_mode="dense", kl_only=True,
            log_interval=max(1, parsed.warmup_steps // 5),
        )
        print(f"\n  After warmup: val PPL (sparse mode) = {warmup_result['val_ppl']:.2f}")
        results["after_warmup_ppl"] = warmup_result["val_ppl"]
        results["warmup_kl"] = warmup_result["mean_kl"]

    # Step 3: sparse finetuning: all params
    print(f"\n{'='*55}")
    print(f"Step 3: DSA sparse fine-tuning ({parsed.sparse_steps} steps, all params)")
    print(f"  top-k = {cfg.dsa_top_k}  ({cfg.dsa_top_k}/{parsed.seq_len} = {cfg.dsa_top_k/parsed.seq_len*100:.0f}% of context)")
    print("="*55)
    n_trainable = unfreeze_all(model)
    print(f"  Trainable: {n_trainable/1e6:.1f}M params")
    sparse_result = run_stage(
        "sparse", model, train_ids, val_ids,
        n_steps=parsed.sparse_steps, seq_len=parsed.seq_len, batch=parsed.batch_size,
        lr=parsed.sparse_lr, device=device, dsa_mode="sparse", kl_only=False,
        log_interval=max(1, parsed.sparse_steps // 10),
    )
    results["after_sparse_ppl"] = sparse_result["val_ppl"]
    results["sparse_kl"] = sparse_result["mean_kl"]

    # compute-matched dense continuation
    # The DSA model received `sparse_steps` extra gradient steps on top of the
    # pretrained checkpoint. To attribute any PPL change to *sparsity* rather than to
    # simply training longer, continue-train the DENSE model for the same number of
    # steps at the same LR, and compare against THAT.
    print(f"\n{'='*55}")
    print(f"Step 4: Dense continuation ({parsed.sparse_steps} steps): fair baseline")
    print("="*55)
    dense_cont = DeepSeekV32ForCausalLM(dense_cfg).to(device)
    dense_cont.load_state_dict(ckpt["model_state_dict"], strict=False)
    unfreeze_all(dense_cont)
    dense_cont_result = run_stage(
        "dense-cont", dense_cont, train_ids, val_ids,
        n_steps=parsed.sparse_steps, seq_len=parsed.seq_len, batch=parsed.batch_size,
        lr=parsed.sparse_lr, device=device, dsa_mode="dense", kl_only=False,
        log_interval=max(1, parsed.sparse_steps // 10),
    )
    # Dense model has no indexer -> eval in dense mode (override run_stage's sparse eval).
    dense_cont_ppl = evaluate_ppl(dense_cont, val_ids, parsed.seq_len, parsed.batch_size, device, dsa_mode="dense")
    results["dense_continuation_ppl"] = dense_cont_ppl
    print(f"\n  Dense continuation: val PPL = {dense_cont_ppl:.2f}")
    del dense_cont

    # -- Final comparison ------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  VALIDATION PPL: WikiText-2")
    print(f"{'='*55}")
    print(f"  Dense baseline (pre-trained)           : {dense_val_ppl:.2f}")
    if parsed.warmup_steps > 0:
        print(f"  DSA (random indexer, no warmup)       : {before_warmup_ppl:.2f}")
        print(f"  DSA (after warmup, sparse eval)       : {warmup_result['val_ppl']:.2f}")
    print(f"  DSA (warmup + {parsed.sparse_steps} sparse steps)     : {sparse_result['val_ppl']:.2f}")
    print(f"  Dense (+{parsed.sparse_steps} steps, fair baseline) : {dense_cont_ppl:.2f}")
    print(f"{'-'*55}")
    # The comparison is DSA-sparse vs the compute match dense continuation
    # both saw the same number of extra gradient steps. (Comparing to the pre-trained
    # baseline instead would credit DSA for simply training longer.)
    delta = sparse_result["val_ppl"] - dense_cont_ppl
    pct   = delta / dense_cont_ppl * 100
    print(f"  Delta DSA vs compute-matched dense         : {delta:+.2f} ({pct:+.2f}%)")
    print(f"{'='*55}")
    if pct < -2.0:
        print(f"  [OK] At {cfg.dsa_top_k}/{parsed.seq_len} sparsity, DSA beats compute-matched dense by {-pct:.1f}%")
    elif abs(pct) < 5.0:
        print(f"  [OK] DSA preserves quality within {abs(pct):.1f}% of dense at equal compute")
    else:
        print(f"  [!]  Gap > 5%: try more sparse steps, larger top-k, or lower kl_weight")
    results["delta_vs_compute_matched_dense_pct"] = pct

    # Save
    out = Path(parsed.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Report -> {out}\n")


if __name__ == "__main__":
    main()
