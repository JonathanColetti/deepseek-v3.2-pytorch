"""

This is a bit of a god class and kinda ugly but do not plan for any extendability so tech debt is kind of used here

train.py: Two-stage DSA training for DeepSeek V3.2
=====================================================

Single GPU / CPU
----------------
    python3 train.py --config configs/deepseek_v3_2_small.json

Two GPUs (recommended: uses FSDP FULL_SHARD)
----------------------------------------------
    torchrun --nproc_per_node=2 train.py --config configs/deepseek_v3_2_small.json

Quick demo (tiny model, proves gains, no downloads needed)
----------------------------------------------------------
    python3 train.py --demo

All options
-----------
    python3 train.py --help

Training stages
---------------
Stage 1: Dense warm-up:
    Freeze the full model; train only the Lightning Indexer via KL divergence
    loss that aligns the indexer distribution to the main attention distribution.

Stage 2: Sparse training:
    Unfreeze everything; replace dense attention with top-k sparse attention;
    optimise LM loss + lambda * KL alignment loss.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForCausalLM
from deepseek_v3_2.model import DeepSeekV32DecoderLayer

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def is_main() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def log_main(msg: str, level: int = logging.INFO) -> None:
    if is_main():
        log.log(level, msg)




def setup_distributed() -> None:
    """Initialise NCCL/Gloo process group when launched via torchrun."""
    if WORLD_SIZE <= 1:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(LOCAL_RANK)
    log_main(f"Distributed: {WORLD_SIZE} ranks, backend={backend}")


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", LOCAL_RANK)
    return torch.device("cpu")


class SyntheticTokenDataset(Dataset):
    """Random token sequences for quickkk test"""

    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 2048, seed: int = 42):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len), generator=rng)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.data[idx]}


class TokenDataset(Dataset):
    """Pre tokenised, chunked dataset for LM training.

    All tokens are concatenated and sliced into non-overlapping `seq_len` length
    windows (no padding, no waste).  The `labels` for each window are the same
    tensor shifted by one: handled inside the model's `forward` via `labels=input_ids`.
    """

    def __init__(self, token_ids: torch.Tensor, seq_len: int):
        n = (len(token_ids) // seq_len) * seq_len
        self.chunks = token_ids[:n].view(-1, seq_len)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.chunks[i]}


def get_wikitext2_dataset(seq_len: int, vocab_size: int, split: str = "train") -> Dataset:
    """Tokenise WikiText-2 with the GPT-2 tokeniser and pack into seq_len chunks.

    Falls back to synthetic data when the ``datasets`` library is not installed.
    Caches the token tensor to ``/tmp/wikitext2_<split>_<seq_len>.pt`` so repeated
    runs don't re-tokenise.
    """
    cache = Path(f"/tmp/wikitext2_{split}_{seq_len}.pt")
    if cache.exists():
        log_main(f"Loading tokenised WikiText-2 from cache {cache}")
        ids = torch.load(cache, weights_only=True)
        return TokenDataset(ids, seq_len)

    # kinda ugly but it is what it is
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        log_main(
            "datasets not installed: falling back to synthetic data. "
            "Run `pip install datasets` to use WikiText-2.",
            logging.WARNING,
        )
        return SyntheticTokenDataset(vocab_size, seq_len)

    log_main(f"Tokenising WikiText-2 ({split} split) ...")
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 1_000_000
    # GPT-2 has no pad token :( add one so batch encode works cleanly.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [t for t in raw["text"] if t.strip()]
    full_text = tokenizer.eos_token.join(texts)
    log_main(f"  Tokenising {len(full_text):,} chars ...")
    ids = torch.tensor(
        tokenizer.encode(full_text, add_special_tokens=False),
        dtype=torch.long,
    )
    # clamp to model vocab size in case tokeniser and config diff
    ids = ids.clamp(max=vocab_size - 1)
    torch.save(ids, cache)
    log_main(f"  {len(ids):,} tokens -> {len(ids)//seq_len:,} chunks of {seq_len}")
    return TokenDataset(ids, seq_len)


def wrap_model_distributed(model: nn.Module, device: torch.device) -> nn.Module:
    """Apply FSDP (2+ GPUs) or DDP (single GPU) wrapping."""
    if WORLD_SIZE <= 1:
        return model.to(device)

    

    if torch.cuda.is_available():
        # 4 FSDP shard per decoder layer, bf16 mixed precision

        # only import if cuda is available ...
        import functools
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={DeepSeekV32DecoderLayer},
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp,
            device_id=LOCAL_RANK,
            use_orig_params=True,
        )
        log_main(f"Model wrapped with FSDP FULL_SHARD across {WORLD_SIZE} GPUs")

        # Activation checkpointing on each decoder layer.
        
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=lambda m: isinstance(m, DeepSeekV32DecoderLayer),
        )
    else:
        # CPU multi process instead if no cuda
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)
        log_main(f"Model wrapped with DDP across {WORLD_SIZE} processes (CPU)")

    return model

def build_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warm-up -> cosine decay down to lr/10."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / WORLD_SIZE


def run_training(args: argparse.Namespace) -> List[Dict]:
    """Full two-stage training loop. Returns a list of per-step metric dicts."""

    if args.demo:
        cfg = _demo_config()
    else:
        with open(args.config) as f:
            cfg = DeepSeekV32Config(**json.load(f))

    # CLI overrides.
    if args.warmup_steps is not None:
        cfg.dsa_warmup_steps = args.warmup_steps
    if args.sparse_steps is not None:
        cfg.dsa_sparse_steps = args.sparse_steps
    if args.kl_weight is not None:
        cfg.dsa_kl_loss_weight = args.kl_weight

    device = get_device()
    dtype = _resolve_dtype(args.dtype)

    log_main(f"Building model (hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, "
             f"experts={cfg.n_routed_experts}, use_dsa={cfg.use_dsa})")
    model = DeepSeekV32ForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_indexer = sum(p.numel() for n, p in model.named_parameters() if "indexer" in n) / 1e3
    log_main(f"Parameters: {n_params:.1f}M total, {n_indexer:.1f}K indexer "
             f"({n_indexer / n_params / 10:.3f}% overhead)")

    model = wrap_model_distributed(model, device)
    if dtype != torch.float32 and WORLD_SIZE <= 1:
        model = model.to(dtype)

    if args.dataset == "wikitext2":
        dataset = get_wikitext2_dataset(args.seq_len, cfg.vocab_size)
    else:
        dataset = SyntheticTokenDataset(cfg.vocab_size, args.seq_len)

    sampler = DistributedSampler(dataset, shuffle=True) if WORLD_SIZE > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    grad_accum = getattr(args, "grad_accum", 1)
    effective_batch = args.batch_size * WORLD_SIZE * grad_accum
    log_main(f"Effective batch size: {effective_batch} "
             f"({args.batch_size} x {WORLD_SIZE} GPUs x {grad_accum} accum)")

    total_steps = cfg.dsa_warmup_steps + cfg.dsa_sparse_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )
    scheduler = build_cosine_scheduler(optimizer, warmup_steps=min(100, total_steps // 10), total_steps=total_steps)

    out_dir = Path(args.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        log_main(f"Outputs -> {out_dir}")

    history: List[Dict] = []
    global_step = 0
    data_iter = _infinite(loader, sampler)

    if cfg.dsa_warmup_steps > 0 and cfg.use_dsa:
        log_main(f"\n{'='*60}")
        log_main(f"Stage 1: Dense warm-up  ({cfg.dsa_warmup_steps} steps)")
        log_main(f"  Freeze model; train Lightning Indexer via KL loss")
        log_main(f"{'='*60}")

        _freeze_for_warmup(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_main(f"  Trainable params: {n_trainable / 1e3:.1f}K (indexer only)")

        t0 = time.perf_counter()
        for warmup_i in range(cfg.dsa_warmup_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum_kl = 0.0

            for micro in range(grad_accum):
                input_ids = next(data_iter)["input_ids"].to(device)
                is_last_micro = (micro == grad_accum - 1)
                ctx = _no_sync_ctx(model, is_last_micro)
                with ctx:
                    out = model(
                        input_ids=input_ids,
                        labels=None,
                        dsa_mode="dense",
                        collect_dsa_losses=True,
                        use_cache=False,
                    )
                    kl = out.dsa_kl_loss
                    if kl is None:
                        raise RuntimeError("No DSA KL loss::: is use_dsa=True in the config?")
                    (kl / grad_accum).backward()
                    accum_kl += kl.detach().item() / grad_accum

            kl_scalar = all_reduce_mean(
                torch.tensor(accum_kl, device=device)
            ).item()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip
            )
            optimizer.step()
            scheduler.step()

            global_step += 1
            entry = {
                "stage": "warmup",
                "step": global_step,
                "kl_loss": kl_scalar,
                "lr": scheduler.get_last_lr()[0],
            }
            history.append(entry)

            if global_step % args.log_interval == 0 and is_main():
                elapsed = time.perf_counter() - t0
                log_main(
                    f"  step {global_step:5d}/{cfg.dsa_warmup_steps}  "
                    f"kl={kl_scalar:.4f}  "
                    f"lr={entry['lr']:.2e}  "
                    f"({elapsed:.1f}s)"
                )

        _log_stage_summary("Warm-up", history, "warmup")

    log_main(f"\n{'='*60}")
    log_main(f"Stage 2: Sparse training  ({cfg.dsa_sparse_steps} steps)")
    dsa_label = f"DSA top-k={cfg.dsa_top_k}" if cfg.use_dsa else "Dense (baseline)"
    log_main(f"  {dsa_label}; LM loss + {cfg.dsa_kl_loss_weight}*KL loss")
    log_main(f"{'='*60}")

    _unfreeze_all(model)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_main(f"  Trainable params: {n_trainable / 1e6:.2f}M")

    t0 = time.perf_counter()
    dsa_mode_str = "sparse" if cfg.use_dsa else "dense"
    for step_i in range(cfg.dsa_sparse_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_lm = 0.0
        accum_kl = 0.0

        for micro in range(grad_accum):
            input_ids = next(data_iter)["input_ids"].to(device)
            is_last_micro = (micro == grad_accum - 1)
            ctx = _no_sync_ctx(model, is_last_micro)
            with ctx:
                out = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    dsa_mode=dsa_mode_str,
                    collect_dsa_losses=cfg.use_dsa,
                    use_cache=False,
                )
                lm = out.loss
                kl = out.dsa_kl_loss if out.dsa_kl_loss is not None else torch.zeros(1, device=device)
                total = (lm + cfg.dsa_kl_loss_weight * kl) / grad_accum
                total.backward()
                accum_lm += lm.detach().item() / grad_accum
                accum_kl += kl.detach().item() / grad_accum

        lm_scalar = all_reduce_mean(torch.tensor(accum_lm, device=device)).item()
        kl_scalar = all_reduce_mean(torch.tensor(accum_kl, device=device)).item() if cfg.use_dsa else 0.0

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        global_step += 1
        entry = {
            "stage": "sparse",
            "step": global_step,
            "lm_loss": lm_scalar,
            "kl_loss": kl_scalar,
            "loss": lm_scalar + cfg.dsa_kl_loss_weight * kl_scalar,
            "ppl": perplexity(lm_scalar),
            "grad_norm": float(grad_norm),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(entry)

        if global_step % args.log_interval == 0 and is_main():
            elapsed = time.perf_counter() - t0
            log_main(
                f"  step {step_i+1:5d}/{cfg.dsa_sparse_steps}  "
                f"loss={lm_scalar:.4f}  ppl={entry['ppl']:.2f}  "
                f"kl={kl_scalar:.4f}  "
                f"grad_norm={entry['grad_norm']:.2f}  "
                f"lr={entry['lr']:.2e}  "
                f"({elapsed:.1f}s)"
            )

        # checpoint all ranks must participate (FSDP collective).
        if args.save_interval > 0 and global_step % args.save_interval == 0:
            _save_checkpoint(model, optimizer, scheduler, global_step, out_dir)

    _log_stage_summary("Sparse training", history, "sparse")

    # last check point
    _save_checkpoint(model, optimizer, scheduler, global_step, out_dir, final=True)
    if is_main():
        metrics_path = out_dir / "metrics.json"
        metrics_path.write_text(json.dumps(history, indent=2))
        log_main(f"\nMetrics saved -> {metrics_path}")

    return history


def _demo_config():
    """Tiny model config for a fast demo that fits on any hardware."""
    
    return DeepSeekV32Config(
        vocab_size=4096,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=2048,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        moe_intermediate_size=128,
        first_k_dense_replace=1,
        routed_scaling_factor=1.0,
        use_dsa=True,
        dsa_top_k=64,
        dsa_n_indexer_heads=4,
        dsa_indexer_dim=32,
        dsa_warmup_steps=100,
        dsa_sparse_steps=300,
        dsa_kl_loss_weight=0.1,
    )


def _resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _infinite(loader: DataLoader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def _no_sync_ctx(model: nn.Module, is_last_micro: bool):
    """Return a context manager that suppresses gradient sync for all but the last micro-step.

    Works with FSDP, DDP, and plain modules.
    """
    from contextlib import nullcontext
    if is_last_micro:
        return nullcontext()
    if hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def _freeze_for_warmup(model: nn.Module) -> None:
    """Freeze everything except Lightning Indexer parameters (Stage 1)."""
    for name, param in model.named_parameters():
        param.requires_grad = "indexer" in name


def _unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    step: int,
    out_dir: Path,
    final: bool = False,
) -> None:
    """Save a checkpoint.

    FSDP note: gathering the full state dict is a collective: ALL ranks must
    call this function.  Only rank-0 writes to disk.
    """
    tag = "final" if final else f"step_{step:06d}"
    ckpt_dir = out_dir / f"checkpoint-{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # again not a big fan but
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if isinstance(model, FSDP):
            # All ranks must participate in this collective.
            try:
                from torch.distributed.checkpoint.state_dict import (
                    get_model_state_dict,
                    StateDictOptions,
                )
                opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
                state = get_model_state_dict(model, options=opts)
            except Exception:
                # Fallback: sharded state dict per rank (not cross-rank portable).
                state = model.state_dict()
        else:
            raw = model.module if hasattr(model, "module") else model
            state = raw.state_dict()
    except Exception:
        raw = model.module if hasattr(model, "module") else model
        state = raw.state_dict()

    # Only rank 0 writes files other ranks already contributed to the collective.
    if is_main():
        torch.save(
            {
                "step": step,
                "model_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            ckpt_dir / "checkpoint.pt",
        )
        log_main(f"Checkpoint saved -> {ckpt_dir}")


def _log_stage_summary(label: str, history: List[Dict], stage: str) -> None:
    rows = [h for h in history if h["stage"] == stage]
    if not rows or not is_main():
        return
    first = rows[0]
    last = rows[-1]
    parts = [f"\n{label} summary ({len(rows)} steps)"]
    if "kl_loss" in first:
        parts.append(f"  KL loss:  {first['kl_loss']:.4f} -> {last['kl_loss']:.4f}")
    if "lm_loss" in first:
        parts.append(f"  LM loss:  {first['lm_loss']:.4f} -> {last['lm_loss']:.4f}")
    if "ppl" in first:
        parts.append(f"  PPL:      {first['ppl']:.2f} -> {last['ppl']:.2f}")
    log_main("\n".join(parts))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("Model")
    g.add_argument("--config", default="configs/deepseek_v3_2_small.json",
                   help="Path to a DeepSeekV32Config JSON file.")
    g.add_argument("--demo", action="store_true",
                   help="Use a built-in tiny config for a quick end-to-end demo.")

    g = p.add_argument_group("Data")
    g.add_argument("--dataset", choices=["synthetic", "wikitext2"], default="synthetic",
                   help="synthetic: random tokens (no download).  "
                        "wikitext2: requires `datasets` and internet.")
    g.add_argument("--seq-len", type=int, default=128, metavar="N",
                   help="Token sequence length per sample (default: 128).")
    g.add_argument("--batch-size", type=int, default=4, metavar="N",
                   help="Per-GPU batch size (default: 4).")

    g = p.add_argument_group("Training")
    g.add_argument("--warmup-steps", type=int, default=None, metavar="N",
                   help="Override config dsa_warmup_steps.")
    g.add_argument("--sparse-steps", type=int, default=None, metavar="N",
                   help="Override config dsa_sparse_steps.")
    g.add_argument("--lr", type=float, default=3e-4, metavar="LR",
                   help="Peak learning rate (default: 3e-4).")
    g.add_argument("--kl-weight", type=float, default=None, metavar="lambda",
                   help="Override config dsa_kl_loss_weight.")
    g.add_argument("--grad-clip", type=float, default=1.0, metavar="C",
                   help="Gradient norm clip value (default: 1.0).")
    g.add_argument("--grad-accum", type=int, default=1, metavar="N",
                   help="Gradient accumulation steps. "
                        "Effective batch = batch-size x GPUs x grad-accum (default: 1).")
    g.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                   help="Compute dtype for single-GPU / CPU. FSDP always uses bf16.")
    g.add_argument("--compare", action="store_true",
                   help="Run dense baseline first, then DSA, and print a side-by-side "
                        "comparison. Requires --config pointing to a DSA config.")

    g = p.add_argument_group("I/O")
    g.add_argument("--output-dir", default="./output", metavar="DIR",
                   help="Directory for checkpoints and metrics (default: ./output).")
    g.add_argument("--log-interval", type=int, default=10, metavar="N",
                   help="Log every N steps (default: 10).")
    g.add_argument("--save-interval", type=int, default=0, metavar="N",
                   help="Save a checkpoint every N steps; 0 = only at end (default: 0).")

    return p


def _final_ppl(history: List[Dict]) -> Optional[float]:
    rows = [h for h in history if h["stage"] == "sparse" and "ppl" in h]
    # smooth over last 10 steps to reduce noise.
    if not rows:
        return None
    tail = rows[-min(10, len(rows)):]
    return sum(r["ppl"] for r in tail) / len(tail)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # auto switch to fp32 on CPU.
    if not torch.cuda.is_available() and args.dtype == "bf16":
        args.dtype = "fp32"
        # maybe print

    setup_distributed()
    try:
        if getattr(args, "compare", False):
            # ehhh
            import copy, json as _json

            with open(args.config) as f:
                base_dict = _json.load(f)

            results: Dict[str, Optional[float]] = {}

            for label, overrides in [
                ("Dense (baseline)", {"use_dsa": False, "dsa_warmup_steps": 0}),
                ("DSA  (sparse attn)", {}),
            ]:
                log_main(f"\n{'#'*60}")
                log_main(f"  Running: {label}")
                log_main(f"{'#'*60}")

                run_dict = {**base_dict, **overrides}
                tmp_cfg = Path(args.output_dir) / f"_tmp_{label.split()[0].lower()}.json"
                if is_main():
                    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
                    tmp_cfg.write_text(_json.dumps(run_dict))
                if dist.is_initialized():
                    dist.barrier()

                run_args = copy.copy(args)
                run_args.config = str(tmp_cfg)
                run_args.compare = False
                run_args.output_dir = str(Path(args.output_dir) / label.split()[0].lower())

                history = run_training(run_args)
                results[label] = _final_ppl(history)

            if is_main():
                print(f"\n{'='*55}")
                print(f"  Comparison (avg of last 10 sparse steps)")
                print(f"{'='*55}")
                for label, ppl_val in results.items():
                    print(f"  {label:<25s}  PPL = {ppl_val:.2f}" if ppl_val else f"  {label}")
                delta = None
                vals = list(results.values())
                if len(vals) == 2 and vals[0] and vals[1]:
                    delta = vals[1] - vals[0]
                    print(f"\n  Delta PPL (DSA - Dense) = {delta:+.2f}  "
                          f"({'within noise' if abs(delta) < 5 else 'significant'})")
                print(f"{'='*55}\n")
        else:
            history = run_training(args)
            if is_main() and history:
                sparse_rows = [h for h in history if h["stage"] == "sparse"]
                if sparse_rows:
                    final = sparse_rows[-1]
                    print(f"\n{'='*50}")
                    print(f"  Final LM loss : {final['lm_loss']:.4f}")
                    print(f"  Final PPL     : {final['ppl']:.2f}")
                    if final.get("kl_loss"):
                        print(f"  Final KL loss : {final['kl_loss']:.4f}")
                    print(f"  Checkpoints   : {args.output_dir}")
                    print(f"{'='*50}\n")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
