# coding=utf-8
"""Two-stage DSA trainer (dense indexer warm-up, then sparse training)."""

from __future__ import annotations

import itertools
from typing import Optional

import torch


class DSATrainer:
    """Implements the two-stage DeepSeek V3.2 indexer training recipe.

    Stage 1 (dense warm-up): freeze the whole model except the Lightning Indexer.
        Run a *dense* attention forward and align the indexer distribution to the
        (detached) aggregated attention distribution via a KL loss.

    Stage 2 (sparse training): unfreeze all parameters. Run a *sparse* (top-k)
        forward producing the LM loss, plus a scaled KL alignment loss restricted
        to the selected tokens.
    """

    def __init__(self, model, config, optimizer, scheduler=None):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.kl_weight = config.dsa_kl_loss_weight
        self.global_step = 0

    # parameter freezing
    def freeze_for_warmup(self) -> None:
        """Stage 1: train only the Lightning Indexer parameters."""
        for name, param in self.model.named_parameters():
            param.requires_grad = "indexer" in name

    def unfreeze_all(self) -> None:
        """Stage 2: train all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    # single steps
    def _to_device(self, t):
        if t is None:
            return None
        return t.to(next(self.model.parameters()).device)

    def warmup_step(self, input_ids, attention_mask=None) -> dict:
        """One dense warm-up step: indexer KL loss only (model frozen)."""
        self.model.train()
        input_ids = self._to_device(input_ids)
        attention_mask = self._to_device(attention_mask)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dsa_mode="dense",
            collect_dsa_losses=True,
            use_cache=False,
        )
        loss = out.dsa_kl_loss
        if loss is None:
            raise RuntimeError("No DSA KL loss produced; is use_dsa enabled?")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return {"kl_loss": float(loss.detach()), "stage": "warmup", "step": self.global_step}

    def sparse_step(self, input_ids, attention_mask=None) -> dict:
        """One sparse step: LM loss + scaled indexer KL loss (all params trained)."""
        self.model.train()
        input_ids = self._to_device(input_ids)
        attention_mask = self._to_device(attention_mask)

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            dsa_mode="sparse",
            collect_dsa_losses=True,
            use_cache=False,
        )
        lm_loss = out.loss
        kl_loss = out.dsa_kl_loss if out.dsa_kl_loss is not None else torch.zeros_like(lm_loss)
        loss = lm_loss + self.kl_weight * kl_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return {
            "loss": float(loss.detach()),
            "lm_loss": float(lm_loss.detach()),
            "kl_loss": float(kl_loss.detach()),
            "stage": "sparse",
            "step": self.global_step,
        }

    # full schedule
    def train(self, train_dataloader, num_warmup_steps: Optional[int] = None, num_sparse_steps: Optional[int] = None):
        """Run the full two-stage schedule, returning a list of per-step metric dicts."""
        num_warmup_steps = num_warmup_steps if num_warmup_steps is not None else self.config.dsa_warmup_steps
        num_sparse_steps = num_sparse_steps if num_sparse_steps is not None else self.config.dsa_sparse_steps

        history = []
        loader = itertools.cycle(train_dataloader)

        # dense warmup
        if num_warmup_steps > 0:
            self.freeze_for_warmup()
            for _ in range(num_warmup_steps):
                batch = next(loader)
                history.append(self.warmup_step(batch["input_ids"], batch.get("attention_mask")))

        # sparse training
        self.unfreeze_all()
        for _ in range(num_sparse_steps):
            batch = next(loader)
            history.append(self.sparse_step(batch["input_ids"], batch.get("attention_mask")))

        return history
