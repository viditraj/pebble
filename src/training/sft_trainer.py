"""
Supervised Fine-Tuning trainer for Pebble.

Differences from pretraining trainer:
- Uses loss masking (only compute loss on assistant tokens)
- Lower learning rate to avoid catastrophic forgetting
- Loads from a pretrained checkpoint
- Uses SFTDataset instead of PebbleDataset
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.scheduler import get_lr


class SFTTrainer:
    """Fine-tuning trainer with loss masking."""
    
    def __init__(self, model, train_dataset, val_dataset, config: dict):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        tc = config["sft"]
        self.micro_batch_size = tc["micro_batch_size"]
        self.grad_accum_steps = tc["gradient_accumulation_steps"]
        self.max_lr = tc["learning_rate"]
        self.min_lr = tc["learning_rate"] / 10  # decay to 1/10
        self.warmup_steps = tc["warmup_steps"]
        self.total_steps = tc["total_steps"]
        self.weight_decay = tc["weight_decay"]
        self.grad_clip = tc["grad_clip"]
        self.save_every = tc.get("save_every", 500)
        self.eval_every = tc.get("eval_every", 200)
        self.log_every = tc.get("log_every", 10)
        
        self.effective_batch_size = self.micro_batch_size * self.grad_accum_steps
        self.seq_len = config["data"]["seq_len"]
        self.tokens_per_step = self.effective_batch_size * self.seq_len
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Optimizer — same structure as pretraining
        self.optimizer = self._create_optimizer()
        
        # BF16
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        
        # Data loaders — no shuffle needed, we sample randomly from conversations
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.micro_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            )
        else:
            self.val_loader = None
        
        self.step = 0
        self.best_val_loss = float("inf")
        self.log_file = os.path.join(
            config.get("checkpoint_dir", "checkpoints"), "sft_log.jsonl"
        )
    
    def _create_optimizer(self):
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.max_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        return optimizer
    
    def _update_lr(self):
        lr = get_lr(self.step, self.max_lr, self.min_lr,
                    self.warmup_steps, self.total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr
    
    def _masked_loss(self, logits, targets, mask):
        """
        Compute cross-entropy loss only on masked positions.
        
        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
            mask: (batch, seq_len) — True where loss should be computed
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Flatten for cross_entropy
        loss_per_token = nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            targets.view(-1),
            reduction="none"
        )  # (batch * seq_len,)
        
        loss_per_token = loss_per_token.view(batch_size, seq_len)
        
        # Apply mask and average
        masked_loss = (loss_per_token * mask.float()).sum()
        n_masked = mask.sum()
        
        if n_masked == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return masked_loss / n_masked
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        total_masked = 0
        n_batches = 0
        
        for x, y, mask in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)
            
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                logits = self.model(x)
                loss = self._masked_loss(logits, y, mask)
            
            total_loss += loss.item() * mask.sum().item()
            total_masked += mask.sum().item()
            n_batches += 1
            if n_batches >= 50:
                break
        
        self.model.train()
        return total_loss / max(total_masked, 1)
    
    def save_checkpoint(self, path=None):
        ckpt_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        if path is None:
            path = os.path.join(ckpt_dir, f"sft_step_{self.step}.pt")
        
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, path)
        
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }, os.path.join(ckpt_dir, "sft_latest.pt"))
        
        print(f"  SFT checkpoint saved: {path}")
    
    def _log(self, metrics):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def train(self):
        print(f"\n{'='*60}")
        print(f"  PEBBLE SFT TRAINING")
        print(f"{'='*60}")
        print(f"  Model params:     {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Micro batch size: {self.micro_batch_size}")
        print(f"  Grad accum steps: {self.grad_accum_steps}")
        print(f"  Effective batch:  {self.effective_batch_size}")
        print(f"  Total steps:      {self.total_steps:,}")
        print(f"  Learning rate:    {self.max_lr}")
        print(f"  Precision:        {'BF16' if self.use_bf16 else 'FP32'}")
        print(f"{'='*60}\n")
        
        self.model.train()
        train_iter = iter(self.train_loader)
        
        running_loss = 0.0
        running_count = 0
        t0 = time.time()
        step_t0 = time.time()
        
        while self.step < self.total_steps:
            lr = self._update_lr()
            self.optimizer.zero_grad()
            step_loss = 0.0
            
            for _ in range(self.grad_accum_steps):
                try:
                    x, y, mask = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y, mask = next(train_iter)
                
                x = x.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)
                
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    logits = self.model(x)
                    loss = self._masked_loss(logits, y, mask)
                    loss = loss / self.grad_accum_steps
                
                loss.backward()
                step_loss += loss.item()
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()
            
            self.step += 1
            running_loss += step_loss
            running_count += 1
            
            if self.step % self.log_every == 0:
                elapsed = time.time() - step_t0
                avg_loss = running_loss / running_count
                
                print(f"  step {self.step:>5} | "
                      f"loss {avg_loss:.4f} | "
                      f"lr {lr:.2e} | "
                      f"grad_norm {grad_norm:.2f}")
                
                self._log({
                    "step": self.step, "loss": round(avg_loss, 4),
                    "lr": lr, "grad_norm": round(
                        grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4
                    ),
                })
                
                running_loss = 0.0
                running_count = 0
                step_t0 = time.time()
            
            if self.step % self.eval_every == 0:
                val_loss = self.validate()
                if val_loss is not None:
                    print(f"  *** val_loss {val_loss:.4f} ***")
                    self._log({"step": self.step, "val_loss": round(val_loss, 4)})
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(
                            os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "sft_best.pt")
                        )
            
            if self.step % self.save_every == 0:
                self.save_checkpoint()
        
        total_time = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  SFT complete! Steps: {self.step}, Time: {total_time/60:.1f}min")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        self.save_checkpoint()