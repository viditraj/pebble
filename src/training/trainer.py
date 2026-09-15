"""
Training loop for Pebble.

Handles:
- Mixed precision (BF16) training
- Gradient accumulation
- Learning rate scheduling (warmup + cosine decay)
- Gradient clipping
- Checkpointing (save/resume)
- Validation
- Logging
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.scheduler import get_lr


class RandomChunkSampler(Sampler):
    """
    Memory-efficient random sampler for huge datasets.
    
    Instead of creating a full permutation of 451M indices (3.6 GB RAM),
    this generates random indices on the fly. Each epoch samples
    `num_samples` random positions from the dataset.
    """
    
    def __init__(self, data_source, num_samples: int = 500_000):
        self.data_len = len(data_source)
        self.num_samples = min(num_samples, self.data_len)
    
    def __iter__(self):
        # Generate random indices in small batches — never holds all in RAM
        for _ in range(self.num_samples):
            yield torch.randint(0, self.data_len, (1,)).item()
    
    def __len__(self):
        return self.num_samples


class Trainer:
    """
    Training engine for Pebble.
    
    Usage:
        trainer = Trainer(model, train_dataset, val_dataset, config)
        trainer.train()
    """
    
    def __init__(self, model, train_dataset, val_dataset, config: dict):
        """
        Args:
            model: The Pebble model (nn.Module).
            train_dataset: PebbleDataset for training.
            val_dataset: PebbleDataset for validation.
            config: Training config dict (from YAML).
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Extract config values
        tc = config["training"]
        self.micro_batch_size = tc["micro_batch_size"]
        self.grad_accum_steps = tc["gradient_accumulation_steps"]
        self.max_lr = tc["max_lr"]
        self.min_lr = tc["min_lr"]
        self.warmup_steps = tc["warmup_steps"]
        self.total_steps = tc["total_steps"]
        self.weight_decay = tc["weight_decay"]
        self.grad_clip = tc["grad_clip"]
        self.save_every = tc["save_every"]
        self.eval_every = tc["eval_every"]
        self.log_every = tc["log_every"]
        
        self.effective_batch_size = self.micro_batch_size * self.grad_accum_steps
        self.tokens_per_step = self.effective_batch_size * config["data"]["seq_len"]
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Optimizer — AdamW with weight decay only on 2D parameters
        # (weight matrices get decay, biases and norms don't)
        self.optimizer = self._create_optimizer()
        
        # Mixed precision
        self.use_bf16 = (tc["precision"] == "bf16" and 
                         torch.cuda.is_available() and 
                         torch.cuda.is_bf16_supported())
        self.autocast_dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        
        # Data loaders
        # Use RandomChunkSampler to avoid 3.6GB RAM for shuffling 451M indices
        # num_samples per "epoch" = enough for all steps in one pass
        samples_per_epoch = self.total_steps * self.grad_accum_steps * self.micro_batch_size
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            sampler=RandomChunkSampler(train_dataset, num_samples=samples_per_epoch),
            num_workers=0,       # no extra worker processes = less RAM
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
        
        # Training state (for checkpointing/resume)
        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")
        
        # Logging
        self.log_file = os.path.join(config.get("checkpoint_dir", "checkpoints"), "train_log.jsonl")
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def _create_optimizer(self):
        """
        Create AdamW optimizer with parameter groups.
        
        Key insight: weight decay should only apply to weight matrices,
        NOT to biases, layer norms, or embedding layers.
        Why? Biases and norms have few parameters — regularizing them
        hurts more than it helps. Weight matrices are where overfitting
        happens.
        """
        # Separate parameters into two groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't decay 1D params (biases, norms) or embeddings
            if param.dim() < 2:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.max_lr,
            betas=(0.9, 0.95),    # LLaMA-style: slightly less momentum on variance
            eps=1e-8,
        )
        
        n_decay = sum(p.numel() for p in decay_params)
        n_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"  Optimizer: AdamW")
        print(f"    Decay params:    {n_decay:,} ({len(decay_params)} tensors)")
        print(f"    No-decay params: {n_no_decay:,} ({len(no_decay_params)} tensors)")
        
        return optimizer
    
    def _update_lr(self):
        """Set learning rate for current step."""
        lr = get_lr(self.step, self.max_lr, self.min_lr, 
                    self.warmup_steps, self.total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr
    
    @torch.no_grad()
    def validate(self):
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        max_val_batches = 100  # Don't validate on entire set each time
        
        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                logits = self.model(x)
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
            
            total_loss += loss.item()
            n_batches += 1
            if n_batches >= max_val_batches:
                break
        
        self.model.train()
        return total_loss / max(n_batches, 1)
    
    def save_checkpoint(self, path=None):
        """Save model, optimizer, and training state."""
        ckpt_dir = self.config.get("checkpoint_dir", "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        if path is None:
            path = os.path.join(ckpt_dir, f"step_{self.step}.pt")
        
        checkpoint = {
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)
        
        # Also save as "latest" for easy resume
        latest_path = os.path.join(ckpt_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, path=None):
        """Resume training from a checkpoint."""
        if path is None:
            ckpt_dir = self.config.get("checkpoint_dir", "checkpoints")
            path = os.path.join(ckpt_dir, "latest.pt")
        
        if not os.path.exists(path):
            print(f"  No checkpoint found at {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.tokens_seen = checkpoint["tokens_seen"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        
        print(f"  Resumed from step {self.step} ({self.tokens_seen:,} tokens seen)")
        return True
    
    def _log(self, metrics: dict):
        """Append metrics to the log file."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"  PEBBLE TRAINING")
        print(f"{'='*60}")
        print(f"  Model params:     {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Micro batch size: {self.micro_batch_size}")
        print(f"  Grad accum steps: {self.grad_accum_steps}")
        print(f"  Effective batch:  {self.effective_batch_size}")
        print(f"  Tokens per step:  {self.tokens_per_step:,}")
        print(f"  Total steps:      {self.total_steps:,}")
        print(f"  Total tokens:     {self.total_steps * self.tokens_per_step:,}")
        print(f"  Precision:        {'BF16' if self.use_bf16 else 'FP32'}")
        print(f"  Device:           {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU:              {torch.cuda.get_device_name()}")
            print(f"  GPU Memory:       {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"{'='*60}\n")
        
        self.model.train()
        train_iter = iter(self.train_loader)
        
        # Running loss for logging
        running_loss = 0.0
        running_count = 0
        
        t0 = time.time()
        step_t0 = time.time()
        
        while self.step < self.total_steps:
            # Update learning rate
            lr = self._update_lr()
            
            # === Gradient accumulation loop ===
            self.optimizer.zero_grad()
            step_loss = 0.0
            
            for micro_step in range(self.grad_accum_steps):
                # Get next batch (restart iterator if exhausted = new epoch)
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass with mixed precision
                with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                    logits = self.model(x)
                    loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)),
                        y.view(-1)
                    )
                    # Divide by accumulation steps so the total gradient
                    # is the AVERAGE over all micro-batches
                    loss = loss / self.grad_accum_steps
                
                # Backward pass (accumulates gradients)
                loss.backward()
                
                step_loss += loss.item()
            
            # Gradient clipping (prevents exploding gradients)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Update counters
            self.step += 1
            self.tokens_seen += self.tokens_per_step
            running_loss += step_loss
            running_count += 1
            
            # === Logging ===
            if self.step % self.log_every == 0:
                step_time = time.time() - step_t0
                tokens_per_sec = self.tokens_per_step * self.log_every / step_time
                avg_loss = running_loss / running_count
                
                gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                
                print(f"  step {self.step:>6} | "
                      f"loss {avg_loss:.4f} | "
                      f"lr {lr:.2e} | "
                      f"grad_norm {grad_norm:.2f} | "
                      f"tok/s {tokens_per_sec:,.0f} | "
                      f"GPU {gpu_mem:.1f}GB")
                
                self._log({
                    "step": self.step,
                    "loss": round(avg_loss, 4),
                    "lr": lr,
                    "grad_norm": round(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, 4),
                    "tokens_per_sec": round(tokens_per_sec),
                    "tokens_seen": self.tokens_seen,
                    "gpu_mem_gb": round(gpu_mem, 2),
                })
                
                running_loss = 0.0
                running_count = 0
                step_t0 = time.time()
            
            # === Validation ===
            if self.step % self.eval_every == 0:
                val_loss = self.validate()
                print(f"  *** val_loss {val_loss:.4f} (best: {self.best_val_loss:.4f}) ***")
                
                self._log({"step": self.step, "val_loss": round(val_loss, 4)})
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(self.config.get("checkpoint_dir", "checkpoints"), "best.pt")
                    )
            
            # === Checkpointing ===
            if self.step % self.save_every == 0:
                self.save_checkpoint()
        
        # Final save
        total_time = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Steps: {self.step:,}")
        print(f"  Tokens: {self.tokens_seen:,}")
        print(f"  Time: {total_time/3600:.1f} hours")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
        self.save_checkpoint()