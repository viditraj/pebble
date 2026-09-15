"""
Train Pebble language model.

Usage:
    python scripts/train.py                           # train from scratch
    python scripts/train.py --resume                  # resume from checkpoint
    python scripts/train.py --test-steps 100          # quick test (100 steps)

Always run from project root: /home/vidit_singh/pebble
"""

import os
import sys
import yaml
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from model.pebble import Pebble
from data.dataset import PebbleDataset
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Pebble")
    parser.add_argument("--config", default=os.path.join(PROJECT_DIR, "configs", "pebble_25m.yaml"))
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--test-steps", type=int, default=None, help="Run only N steps (for testing)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override total_steps if test mode
    if args.test_steps:
        config["training"]["total_steps"] = args.test_steps
        config["training"]["eval_every"] = min(50, args.test_steps)
        config["training"]["save_every"] = args.test_steps
        config["training"]["log_every"] = 10
        print(f"TEST MODE: {args.test_steps} steps only\n")

    config["checkpoint_dir"] = os.path.join(PROJECT_DIR, "checkpoints")

    # Create model
    mc = config["model"]
    from model.pebble import PebbleConfig
    model_config = PebbleConfig(
        vocab_size=mc["vocab_size"],
        d_model=mc["d_model"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        n_kv_heads=mc["n_kv_heads"],
        d_ff=mc["d_ff"],
        max_seq_len=mc["max_seq_len"],
    )
    model = Pebble(model_config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {n_params:,} parameters")

    # Create datasets
    dc = config["data"]
    train_path = os.path.join(PROJECT_DIR, dc["train_path"])
    val_path = os.path.join(PROJECT_DIR, dc["val_path"])

    train_dataset = PebbleDataset(train_path, seq_len=dc["seq_len"])
    val_dataset = PebbleDataset(val_path, seq_len=dc["seq_len"])
    print(f"Train: {train_dataset}")
    print(f"Val:   {val_dataset}")

    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint()

    # Train!
    trainer.train()


if __name__ == "__main__":
    main()