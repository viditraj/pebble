"""
Supervised Fine-Tuning for Pebble.

Loads pretrained checkpoint and fine-tunes on tool-calling conversations.

Usage:
    python scripts/sft_train.py
    python scripts/sft_train.py --test-steps 50    # quick test
"""

import os
import sys
import yaml
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from model.pebble import Pebble, PebbleConfig
from data.sft_dataset import SFTDataset
from training.sft_trainer import SFTTrainer
from tokenizer.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT for Pebble")
    parser.add_argument("--config", default=os.path.join(PROJECT_DIR, "configs", "pebble_25m.yaml"))
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_DIR, "checkpoints", "best.pt"),
                        help="Pretrained checkpoint to load")
    parser.add_argument("--test-steps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.test_steps:
        config["sft"]["total_steps"] = args.test_steps
        config["sft"]["eval_every"] = min(25, args.test_steps)
        config["sft"]["save_every"] = args.test_steps
        config["sft"]["log_every"] = 5
        print(f"TEST MODE: {args.test_steps} steps\n")

    config["checkpoint_dir"] = os.path.join(PROJECT_DIR, "checkpoints")

    # Load tokenizer for special token IDs
    tokenizer_path = os.path.join(PROJECT_DIR, "checkpoints", "tokenizer.json")
    tokenizer = BPETokenizer.load(tokenizer_path)
    sp = tokenizer.special_tokens
    print(f"Tokenizer loaded: {len(tokenizer)} tokens")

    # Create model
    mc = config["model"]
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

    # Load pretrained weights
    print(f"\nLoading pretrained checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    pretrain_step = ckpt.get("step", "?")
    pretrain_val = ckpt.get("best_val_loss", "?")
    print(f"  Pretrained at step {pretrain_step}, val_loss={pretrain_val}")

    # Create SFT dataset
    sft_data_path = os.path.join(PROJECT_DIR, "data", "processed", "sft_tool.bin")
    seq_len = config["data"]["seq_len"]

    # Use 95% for train, 5% for val (split by conversation count)
    full_dataset = SFTDataset(
        sft_data_path, seq_len=seq_len,
        assistant_token_id=sp["<|assistant|>"],
        end_token_id=sp["<|end|>"],
        tool_result_start_id=sp["<|tool_result_start|>"],
        tool_result_end_id=sp["<|tool_result_end|>"],
        bos_token_id=sp["<|bos|>"],
    )

    # Split into train/val
    n_total = len(full_dataset)
    n_val = max(int(n_total * 0.05), 100)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"SFT data: {n_train:,} train, {n_val:,} val conversations")

    # Create trainer and train
    trainer = SFTTrainer(model, train_dataset, val_dataset, config)
    trainer.train()


if __name__ == "__main__":
    main()