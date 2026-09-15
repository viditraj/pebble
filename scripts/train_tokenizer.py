"""
Train Pebble's BPE tokenizer on real text data.

Usage:
    python scripts/train_tokenizer.py --vocab-size 16384 --sample-size 100000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokenizer.bpe import BPETokenizer

# Pebble's special tokens
SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<|tool_result_start|>",
    "<|tool_result_end|>",
    "<|function_name|>",
    "<|parameters|>",
    "<|think_start|>",
    "<|think_end|>",
]


def main():
    parser = argparse.ArgumentParser(description="Train Pebble's BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--sample-size", type=int, default=100_000,
                        help="Number of stories to train on")
    parser.add_argument("--output", type=str, default="checkpoints/tokenizer.json")
    args = parser.parse_args()

    # === Step 1: Download data ===
    print("=== Downloading training data ===")
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories", split="train")

    n = min(args.sample_size, len(ds))
    texts = [ds[i]["text"] for i in range(n)]
    corpus = "\n\n".join(texts)

    print(f"  Stories: {n:,}")
    print(f"  Corpus: {len(corpus):,} chars ({len(corpus) / 1024 / 1024:.1f} MB)")

    # === Step 2: Train tokenizer ===
    print(f"\n=== Training tokenizer (vocab_size={args.vocab_size}) ===")
    tokenizer = BPETokenizer.train_fast(
        text=corpus,
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        verbose=True,
    )

    # === Step 3: Save ===
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer.save(args.output)
    print(f"\nTokenizer saved to {args.output}")

    # === Step 4: Quality checks ===
    print(f"\n=== Quality Checks ===")

    test_strings = [
        "Once upon a time, there was a little girl named Lily.",
        "The cat sat on the mat.",
        "Hello, world! How are you today?",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "The weather in Tokyo is 72 degrees and partly cloudy.",
        "What's 25 * 47?",
    ]

    for s in test_strings:
        ids = tokenizer.encode(s)
        decoded = tokenizer.decode(ids)
        ratio = len(s.encode("utf-8")) / len(ids)
        status = "PASS" if s == decoded else "FAIL"
        print(f"  [{status}] {ratio:.1f}x compression | {len(ids):>3} tokens | {s[:60]}")

    # Show some interesting learned tokens
    print(f"\n=== Sample Vocabulary ===")
    print(f"  First 10 merges (most common patterns):")
    for i in range(min(10, len(tokenizer.merges))):
        token_id = 256 + i
        token_bytes = tokenizer.vocab[token_id]
        token_str = token_bytes.decode("utf-8", errors="replace")
        print(f"    token {token_id}: {repr(token_str)}")

    print(f"\n  Last 10 merges (least common patterns):")
    for i in range(max(0, len(tokenizer.merges) - 10), len(tokenizer.merges)):
        token_id = 256 + i
        token_bytes = tokenizer.vocab[token_id]
        token_str = token_bytes.decode("utf-8", errors="replace")
        print(f"    token {token_id}: {repr(token_str)}")

    # Compression stats on the full corpus
    print(f"\n=== Corpus Compression ===")
    sample = corpus[:100_000]  # test on first 100K chars to avoid slow encode
    ids = tokenizer.encode(sample)
    print(f"  Original: {len(sample.encode('utf-8')):,} bytes")
    print(f"  Encoded:  {len(ids):,} tokens")
    print(f"  Compression: {len(sample.encode('utf-8')) / len(ids):.2f}x")
    print(f"  Avg token length: {len(sample.encode('utf-8')) / len(ids):.1f} bytes/token")


if __name__ == "__main__":
    main()