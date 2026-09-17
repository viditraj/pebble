"""
Prepare training data for Pebble-70M.

Uses OpenWebText (~1B tokens) for pretraining.
Memory-safe: streams data and writes in chunks.
Resumable: saves progress every chunk.

Usage (from project root):
    python scripts/prepare_data_70m.py
"""

import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from tokenizer.bpe import BPETokenizer

# --- Worker globals (one per process) ---
_tokenizer = None
_bos_id = None
_eos_id = None


def _init_worker(tokenizer_path):
    global _tokenizer, _bos_id, _eos_id
    _tokenizer = BPETokenizer.load(tokenizer_path)
    _bos_id = _tokenizer.special_tokens.get("<|bos|>")
    _eos_id = _tokenizer.special_tokens.get("<|eos|>")


def _tokenize_doc(text):
    lines = text.split("\n")
    lines = [" ".join(line.split()) for line in lines]
    text = "\n".join(lines).replace("\x00", "").strip()
    if len(text) < 20:
        return None
    tokens = []
    if _bos_id is not None:
        tokens.append(_bos_id)
    tokens.extend(_tokenizer.encode(text))
    if _eos_id is not None:
        tokens.append(_eos_id)
    return tokens


def tokenize_dataset(ds_iter, total_docs, raw_path, progress_path,
                     tokenizer_path, n_workers, chunk_size, label,
                     max_tokens=None):
    """
    Tokenize a dataset iterator and append to raw_path.
    Returns (docs_processed, tokens_written).
    """
    # Check for resume
    docs_done = 0
    tokens_written = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        docs_done = progress["docs_done"]
        tokens_written = progress["tokens_written"]
        print(f"  [{label}] RESUMING from: {docs_done:,} docs, {tokens_written:,} tokens")
        if docs_done >= total_docs or (max_tokens and tokens_written >= max_tokens):
            return docs_done, tokens_written

    remaining = total_docs - docs_done
    print(f"\n  [{label}] Tokenizing {remaining:,} docs with {n_workers} workers...")
    t0 = time.time()

    # Skip already-processed docs
    if docs_done > 0:
        print(f"  [{label}] Skipping {docs_done:,} already-processed docs...")
        for _ in range(docs_done):
            next(ds_iter)

    chunk_texts = []
    current_doc = docs_done

    with Pool(n_workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        for example in ds_iter:
            chunk_texts.append(example["text"])
            current_doc += 1

            if len(chunk_texts) >= chunk_size:
                # Tokenize chunk
                results = pool.map(_tokenize_doc, chunk_texts)
                del chunk_texts
                chunk_texts = []

                chunk_tokens = []
                for tokens in results:
                    if tokens is not None:
                        chunk_tokens.extend(tokens)
                del results

                if chunk_tokens:
                    arr = np.array(chunk_tokens, dtype=np.uint16)
                    with open(raw_path, "ab") as f:
                        f.write(arr.tobytes())
                    tokens_written += len(chunk_tokens)
                    del arr
                del chunk_tokens

                docs_done = current_doc

                # Save progress
                tmp = progress_path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"docs_done": docs_done, "tokens_written": tokens_written}, f)
                os.replace(tmp, progress_path)

                elapsed = time.time() - t0
                rate = (docs_done - (total_docs - remaining)) / elapsed if elapsed > 0 else 0
                print(f"  [{label}] {docs_done:>10,}/{total_docs:,} | "
                      f"{tokens_written:>12,} tokens | "
                      f"{rate:.0f} docs/s")

                # Stop if we've reached max_tokens
                if max_tokens and tokens_written >= max_tokens:
                    print(f"  [{label}] Reached {max_tokens:,} token target, stopping.")
                    break

        # Process remaining docs in the last partial chunk
        if chunk_texts:
            results = pool.map(_tokenize_doc, chunk_texts)
            chunk_tokens = []
            for tokens in results:
                if tokens is not None:
                    chunk_tokens.extend(tokens)
            if chunk_tokens:
                arr = np.array(chunk_tokens, dtype=np.uint16)
                with open(raw_path, "ab") as f:
                    f.write(arr.tobytes())
                tokens_written += len(chunk_tokens)

            docs_done = current_doc
            tmp = progress_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"docs_done": docs_done, "tokens_written": tokens_written}, f)
            os.replace(tmp, progress_path)

    elapsed = time.time() - t0
    print(f"  [{label}] Done: {docs_done:,} docs, {tokens_written:,} tokens in {elapsed/60:.1f}min")
    return docs_done, tokens_written


def main():
    # === Paths ===
    tokenizer_path = os.path.join(PROJECT_DIR, "checkpoints", "tokenizer.json")
    output_dir = os.path.join(PROJECT_DIR, "data", "processed")
    raw_path = os.path.join(output_dir, "all_tokens_70m.bin")
    train_path = os.path.join(output_dir, "train_70m.bin")
    val_path = os.path.join(output_dir, "val_70m.bin")

    val_fraction = 0.005  # 0.5% for val (larger dataset)
    chunk_size = 10_000
    n_workers = 6  # L4 machine has 8 vCPUs, leave 2 for OS/IO

    os.makedirs(output_dir, exist_ok=True)

    # === Verify tokenizer ===
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"Tokenizer: vocab_size={len(tokenizer)}")

    # === Fresh start if no raw file ===
    if not os.path.exists(raw_path):
        with open(raw_path, "wb") as _:
            pass

    # === OpenWebText (~1B tokens) ===
    print("\n" + "=" * 60)
    print("  DATASET: OpenWebText")
    print("=" * 60)
    from datasets import load_dataset

    owt_progress = os.path.join(output_dir, "progress_owt_70m.json")
    owt_target_tokens = 1_000_000_000

    # Download the full dataset first (cached by HF), then slice
    # This is MUCH faster than streaming because HF downloads all
    # shards in parallel and caches them locally.
    print("  Downloading dataset (cached after first run)...")
    owt = load_dataset("Skylion007/openwebtext", split="train")
    owt_total = len(owt)
    # Take first 300K docs (or all if fewer) — should give ~1B tokens
    owt_max_docs = min(300_000, owt_total)
    print(f"  Total docs available: {owt_total:,}")
    print(f"  Using first {owt_max_docs:,} docs, target {owt_target_tokens:,} tokens")

    owt_docs, owt_tokens = tokenize_dataset(
        iter(owt), owt_max_docs, raw_path, owt_progress,
        tokenizer_path, n_workers, chunk_size, "OpenWebText",
        max_tokens=owt_target_tokens
    )
    del owt  # free memory

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {owt_tokens:,} tokens")
    print(f"{'=' * 60}")

    # === Split into train/val ===
    print(f"\nSplitting into train/val...")
    all_data = np.memmap(raw_path, dtype=np.uint16, mode="r")
    total_on_disk = len(all_data)

    n_val = int(total_on_disk * val_fraction)
    n_train = total_on_disk - n_val

    print(f"  Total:  {total_on_disk:,} tokens")
    print(f"  Train:  {n_train:,}")
    print(f"  Val:    {n_val:,}")

    # Write train.bin
    train_mm = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(n_train,))
    copy_chunk = 50_000_000
    for i in range(0, n_train, copy_chunk):
        end = min(i + copy_chunk, n_train)
        train_mm[i:end] = all_data[i:end]
    train_mm.flush()
    del train_mm

    # Write val.bin
    val_mm = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(n_val,))
    val_mm[:] = all_data[n_train:]
    val_mm.flush()
    del val_mm
    del all_data

    print(f"  {train_path}: {os.path.getsize(train_path)/1024**2:.1f} MB")
    print(f"  {val_path}: {os.path.getsize(val_path)/1024**2:.1f} MB")

    # === Verify ===
    print("\nVerification...")
    check = np.memmap(train_path, dtype=np.uint16, mode="r")
    sample = tokenizer.decode(check[:200].tolist())
    print(f"  First 200 tokens:\n  ---\n  {sample[:500]}\n  ---")
    del check

    # === Cleanup temp files ===
    for f in [raw_path, owt_progress]:
        if os.path.exists(f):
            os.remove(f)
    print("\nDone! Temp files cleaned up.")


if __name__ == "__main__":
    main()
