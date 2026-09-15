"""
Prepare training data for Pebble (parallelized + resumable + memory-safe).

Memory-safe: never holds more than one chunk of tokens in memory.
Resumable: saves progress every chunk, survives crashes.

Usage (from project root):
    python scripts/prepare_data.py
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


def main():
    # === Paths (all absolute, works from any cwd) ===
    tokenizer_path = os.path.join(PROJECT_DIR, "checkpoints", "tokenizer.json")
    output_dir = os.path.join(PROJECT_DIR, "data", "processed")
    raw_path = os.path.join(output_dir, "all_tokens.bin")
    progress_path = os.path.join(output_dir, "progress.json")
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    val_fraction = 0.01
    chunk_size = 5_000       # smaller chunks = less memory per batch
    n_workers = 12           # half of 16 cores — leaves RAM for the main process

    os.makedirs(output_dir, exist_ok=True)

    # === Verify tokenizer ===
    tokenizer = BPETokenizer.load(tokenizer_path)
    bos_id = tokenizer.special_tokens.get("<|bos|>")
    eos_id = tokenizer.special_tokens.get("<|eos|>")
    print(f"Tokenizer: vocab_size={len(tokenizer)}, BOS={bos_id}, EOS={eos_id}")

    if bos_id != 16371:
        print(f"\n  WARNING: Expected BOS=16371, got BOS={bos_id}")
        print(f"  Make sure you're using the correct tokenizer!")
        return

    # === Load dataset (streaming = low memory) ===
    print("\nLoading TinyStories...")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train")
    total_docs = len(ds)
    print(f"  Total stories: {total_docs:,}")

    # === Check for resume ===
    docs_done = 0
    tokens_written = 0
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = json.load(f)
        docs_done = progress["docs_done"]
        tokens_written = progress["tokens_written"]
        print(f"  RESUMING from: {docs_done:,} docs, {tokens_written:,} tokens already written")
    else:
        # Fresh start
        with open(raw_path, "wb") as f:
            pass  # create empty file

    if docs_done >= total_docs:
        print("  All docs already tokenized, skipping to split step.")
    else:
        remaining = total_docs - docs_done
        print(f"\nTokenizing {remaining:,} remaining docs with {n_workers} workers...")
        print(f"  Chunk size: {chunk_size:,} (saves progress every chunk)")
        t0 = time.time()

        with Pool(n_workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
            for start in range(docs_done, total_docs, chunk_size):
                end = min(start + chunk_size, total_docs)

                # Load only this chunk's texts from the dataset
                chunk_texts = [ds[i]["text"] for i in range(start, end)]

                # Tokenize in parallel
                results = pool.map(_tokenize_doc, chunk_texts)

                # Free the text immediately
                del chunk_texts

                # Flatten results into a single array and write to disk
                chunk_tokens = []
                chunk_docs = 0
                for tokens in results:
                    if tokens is not None:
                        chunk_tokens.extend(tokens)
                        chunk_docs += 1
                del results  # free worker results

                # Append to binary file
                if chunk_tokens:
                    arr = np.array(chunk_tokens, dtype=np.uint16)
                    with open(raw_path, "ab") as f:
                        f.write(arr.tobytes())
                    tokens_written += len(chunk_tokens)
                    del arr
                del chunk_tokens

                docs_done = end

                # Save progress (atomic-ish: write then rename)
                tmp_progress = progress_path + ".tmp"
                with open(tmp_progress, "w") as f:
                    json.dump({"docs_done": docs_done, "tokens_written": tokens_written}, f)
                os.replace(tmp_progress, progress_path)

                elapsed = time.time() - t0
                docs_so_far = end - (total_docs - remaining)  # docs done in this run
                rate = docs_so_far / elapsed if elapsed > 0 else 0
                eta = (total_docs - end) / rate if rate > 0 else 0
                print(f"  {end:>10,}/{total_docs:,} | "
                      f"{tokens_written:>12,} tokens | "
                      f"{rate:.0f} docs/s | "
                      f"ETA {eta/60:.0f}min")

        print(f"\n  Tokenization complete: {docs_done:,} docs, {tokens_written:,} tokens")

    # === Split into train/val ===
    print(f"\nSplitting into train/val...")
    all_data = np.memmap(raw_path, dtype=np.uint16, mode="r")
    total_tokens = len(all_data)

    n_val = int(total_tokens * val_fraction)
    n_train = total_tokens - n_val

    print(f"  Total:  {total_tokens:,} tokens")
    print(f"  Train:  {n_train:,}")
    print(f"  Val:    {n_val:,}")

    # Write train.bin
    train_mm = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(n_train,))
    # Copy in chunks to avoid memory spike
    copy_chunk = 50_000_000  # 50M tokens at a time = 100MB
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
    if os.path.exists(raw_path):
        os.remove(raw_path)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print("\nDone! Temp files cleaned up.")


if __name__ == "__main__":
    main()
