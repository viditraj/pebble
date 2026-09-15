"""
Memory-mapped dataset for efficient language model training.

Reads from a binary file of uint16 tokens (created by prepare_data.py)
and serves random fixed-length windows for next-token prediction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class PebbleDataset(Dataset):
    """
    A memory-mapped dataset that serves fixed-length token sequences.

    The dataset reads from a binary file of uint16 token IDs.
    Each sample is a random contiguous window of (seq_len + 1) tokens,
    split into input (first seq_len) and target (last seq_len, shifted by 1).

    Memory-mapped access means:
    - The full dataset lives on disk, not in RAM
    - Only accessed regions are loaded on demand
    - The OS manages caching transparently
    """

    def __init__(self, data_path: str, seq_len: int):
        """
        Args:
            data_path: Path to the .bin file (uint16 numpy memmap).
            seq_len: Sequence length for training (e.g., 1024).
                     Each sample will be seq_len+1 tokens
                     (input = first seq_len, target = last seq_len).
        """
        self.seq_len = seq_len

        # Open the memory-mapped file (does NOT load into RAM)
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

        # Number of valid starting positions
        # We need seq_len+1 contiguous tokens, so last valid start is n_tokens - seq_len - 1
        self.n_samples = self.n_tokens - seq_len - 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a single training example.

        Args:
            idx: Starting position in the token stream.

        Returns:
            (input_ids, target_ids) — both are LongTensors of shape (seq_len,)
        """
        # Read seq_len + 1 tokens starting at idx
        chunk = self.data[idx : idx + self.seq_len + 1].astype(np.int64)

        # Input: tokens [0, 1, 2, ..., seq_len-1]
        x = torch.from_numpy(chunk[:-1])

        # Target: tokens [1, 2, 3, ..., seq_len]  (shifted by 1)
        y = torch.from_numpy(chunk[1:])

        return x, y

    def __repr__(self):
        return (f"PebbleDataset(tokens={self.n_tokens:,}, "
                f"seq_len={self.seq_len}, "
                f"samples={self.n_samples:,}, "
                f"size={self.n_tokens * 2 / 1024**2:.1f} MB)")