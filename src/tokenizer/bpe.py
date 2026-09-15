"""
Byte-Pair Encoding (BPE) tokenizer built from scratch.

This implements the same algorithm used by GPT-2, LLaMA, and most modern LLMs.
The tokenizer starts from raw bytes (0-255) and iteratively merges the most
frequent adjacent pair into a new token until the vocabulary reaches the target size.

Usage:
    # Train
    tokenizer = BPETokenizer.train(text, vocab_size=16384)
    tokenizer.save("tokenizer.json")

    # Use
    tokenizer = BPETokenizer.load("tokenizer.json")
    ids = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)


    Understanding Tokenization
The Fundamental Problem
Neural networks work with numbers. Text is characters. We need a bridge.

The naive approach — one-hot encoding every character — creates absurdly long sequences. The sentence "Hello, world!" is 13 characters. At 1024-token context length, you'd barely fit a paragraph. And each character carries almost no semantic meaning on its own.

The other extreme — one token per word — creates a massive vocabulary. English has ~170,000 words in current use, plus proper nouns, misspellings, code, URLs, numbers... You'd need millions of tokens, and the embedding table alone would be billions of parameters.

Subword tokenization finds the sweet spot:



Character-level:  H  e  l  l  o  ,     w  o  r  l  d  !     (13 tokens)
Word-level:       Hello  ,  world  !                          (4 tokens, huge vocab)
BPE subword:      Hello  ,  world  !                          (4 tokens, small vocab)
                  un  happi  ness                              (3 tokens — rare word split)
                  un  believ  able                             (3 tokens — shares "un" prefix!)
Common words stay as single tokens. Rare words get split into meaningful pieces. The model learns that "un-" means negation because it sees it as a recurring subword across "unhappy", "unlikely", "unusual".

How BPE Works — Step by Step
BPE (Byte-Pair Encoding) was originally a data compression algorithm from 1994. Its application to NLP tokenization is brilliantly simple:

Start: Every unique byte (0-255) is a token. Any text can be encoded as raw bytes.

Repeat:

Look at your training text, encoded with current tokens
Count every adjacent pair of tokens
Find the most frequent pair
Merge that pair into a new token
Add to vocabulary
Go back to step 1
Stop when vocabulary reaches target size.

Let me walk through a concrete example:



Training text: "aabaabaab"
 
Step 0: Character-level tokens
  Vocabulary: {a, b}
  Encoded: [a, a, b, a, a, b, a, a, b]
 
Step 1: Count pairs
  (a,a) appears 3 times  ← WINNER
  (a,b) appears 3 times
  (b,a) appears 2 times
 
Step 2: Merge (a,a) → new token "aa"
  Vocabulary: {a, b, aa}
  Encoded: [aa, b, aa, b, aa, b]
 
Step 3: Count pairs again
  (aa,b) appears 3 times  ← WINNER
  (b,aa) appears 2 times
 
Step 4: Merge (aa,b) → new token "aab"
  Vocabulary: {a, b, aa, aab}
  Encoded: [aab, aab, aab]
 
Done! "aabaabaab" is now just 3 tokens.
The merge rules define the tokenizer:



Rule 1: a + a → aa
Rule 2: aa + b → aab
These rules are applied IN ORDER during encoding. That's the entire algorithm.

Why Byte-Level?
Modern BPE (used by GPT-2, LLaMA, etc.) starts from bytes (0-255), not characters. Why?

Universal: Any file can be represented as bytes — English, Chinese, emoji, binary, code. No "unknown character" problem.
Small base: Only 256 base tokens, vs thousands for Unicode characters.
No preprocessing: No need for language-specific tokenization rules.
The UTF-8 encoding of text gives us the bytes:



"Hello"  →  [72, 101, 108, 108, 111]           (ASCII, 1 byte each)
"café"   →  [99, 97, 102, 195, 169]            (é = 2 bytes in UTF-8)
"你好"   →  [228, 189, 160, 229, 165, 189]     (Chinese = 3 bytes each)
BPE merges the most frequent byte pairs into new tokens. After training, common English words become single tokens, while rare Unicode characters stay as 2-3 byte-tokens.
"""

import json
import re
import time
from collections import Counter


class BPETokenizer:
    """
    A byte-level BPE tokenizer.

    Vocabulary structure:
        Tokens 0-255:     Raw byte values (the base alphabet)
        Tokens 256+:      Merged tokens (learned from training data)
        Last N tokens:     Special tokens (<pad>, <bos>, <eos>, <unk>, tool-calling tokens)
    """

    def __init__(self, merges: list[tuple[int, int]], vocab_size: int, special_tokens: dict[str, int] | None = None):
        """
        Args:
            merges: Ordered list of (token_a, token_b) merge rules.
                    The i-th merge creates token 256+i.
            vocab_size: Total vocabulary size (including bytes + merges + special tokens).
            special_tokens: Mapping of special token strings to their IDs.
        """
        self.merges = merges
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {}

        # Build the reverse mapping: special token ID → string
        self.special_tokens_reverse = {v: k for k, v in self.special_tokens.items()}

        # Build merge lookup: (token_a, token_b) → merged_token_id
        # This makes encoding O(1) per merge lookup instead of scanning the list
        self.merge_map = {}
        for i, (a, b) in enumerate(merges):
            self.merge_map[(a, b)] = 256 + i

        # Build vocabulary: token_id → bytes
        # Tokens 0-255 are single bytes
        # Token 256 is the first merge, 257 is the second, etc.
        self.vocab = {}
        for i in range(256):
            self.vocab[i] = bytes([i])
        for i, (a, b) in enumerate(merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

    @classmethod
    def train(cls, text: str, vocab_size: int, special_tokens: list[str] | None = None, verbose: bool = True) -> "BPETokenizer":
        """
        Train a BPE tokenizer on the given text.

        Args:
            text: Training text corpus.
            vocab_size: Target vocabulary size (including 256 byte tokens + special tokens).
            special_tokens: List of special token strings to reserve at the end of vocab.
            verbose: Print progress during training.

        Returns:
            A trained BPETokenizer.
        """
        special_tokens = special_tokens or []

        # Number of merges to learn:
        # vocab_size - 256 (bytes) - len(special_tokens) = number of merge tokens
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges > 0, f"vocab_size {vocab_size} too small for {len(special_tokens)} special tokens"

        # Step 1: Convert text to bytes
        # This is our initial token sequence — every byte is a token
        tokens = list(text.encode("utf-8"))

        if verbose:
            print(f"Training BPE tokenizer...")
            print(f"  Text length: {len(text):,} characters")
            print(f"  Byte length: {len(tokens):,} bytes")
            print(f"  Target vocab size: {vocab_size:,}")
            print(f"  Merges to learn: {num_merges:,}")
            print(f"  Special tokens: {len(special_tokens)}")
            print()

        merges = []

        for i in range(num_merges):
            # Step 2: Count all adjacent pairs
            pair_counts = cls._count_pairs(tokens)

            if not pair_counts:
                # No more pairs to merge (text is fully compressed)
                if verbose:
                    print(f"  No more pairs to merge at step {i}. Stopping early.")
                break

            # Step 3: Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            # Step 4: Create new token ID and record the merge
            new_token_id = 256 + i
            merges.append(best_pair)

            # Step 5: Apply the merge — replace all occurrences of the pair
            tokens = cls._apply_merge(tokens, best_pair, new_token_id)

            if verbose and (i < 10 or i % 500 == 0 or i == num_merges - 1):
                print(f"  Merge {i+1:>5}/{num_merges}: {best_pair} → {new_token_id}  "
                      f"(count={best_count:,}, tokens remaining={len(tokens):,})")

        # Assign IDs to special tokens (at the end of the vocabulary)
        special_token_map = {}
        for j, token_str in enumerate(special_tokens):
            special_token_map[token_str] = 256 + len(merges) + j

        return cls(merges=merges, vocab_size=vocab_size, special_tokens=special_token_map)

    @classmethod
    def train_fast(cls, text: str, vocab_size: int, special_tokens: list[str] | None = None, verbose: bool = True) -> "BPETokenizer":
        """
        Train BPE tokenizer using word-level frequency optimization.

        Instead of scanning the entire corpus every iteration, this method:
        1. Splits text into words using a regex (GPT-2 style)
        2. Converts each word to bytes
        3. Counts word frequencies
        4. Performs merges on the word vocabulary, weighted by frequency

        This is O(num_merges * num_unique_words) instead of O(num_merges * corpus_size),
        which is dramatically faster for large texts with repeated words.

        This is how real tokenizers (GPT-2, LLaMA) handle training.
        """
        special_tokens = special_tokens or []
        num_merges = vocab_size - 256 - len(special_tokens)
        assert num_merges > 0, f"vocab_size {vocab_size} too small for {len(special_tokens)} special tokens"

        # GPT-2 style regex: splits text into words, numbers, punctuation, whitespace
        # This prevents merges across word boundaries (we don't want "the" + " cat" = "the cat")
        # Each match is a "word" that gets independently tokenized
        pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+""")

        if verbose:
            print(f"Training BPE tokenizer (fast mode)...")
            print(f"  Text length: {len(text):,} characters")
            t0 = time.time()

        # Step 1: Split text into words and count frequencies
        word_freqs = Counter()
        for match in pattern.finditer(text):
            word = match.group()
            word_freqs[tuple(word.encode("utf-8"))] += 1

        if verbose:
            print(f"  Unique words: {len(word_freqs):,}")
            print(f"  Target merges: {num_merges:,}")
            print()

        merges = []

        for i in range(num_merges):
            # Count pairs across all words, weighted by word frequency
            pair_counts = Counter()
            for word_tokens, freq in word_freqs.items():
                for j in range(len(word_tokens) - 1):
                    pair = (word_tokens[j], word_tokens[j + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                if verbose:
                    print(f"  No more pairs at merge {i}. Stopping early.")
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]

            if best_count < 2:
                if verbose:
                    print(f"  Best pair count dropped to {best_count} at merge {i}. Stopping.")
                break

            new_token_id = 256 + i
            merges.append(best_pair)

            # Apply merge to all words in the vocabulary
            new_word_freqs = {}
            for word_tokens, freq in word_freqs.items():
                new_word = cls._apply_merge(list(word_tokens), best_pair, new_token_id)
                new_key = tuple(new_word)
                new_word_freqs[new_key] = new_word_freqs.get(new_key, 0) + freq
            word_freqs = new_word_freqs

            if verbose and (i < 10 or i % 1000 == 0 or i == num_merges - 1):
                elapsed = time.time() - t0
                # Show what the merged token looks like as text
                # Build token → bytes mapping on the fly
                token_bytes = {}
                for b in range(256):
                    token_bytes[b] = bytes([b])
                for mi, (a, b) in enumerate(merges):
                    token_bytes[256 + mi] = token_bytes[a] + token_bytes[b]
                merged_str = token_bytes[new_token_id].decode("utf-8", errors="replace")
                print(f"  Merge {i+1:>5}/{num_merges}: "
                      f"{repr(merged_str):>20s}  count={best_count:>8,}  "
                      f"[{elapsed:.1f}s]")

        if verbose:
            print(f"\n  Done in {time.time() - t0:.1f}s. Learned {len(merges):,} merges.")

        special_token_map = {}
        for j, token_str in enumerate(special_tokens):
            special_token_map[token_str] = 256 + len(merges) + j

        return cls(merges=merges, vocab_size=vocab_size, special_tokens=special_token_map)

    @staticmethod
    def _count_pairs(tokens: list[int]) -> Counter:
        """Count occurrences of each adjacent token pair."""
        counts = Counter()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] += 1
        return counts

    @staticmethod
    def _apply_merge(tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        """
        Replace all occurrences of `pair` in the token list with `new_token`.

        Scans left to right. When pair (a, b) is found, replace with new_token
        and skip the next element (since b was consumed).
        """
        merged = []
        i = 0
        while i < len(tokens):
            # Check if this position starts the pair we want to merge
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                merged.append(new_token)
                i += 2  # skip both tokens of the pair
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        """
        Encode a chunk of bytes into token IDs using the merge_map.

        Instead of applying all 16K merges in order (slow), we repeatedly
        find the highest-priority pair in the current token list and merge it.
        A merge learned earlier (lower ID) has higher priority.

        This is O(n * m_actual) where m_actual is the number of merges that
        actually apply to this text — usually much less than total merges.
        """
        tokens = list(text_bytes)

        while len(tokens) >= 2:
            # Find the pair with the lowest merge index (= highest priority)
            best_pair = None
            best_id = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                merged_id = self.merge_map.get(pair)
                if merged_id is not None and merged_id < best_id:
                    best_pair = pair
                    best_id = merged_id

            if best_pair is None:
                break  # no more merges possible

            tokens = self._apply_merge(tokens, best_pair, best_id)

        return tokens

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.

        Process:
        1. Split text around any special tokens
        2. Convert each text chunk to raw bytes
        3. Apply merges using priority-based merge_map lookup
        4. Return the resulting token IDs
        """
        if self.special_tokens:
            return self._encode_with_special_tokens(text)
        return self._encode_chunk(text.encode("utf-8"))

    def _encode_with_special_tokens(self, text: str) -> list[int]:
        """Encode text that may contain special token strings."""
        sorted_specials = sorted(self.special_tokens.keys(), key=len, reverse=True)

        # Split text around special tokens
        parts = [text]
        for special in sorted_specials:
            new_parts = []
            for part in parts:
                if isinstance(part, int):
                    new_parts.append(part)
                    continue
                pieces = part.split(special)
                for j, piece in enumerate(pieces):
                    if piece:
                        new_parts.append(piece)
                    if j < len(pieces) - 1:
                        new_parts.append(self.special_tokens[special])
            parts = new_parts

        # Encode each string part, pass through token IDs
        result = []
        for part in parts:
            if isinstance(part, int):
                result.append(part)
            else:
                result.extend(self._encode_chunk(part.encode("utf-8")))

        return result

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs back to text.

        Process:
        1. For each token ID, look up its byte representation
        2. Concatenate all bytes
        3. Decode UTF-8 bytes to string

        This is much simpler than encoding — just a lookup table.
        """
        byte_pieces = []
        for token_id in token_ids:
            if token_id in self.special_tokens_reverse:
                # Special token: encode its string representation as bytes
                byte_pieces.append(self.special_tokens_reverse[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                byte_pieces.append(self.vocab[token_id])
            else:
                # Unknown token — shouldn't happen with a well-formed tokenizer
                byte_pieces.append(b"<?>")

        # Concatenate all byte pieces and decode to string
        all_bytes = b"".join(byte_pieces)
        # errors="replace" handles any invalid UTF-8 sequences gracefully
        return all_bytes.decode("utf-8", errors="replace")

    def save(self, path: str):
        """Save tokenizer to a JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        # JSON stores tuples as lists, convert back
        merges = [tuple(m) for m in data["merges"]]
        return cls(
            merges=merges,
            vocab_size=data["vocab_size"],
            special_tokens=data.get("special_tokens", {}),
        )

    def __len__(self) -> int:
        return self.vocab_size