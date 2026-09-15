import torch
import torch.nn as nn

"""
Rotary Positional Embedding (RoPE) implementation.

This module implements the rotary positional encoding mechanism used in transformer models
to encode relative positional information in attention mechanisms.

This is the most mathematically beautiful component in the whole model. Take your time with this one.

The Problem: Transformers Have No Sense of Order
Here's a surprising fact. If you feed a transformer these two sentences:



"The cat sat on the mat"
"mat the on sat cat The"
Without positional encoding, the self-attention layer produces the exact same output for both. Why? Because attention computes Q @ K^T — a dot product between every pair of tokens. Dot products are commutative. The operation doesn't know which token came first, second, or last.

This is called permutation invariance, and it's a fundamental property of the attention mechanism. It's great for things like sets, but terrible for language where word ORDER is meaning ("dog bites man" vs "man bites dog").

So we need to inject position information somehow.

The Evolution of Positional Encodings
1. Sinusoidal (2017, original Transformer)



PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
Add a fixed sinusoidal pattern to each token based on its position. Simple but crude — the position info gets mixed with the token's semantic meaning.

2. Learned (GPT-2, 2019)



self.pos_emb = nn.Embedding(max_seq_len, d_model)  # just learn it
Let the model learn a position vector for each position. Problem: can't generalize beyond the max sequence length seen during training. Position 1025 is meaningless if you trained with max 1024.

3. ALiBi (2022) Instead of adding position info to embeddings, subtract a penalty from attention scores based on distance. Simple, but doesn't perform as well as RoPE.

4. RoPE (2021, used in LLaMA, Mistral, Qwen, Pebble) The key insight: encode position by rotating the query and key vectors. This means the dot product q · k naturally encodes the RELATIVE distance between two tokens — not their absolute positions.

RoPE: The Core Idea
Imagine you have a 2D vector. You can encode a position by rotating it:



Position 0: rotate by 0°
Position 1: rotate by θ°
Position 2: rotate by 2θ°
Position 3: rotate by 3θ°
...
Now here's the magic. If token at position m has query q and token at position n has key k, then:



rotated_q · rotated_k = q · R(m)^T · R(n) · k = q · R(n-m) · k
The dot product only depends on the relative distance (n-m), not on the absolute positions! The rotation matrices cancel out, leaving only the difference.

This means:

Token at position 5 attending to position 3 → same as position 105 attending to position 103
The model naturally learns RELATIVE position patterns
It can generalize to longer sequences than it was trained on (to some extent)
The Math in Detail
Our head dimension is 64. RoPE processes these 64 dimensions as 32 pairs of 2D rotations.

Each pair (x₁, x₂) gets rotated by angle θᵢ * position:



pair i at position pos:
 
┌ x₁' ┐   ┌ cos(θᵢ·pos)  -sin(θᵢ·pos) ┐   ┌ x₁ ┐
│      │ = │                              │ × │    │
└ x₂' ┘   └ sin(θᵢ·pos)   cos(θᵢ·pos) ┘   └ x₂ ┘
The angle θᵢ for each pair is different, creating a "frequency spectrum":



θᵢ = 1 / (10000^(2i/d))
 
θ₀  = 1/10000^(0/64)  = 1.0        (fast rotation — high frequency)
θ₁  = 1/10000^(2/64)  = 0.724
θ₂  = 1/10000^(4/64)  = 0.524
...
θ₃₁ = 1/10000^(62/64) = 0.00015    (very slow rotation — low frequency)
Why different frequencies? The fast-rotating pairs change rapidly between adjacent positions — they encode fine-grained local position (is this word 1 or 2 positions away?). The slow-rotating pairs change gradually over long distances — they encode coarse global position (is this in the first half or second half of the sequence?). Together, they create a rich multi-scale position representation.

This is analogous to how clocks work: the second hand (high freq) tells you the exact second, the hour hand (low freq) tells you the rough time of day.

Efficient Implementation
We DON'T actually build rotation matrices. Instead, we use the identity:



┌ x₁·cos(θ) - x₂·sin(θ) ┐       ┌ x₁ ┐           ┌ -x₂ ┐
│                          │  =    │    │ · cos(θ) + │      │ · sin(θ)
└ x₁·sin(θ) + x₂·cos(θ) ┘       └ x₂ ┘           └  x₁ ┘
So: rotated = x * cos + rotate_half(x) * sin

Where rotate_half swaps each pair and negates the first element: [x₁, x₂, x₃, x₄, ...] → [-x₂, x₁, -x₄, x₃, ...]

This is element-wise multiplication — very fast on GPUs, no matrix multiplication needed.
"""

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).

    Encodes position by rotating query/key vectors in 2D subspaces.
    The dot product between rotated q and k naturally captures
    RELATIVE position, not absolute — enabling length generalization.

    Used in: LLaMA 1/2/3, Mistral, Qwen, Gemma, and Pebble.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        """
        Args:
            head_dim: Dimension of each attention head (e.g., 64).
                      Must be even — we process pairs of dimensions.
            max_seq_len: Maximum sequence length to precompute. Can generate
                         beyond this at inference time if needed.
            base: Base for the frequency computation. 10000 is the standard
                  from the original paper. Larger base = slower rotation =
                  better for very long sequences.
        """
        super().__init__()

        assert head_dim % 2 == 0, "head_dim must be even for RoPE (we rotate pairs)"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute the cos and sin tables.
        # These are NOT learnable — they're fixed mathematical functions.
        # We use register_buffer so they're saved with the model but not trained.
        cos, sin = self._build_cache(max_seq_len)
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def _build_cache(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute cos and sin for all positions and frequency pairs."""

        # Step 1: Compute the frequency for each pair of dimensions.
        # theta_i = 1 / (base ^ (2i / head_dim))  for i = 0, 1, ..., head_dim/2 - 1
        #
        # Example with head_dim=64, base=10000:
        #   i=0:  theta = 1/10000^(0/64)  = 1.0       (rotates fast)
        #   i=31: theta = 1/10000^(62/64) = 0.00015   (rotates slowly)
        half_dim = self.head_dim // 2
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        # freqs shape: (head_dim/2,) = (32,)

        # Step 2: Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len).float()
        # positions shape: (seq_len,)

        # Step 3: Outer product — every position × every frequency
        # This gives us the angle for each (position, frequency_pair) combination.
        angles = torch.outer(positions, freqs)
        # angles shape: (seq_len, head_dim/2) = (2048, 32)

        # Step 4: Compute cos and sin, then duplicate for both dims in each pair.
        # We need shape (seq_len, head_dim) to match the q/k vectors.
        # [cos(θ₀), cos(θ₀), cos(θ₁), cos(θ₁), ...] — but we use repeat_interleave
        # Actually, simpler: just repeat along last dim since we'll apply to pairs.
        cos = torch.cos(angles).repeat(1, 2)  # (seq_len, head_dim)
        sin = torch.sin(angles).repeat(1, 2)  # (seq_len, head_dim)

        return cos, sin

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Apply rotary embeddings to a query or key tensor.

        Args:
            x: Query or key tensor of shape (batch, n_heads, seq_len, head_dim)
            offset: Position offset (used during inference with KV-cache,
                    where new tokens start at position = past_seq_len)

        Returns:
            Rotated tensor of the same shape.
        """
        seq_len = x.shape[2]

        # Get the precomputed cos/sin for the relevant positions.
        # Shape: (seq_len, head_dim) → broadcast to (1, 1, seq_len, head_dim)
        cos = self.cos_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

        # Apply the rotation: x_rotated = x * cos + rotate_half(x) * sin
        return x * cos + self._rotate_half(x) * sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate pairs of dimensions: [x1, x2, x3, x4, ...] → [-x2, x1, -x4, x3, ...]

        This implements the "swap and negate" that makes the rotation formula work
        without constructing actual rotation matrices.
        """
        # Split into two halves along the last dimension
        x1 = x[..., : x.shape[-1] // 2]   # first half:  [x1, x3, x5, ...]
        x2 = x[..., x.shape[-1] // 2 :]   # second half: [x2, x4, x6, ...]

        # Interleave: [-x2, x1]
        return torch.cat((-x2, x1), dim=-1)