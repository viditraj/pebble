import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import RotaryPositionalEmbedding

"""
Grouped Query Attention (GQA)
This is the biggest and most important component. Attention IS the transformer — everything else is supporting infrastructure. Take your time.

Attention: The Core Intuition
Forget the math for a moment. Here's what attention does intuitively.

You're reading: "The cat sat on the mat because it was tired"

When the model processes the word "it", it needs to figure out what "it" refers to. Attention is the mechanism that lets the model "look back" at all previous words and decide: "it" is most related to "cat".

How? Each token creates three vectors:

Query (Q): "I'm looking for something. What am I looking for?" — The word "it" asks: "what noun am I referring to?"
Key (K): "Here's what I contain. Match against me." — The word "cat" advertises: "I'm an animate noun"
Value (V): "If you match with me, here's the info I'll give you." — The word "cat" offers its semantic representation
The dot product Q_it · K_cat is HIGH (they match), while Q_it · K_on is LOW (preposition doesn't match). After softmax, "it" receives mostly the Value from "cat" and very little from other words.



                  Attention weights for "it":
The   cat   sat   on   the   mat   because   it
0.02  0.71  0.04  0.01  0.02  0.08   0.03    0.09
       ↑↑↑
  "it" attends mostly to "cat"
From Single-Head to Multi-Head Attention
One attention head can only capture ONE type of relationship. But language has many simultaneous relationships:

Head 1 might learn: syntactic subject ("it" → "cat")
Head 2 might learn: spatial proximity ("mat" → "on")
Head 3 might learn: semantic similarity ("cat" → "mat" — both physical objects)
Head 4 might learn: positional patterns (attend to the previous word)
Multi-Head Attention (MHA) runs multiple independent attention computations in parallel, each with their own Q, K, V projections. The results are concatenated and mixed:



d_model = 512, n_heads = 8  →  head_dim = 512/8 = 64
 
Input (512) → split into 8 heads of 64 dims each
           → each head does its own attention
           → concatenate all heads back to 512
           → one final linear projection
The Problem with Standard MHA: KV-Cache Memory
During inference (generating text), the model caches the Key and Value vectors for all past tokens (so it doesn't recompute them). This is the KV-cache.

With standard MHA and 8 heads:



KV-cache per token = 2 (K and V) × 8 heads × 64 head_dim × 2 bytes (BF16) = 2,048 bytes
For 1024 tokens:    = 2,048 × 1024 = 2 MB per layer
For 8 layers:       = 2 MB × 8 = 16 MB total
For our small model, 16 MB is fine. But for LLaMA 70B with 80 layers and 64 heads? The KV-cache alone takes tens of gigabytes. That's the real bottleneck, not the model weights.

GQA: The Solution
Grouped Query Attention shares Key and Value heads across groups of Query heads:



Standard MHA:   8 Q-heads,  8 KV-heads  (1:1 ratio)
GQA:            8 Q-heads,  4 KV-heads  (2:1 ratio — what Pebble uses)
MQA:            8 Q-heads,  1 KV-head   (8:1 ratio — too aggressive for small models)
With GQA, every 2 query heads share the same key and value head:



Q-heads:   Q₀  Q₁ │ Q₂  Q₃ │ Q₄  Q₅ │ Q₆  Q₇
                   │         │         │
KV-heads:  KV₀    │  KV₁    │  KV₂    │  KV₃
 
Group 0:  Q₀,Q₁ share KV₀
Group 1:  Q₂,Q₃ share KV₁
Group 2:  Q₄,Q₅ share KV₂
Group 3:  Q₆,Q₇ share KV₃
KV-cache is halved (4 KV heads instead of 8) with almost no quality loss. LLaMA 2 70B went from 64 KV-heads to 8 KV-heads and the benchmark scores barely changed.

Causal Masking: Why We Can't Look Ahead
During training, we process the entire sequence at once for efficiency. But the model must predict each token using ONLY the tokens BEFORE it — it can't peek at future tokens. We enforce this with a causal mask:



               Key positions →
               0    1    2    3    4
Q-pos 0:  [ 1.0  -inf -inf -inf -inf ]   ← can only see position 0
Q-pos 1:  [ 0.3   0.7 -inf -inf -inf ]   ← can see positions 0,1
Q-pos 2:  [ 0.1   0.2  0.7 -inf -inf ]   ← can see positions 0,1,2
Q-pos 3:  [ 0.1   0.1  0.3  0.5 -inf ]   ← can see positions 0,1,2,3
Q-pos 4:  [ 0.1   0.1  0.1  0.2  0.5 ]   ← can see all positions
The -inf values become 0 after softmax, so the model truly cannot attend to future tokens.

The Scaled Dot-Product: Why √d_k?
The raw attention scores are: scores = Q @ K^T

If head_dim = 64, each dot product sums 64 multiplications. By the central limit theorem, the result has variance proportional to d_k. Large variance means the softmax saturates — one value dominates, gradients vanish.

Dividing by √d_k = √64 = 8 normalizes the variance back to ~1, keeping softmax in its useful range:



Without scaling:  softmax([25.3, -18.7, 31.2, ...])  → [0.00, 0.00, 1.00, ...]  (saturated!)
With scaling:     softmax([3.16, -2.34, 3.90, ...])   → [0.18, 0.01, 0.38, ...]  (useful gradients)
The Complete Algorithm


1. Project input → Q, K, V      (three separate linear layers)
2. Reshape for multi-head        (split d_model into n_heads × head_dim)
3. Expand KV heads for GQA       (repeat each KV head for its group of Q heads)
4. Apply RoPE to Q and K         (inject position information)
5. Compute scores = Q @ K^T / √d (scaled dot product)
6. Apply causal mask              (set future positions to -inf)
7. Apply softmax                  (convert scores to probabilities)
8. Multiply by V                  (weighted sum of values)
9. Concatenate heads              (merge back to d_model)
10. Output projection             (one final linear layer to mix head outputs)
"""

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA).

    Multiple query heads share a smaller number of key/value heads,
    reducing KV-cache memory during inference with minimal quality loss.

    Configuration for Pebble-25M:
        d_model=512, n_heads=8, n_kv_heads=4, head_dim=64
        → 2 query heads per KV group

    Used in: LLaMA 2, LLaMA 3, Mistral, Gemma, and Pebble.

    Reference: https://arxiv.org/abs/2305.13245
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            d_model: Model dimension (e.g., 512).
            n_heads: Number of query heads (e.g., 8).
            n_kv_heads: Number of key/value heads (e.g., 4).
                        Must evenly divide n_heads.
            max_seq_len: Maximum sequence length for RoPE precomputation.
        """
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads  # how many Q heads share each KV head

        # Query projection: projects to ALL query heads
        # d_model → n_heads * head_dim (= d_model, but explicit for clarity)
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # Key and Value projections: project to FEWER heads (n_kv_heads)
        # d_model → n_kv_heads * head_dim
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)

        # Output projection: merge all heads back to d_model
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Rotary positional embeddings (applied to Q and K only, not V)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        # Scale factor for dot-product attention
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional causal mask of shape (seq_len, seq_len).
                  True/1 = attend, False/0 = mask out.
                  If None, no masking is applied (unusual for causal LM).
            offset: Position offset for RoPE (used with KV-cache during inference).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape

        # Step 1: Project input to queries, keys, and values.
        # Each linear layer is just a matrix multiply: x @ W^T
        q = self.wq(x)  # (batch, seq_len, n_heads * head_dim)
        k = self.wk(x)  # (batch, seq_len, n_kv_heads * head_dim)
        v = self.wv(x)  # (batch, seq_len, n_kv_heads * head_dim)

        # Step 2: Reshape to separate the head dimension.
        # We want: (batch, n_heads, seq_len, head_dim) for efficient batched attention.
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: (batch, n_heads, seq_len, head_dim)      = (B, 8, S, 64)
        # k: (batch, n_kv_heads, seq_len, head_dim)   = (B, 4, S, 64)
        # v: (batch, n_kv_heads, seq_len, head_dim)   = (B, 4, S, 64)

        # Step 3: Apply RoPE to queries and keys (NOT to values).
        # Position information is encoded via rotation — values carry content only.
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Step 4: Expand KV heads to match query heads (GQA).
        # Each KV head is repeated n_groups times.
        # (batch, 4, seq, 64) → (batch, 8, seq, 64)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
        # Now k and v have the same number of heads as q.

        # Step 5-8: Compute attention using PyTorch's optimized SDPA.
        # Uses FlashAttention/memory-efficient kernels under the hood —
        # never materializes the full (B, heads, S, S) attention matrix,
        # reducing VRAM from O(S^2) to O(S).
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            is_causal=(mask is None),  # use built-in causal mask if none provided
        )

        # Step 9: Concatenate all heads.
        # (B, n_heads, S, head_dim) → (B, S, n_heads * head_dim) = (B, S, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # Step 10: Final output projection to mix information across heads.
        return self.wo(out)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal (autoregressive) attention mask.

    Returns a (seq_len, seq_len) tensor where:
    - Upper triangle (future positions) = -infinity (masked out)
    - Lower triangle + diagonal (past + current) = 0 (attend freely)

    After adding to attention scores and applying softmax:
    - -inf + score → -inf → softmax → 0.0  (can't attend)
    - 0 + score    → score → softmax → >0   (can attend)

    Example for seq_len=4:
        [[ 0, -inf, -inf, -inf],
         [ 0,    0, -inf, -inf],
         [ 0,    0,    0, -inf],
         [ 0,    0,    0,    0]]
    """
    # torch.triu with diagonal=1 gives us 1s above the main diagonal
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # Convert: 1 → -inf, 0 → 0
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask