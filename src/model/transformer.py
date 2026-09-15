import torch
import torch.nn as nn

from model.rmsnorm import RMSNorm
from model.attention import GroupedQueryAttention
from model.feedforward import SwiGLUFFN

"""
Transformer Block — Assembling the Pieces
Now we snap together everything you've built. This is the satisfying part.

The Block Architecture
A single transformer block is:



    Input (batch, seq_len, d_model)
      │
      ├──────────────────┐
      │                  │
  RMSNorm                │
      │                  │
GQA Attention            │     ← "gather information from other tokens"
      │                  │
      + ←────────────────┘     ← RESIDUAL CONNECTION #1
      │
      ├──────────────────┐
      │                  │
  RMSNorm                │
      │                  │
SwiGLU FFN               │     ← "process the gathered information"
      │                  │
      + ←────────────────┘     ← RESIDUAL CONNECTION #2
      │
    Output (batch, seq_len, d_model)
That's it. Normalize → Attend → Add. Normalize → FFN → Add. Repeat.

Residual Connections: Why They're Critical
The residual (skip) connection is the + in output = x + sublayer(x). It looks trivial but is arguably the most important architectural choice in deep learning.

The gradient flow problem:

Without residuals, a 8-layer network looks like:



output = f₈(f₇(f₆(f₅(f₄(f₃(f₂(f₁(x))))))))
During backpropagation, gradients multiply through each layer. If each layer's gradient is slightly < 1 (say 0.9), then:



Gradient at layer 1 = 0.9⁸ = 0.43 of the original  ← vanishing!
With 32 layers: 0.9³² = 0.03. The early layers barely learn.

With residuals:



output = x + f(x)
gradient = 1 + f'(x)     ← the "1" is the skip connection's gradient
There is ALWAYS a direct path for gradients to flow backward through the +. Even if f'(x) is tiny, the gradient through the skip path is exactly 1. This is why we can train networks with hundreds of layers. Without residuals, training anything beyond ~10 layers is nearly impossible.

The intuition: Each layer learns a DELTA — a small correction to the input, not a complete transformation. Layer 1 makes a small adjustment, layer 2 adjusts further, etc. The input flows through the entire network almost unchanged, picking up refinements along the way.

Pre-Norm: Why We Normalize Before, Not After


Post-norm (GPT-2):   output = Norm(x + Attention(x))
Pre-norm  (LLaMA):   output = x + Attention(Norm(x))      ← what we use
With pre-norm, the residual path is completely clean — x passes through the + without any processing. The gradient through the skip path is exactly 1, no matter what. With post-norm, the gradient passes through the Norm layer, which can distort it.

Pre-norm is more stable, especially for larger/deeper models. The trade-off is slightly worse final performance in some cases, but the stability gains are worth it.

"""



class TransformerBlock(nn.Module):
    """
    A single transformer block with pre-norm architecture.

    Data flow:
        x → RMSNorm → GQA Attention → + residual → RMSNorm → SwiGLU FFN → + residual → out

    This block is the fundamental repeating unit of the transformer.
    Pebble-25M stacks 8 of these blocks.

    Pre-norm (normalize before sublayer) is used instead of post-norm
    (normalize after) for better training stability. The residual
    connections ensure clean gradient flow through the entire network.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        norm_eps: float = 1e-6,
    ):
        """
        Args:
            d_model: Model dimension (e.g., 512).
            n_heads: Number of query attention heads (e.g., 8).
            n_kv_heads: Number of key/value heads for GQA (e.g., 4).
            d_ff: Feed-forward hidden dimension (e.g., 1376).
            max_seq_len: Maximum sequence length for RoPE.
            norm_eps: Epsilon for RMSNorm numerical stability.
        """
        super().__init__()

        # Pre-attention normalization
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)

        # Grouped Query Attention
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
        )

        # Pre-FFN normalization
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)

        # SwiGLU Feed-Forward Network
        self.ffn = SwiGLUFFN(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Causal attention mask of shape (seq_len, seq_len).
            offset: Position offset for RoPE (inference with KV-cache).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Sub-layer 1: Attention with pre-norm and residual
        # x ──┬──→ RMSNorm → Attention ──┐
        #     └────────────────────────── + → h
        h = x + self.attention(self.attn_norm(x), mask=mask, offset=offset)

        # Sub-layer 2: FFN with pre-norm and residual
        # h ──┬──→ RMSNorm → SwiGLU FFN ──┐
        #     └─────────────────────────── + → out
        out = h + self.ffn(self.ffn_norm(h))

        return out