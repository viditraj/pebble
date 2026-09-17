import math
import torch
import torch.nn as nn

from model.rmsnorm import RMSNorm
from model.transformer import TransformerBlock
from model.attention import create_causal_mask

"""
Pebble architecture:
- Token embeddings
- N stacked transformer blocks with pre-norm
- Final RMSNorm
- Output projection (weight-tied with embedding)

Pebble-25M config: vocab=16384, d=512, layers=8, heads=8, kv_heads=4

What You Now Understand
RMSNorm -- why normalization stabilizes training, why RMS is simpler than LayerNorm
RoPE -- how rotation encodes relative position, why dot products only depend on distance
GQA -- how sharing KV heads saves 25% of attention parameters and halves KV-cache
SwiGLU -- how gating outperforms simple activation functions
Residual connections -- why they're essential for gradient flow in deep networks
Pre-norm -- why normalize before the sublayer, not after
Weight tying -- sharing embedding and output projection saves millions of parameters
Weight initialization -- scaled init for residual paths prevents variance explosion
Cross-entropy loss -- how next-token prediction works as a loss function
Autograd -- PyTorch records the computation graph and differentiates through it automatically
These are the same architectural choices used by LLaMA 3, Mistral, Qwen, and Gemma. The model you built is a miniature version of the same architecture running behind billion-dollar AI products.
"""


class PebbleConfig:
    """Configuration for the Pebble language model."""

    def __init__(
        self,
        vocab_size: int = 16384,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 4,
        d_ff: int = 1376,
        max_seq_len: int = 1024,
        norm_eps: float = 1e-6,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.norm_eps = norm_eps


class Pebble(nn.Module):
    """
    Pebble: A small language model built from scratch.

    Architecture follows the LLaMA family:
    - Token embeddings (no positional embedding — RoPE handles position)
    - N stacked transformer blocks with pre-norm
    - Final RMSNorm
    - Output projection (weight-tied with embedding)

    Pebble-25M config: vocab=16384, d=512, layers=8, heads=8, kv_heads=4
    """

    def __init__(self, config: PebbleConfig):
        super().__init__()
        self.config = config

        # Token embedding: integer token IDs → dense vectors
        # Shape: (vocab_size, d_model) = (16384, 512)
        # No positional embedding here — RoPE is applied inside attention
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Stack of transformer blocks
        # nn.ModuleList registers each block as a submodule so PyTorch
        # can find their parameters for optimization
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.n_layers)
        ])

        # Final normalization before the output projection
        # Without this, the output from the last transformer block
        # could have arbitrary scale, making logits unstable
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Output projection: d_model → vocab_size
        # Produces a "logit" (unnormalized score) for each token in the vocabulary
        # The token with the highest logit is the model's prediction
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share the embedding weights with the output projection
        # This means lm_head.weight IS the same tensor as embedding.weight
        # Not a copy — the same object in memory. Gradients from both paths
        # accumulate on the single shared weight matrix.
        self.lm_head.weight = self.embedding.weight

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights for stable training.

        Strategy:
        - Embeddings: normal(0, 0.02) — small random values
        - Linear layers: normal(0, 0.02) — standard for transformers
        - Residual output projections (wo, w2): scaled by 1/sqrt(2*n_layers)
          to prevent variance growth from residual accumulation
        - RMSNorm weights: already initialized to 1.0 by default
        """
        std = 0.02
        residual_std = std / math.sqrt(2 * self.config.n_layers)

        for name, param in self.named_parameters():
            if param.dim() < 2:
                # Skip 1D params (norm weights, biases)
                # RMSNorm weight is already initialized to ones
                continue

            if "wo." in name or "w2." in name:
                # Output projections of attention (wo) and FFN (w2)
                # are in the residual path — scale down to prevent
                # variance growth across layers
                nn.init.normal_(param, mean=0.0, std=residual_std)
            else:
                # All other weight matrices
                nn.init.normal_(param, mean=0.0, std=std)

    def forward(
        self,
        token_ids: torch.Tensor,
        offset: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass: token IDs → next-token logits.

        Args:
            token_ids: Integer tensor of shape (batch, seq_len).
                       Each value is in [0, vocab_size).
            offset: Position offset for RoPE (used during inference
                    with KV-cache, where we only process new tokens).

        Returns:
            Logits tensor of shape (batch, seq_len, vocab_size).
            logits[b][t][v] = how likely token v is to follow
            the sequence up to position t in batch element b.
        """
        seq_len = token_ids.shape[1]

        # Step 1: Convert token IDs to vectors
        # (batch, seq_len) → (batch, seq_len, d_model)
        x = self.embedding(token_ids)

        # Step 2: Create causal mask (only needed for inference with offset).
        # During training, SDPA uses is_causal=True internally (no mask needed).
        mask = None
        if offset > 0:
            mask = create_causal_mask(seq_len, x.device)

        # Step 3: Pass through all transformer blocks
        for layer in self.layers:
            x = layer(x, mask=mask, offset=offset)

        # Step 4: Final normalization
        x = self.final_norm(x)

        # Step 5: Project to vocabulary logits
        # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits