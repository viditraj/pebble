import torch
import torch.nn as nn


"""
Root Mean Square Layer Normalization (RMSNorm).

We start here because it's the easiest to understand and shows up everywhere in the model.

The Problem Normalization Solves
As data flows through a deep network, the distribution of values at each layer can shift wildly. Layer 1 might output values in [-0.1, 0.1], but layer 6 might output [-500, 500]. This is called internal covariate shift, and it makes training unstable — gradients explode or vanish.

Normalization forces each layer's output back to a reasonable scale.

LayerNorm vs RMSNorm
LayerNorm (original transformer, GPT-2):



LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
It centers the data (subtract mean) AND scales it (divide by std). Has two learnable parameters: gamma (scale) and beta (shift).

RMSNorm (LLaMA, Mistral, modern models):

RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
It ONLY scales — no centering, no beta. The insight from the RMSNorm paper is that the centering (mean subtraction) in LayerNorm doesn't actually help much. Removing it makes the operation simpler and ~10-15% faster.

The Math, Step by Step
Given an input vector x of dimension d_model = 512:

x = [0.5, -1.2, 0.8, 0.3, ...]    (512 values)
 
Step 1: Square each element
x² = [0.25, 1.44, 0.64, 0.09, ...]
 
Step 2: Take the mean of the squares
mean(x²) = (0.25 + 1.44 + 0.64 + 0.09 + ...) / 512 = 0.47 (example)
 
Step 3: Take the root (RMS = Root Mean Square)
rms = sqrt(0.47 + 1e-6) = 0.686
         (the 1e-6 is epsilon — prevents division by zero)
 
Step 4: Divide original x by rms
x_norm = x / 0.686 = [0.729, -1.749, 1.166, 0.437, ...]
 
Step 5: Multiply by learnable weight gamma
output = x_norm * gamma    (gamma starts as all 1s, learned during training)

Why gamma? After normalization, all dimensions have similar magnitude. But maybe some dimensions SHOULD be bigger — the model learns this via gamma. It starts at 1.0 (no change) and adjusts during training.

Pre-Norm vs Post-Norm
Original transformer (2017): Post-norm — normalize AFTER the residual connection. Modern models (LLaMA, etc.): Pre-norm — normalize BEFORE attention/FFN.


Post-norm:  output = Norm(x + Attention(x))     ← original transformer
Pre-norm:   output = x + Attention(Norm(x))      ← what we use
Pre-norm is more stable during training. The residual connection always gets a "clean" path for gradients (nothing processed between the addition and the output). This is why modern LLMs can be trained with hundreds of layers without instability.
"""

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Unlike LayerNorm, RMSNorm does not center the input (no mean subtraction).
    It only scales by the root-mean-square, which is simpler and ~10-15% faster.

    Used in: LLaMA, LLaMA 2, LLaMA 3, Mistral, Gemma, Qwen, and Pebble.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: The dimension of the input (e.g., 512 for Pebble-25M).
            eps: Small constant to prevent division by zero. 1e-6 is standard.
        """
        super().__init__()

        # Learnable scale parameter, initialized to 1.0 for each dimension.
        # The model will learn to scale some dimensions up and others down.
        self.weight = nn.Parameter(torch.ones(d_model))

        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Normalized tensor of same shape.
        """
        # x.pow(2)          → square each element
        # .mean(-1, ...)    → mean across last dim (d_model), keep dim for broadcasting
        # + self.eps        → numerical stability
        # torch.rsqrt(...)  → 1/sqrt(...), faster than sqrt then divide
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        # Scale the input by 1/rms, then by the learnable weight
        return x * rms * self.weight