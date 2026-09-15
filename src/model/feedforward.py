import torch
import torch.nn as nn
import torch.nn.functional as F


"""
SwiGLU Feed-Forward Network
After the heavy lifting of attention, this one is refreshing. Short, elegant, and powerful.

What the FFN Does in a Transformer
Each transformer block has two sub-layers:

Attention — tokens talk to each other ("what information should I gather?")
FFN — each token processes its own information independently ("now what do I do with what I gathered?")
A useful analogy: attention is a meeting where everyone shares information. The FFN is individual work where each person processes what they learned at the meeting.

The FFN is applied to each token position independently — token at position 5 doesn't interact with token at position 3 during the FFN. All cross-token interaction happens in attention.

The Evolution of FFN Activations
Original Transformer (2017): ReLU



FFN(x) = ReLU(xW₁ + b₁) @ W₂ + b₂
ReLU(x) = max(0, x)
Simple, but kills negative values entirely. Neurons that output 0 can get "stuck" and never recover (the "dying ReLU" problem).

GPT-2 (2019): GELU



FFN(x) = GELU(xW₁) @ W₂
GELU(x) = x * Φ(x)    (Φ = standard normal CDF)
Smoother than ReLU — instead of a hard cutoff at 0, it gradually tapers near zero. Small negative values get suppressed but not killed.

LLaMA / Modern (2023+): SwiGLU



FFN(x) = (SiLU(xW₁) * xW₃) @ W₂
SiLU(x) = x * sigmoid(x)
The key innovation: gating. Instead of one linear projection + activation, we use TWO projections. One goes through an activation function (the "gate"), the other is raw. We multiply them together element-wise.

Why Gating Works
Think of it this way:



Path 1: xW₁ → SiLU → gate values between 0 and x  (decides WHAT to keep)
Path 2: xW₃ → raw values                           (provides the CONTENT)
Output: gate * content                               (filtered content)
The gate learns to selectively suppress or amplify different features. It's like an attention mechanism WITHIN the FFN — but over features, not positions.

The PaLM paper showed SwiGLU consistently outperforms ReLU and GELU across model sizes.

SiLU (Swish) Activation
SwiGLU uses SiLU (also called Swish) as the gate activation:



SiLU(x) = x * sigmoid(x) = x * (1 / (1 + e^(-x)))
 
x = -5:  SiLU(-5)  = -5 * 0.007   = -0.034  (nearly zero, suppressed)
x = -1:  SiLU(-1)  = -1 * 0.269   = -0.269  (partially suppressed)
x =  0:  SiLU(0)   =  0 * 0.5     =  0.000
x =  1:  SiLU(1)   =  1 * 0.731   =  0.731  (mostly passes through)
x =  5:  SiLU(5)   =  5 * 0.993   =  4.966  (almost fully passes through)
Unlike ReLU (hard cutoff at 0) or GELU (smooth but not gated), SiLU provides a smooth, learnable "valve" that the model controls via the W₁ weights.

FFN Hidden Dimension Sizing
Standard FFN uses d_ff = 4 * d_model (e.g., 2048 for d_model=512).

But SwiGLU has THREE weight matrices instead of two. To keep the same parameter count, LLaMA uses d_ff = (8/3) * d_model:



Standard FFN:  2 matrices × d_model × d_ff = 2 × 512 × 2048 = 2,097,152 params
SwiGLU FFN:    3 matrices × d_model × d_ff = 3 × 512 × 1376 = 2,113,536 params
                                                                  ≈ same!
 
8/3 × 512 = 1365, rounded up to 1376 (multiple of 64, for GPU efficiency)
Rounding to a multiple of 64 (or 128, or 256) matters because GPUs process data in warps/blocks. Dimensions that align with these boundaries get better hardware utilization.

"""

class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation (Gated Linear Unit with Swish).

    Instead of the classic FFN:    ReLU(xW₁) @ W₂
    SwiGLU uses:                   (SiLU(xW₁) * xW₃) @ W₂

    The gating mechanism (W₁ path) learns to selectively filter the
    information (W₃ path), giving the model finer-grained control over
    feature processing.

    Used in: LLaMA 1/2/3, Mistral, Gemma, PaLM, and Pebble.

    Reference: https://arxiv.org/abs/2002.05202 (GLU Variants Improve Transformer)
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: Model dimension (e.g., 512).
            d_ff: FFN hidden dimension (e.g., 1376 for SwiGLU with d_model=512).
                  Typically ≈ (8/3) * d_model, rounded to nearest multiple of 64.
        """
        super().__init__()

        # Gate projection: produces the gating signal
        # The SiLU activation is applied to this path
        self.w1 = nn.Linear(d_model, d_ff, bias=False)

        # Down projection: brings the hidden dimension back to d_model
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

        # Up projection: produces the content signal
        # This path has NO activation — it provides raw features to be gated
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Gate path:    x → W₁ → SiLU        (what to keep)
        # Content path: x → W₃               (what's available)
        # Combined:     gate * content → W₂   (filtered, projected back)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))