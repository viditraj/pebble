"""
Learning rate scheduler: linear warmup + cosine decay.

This is the standard schedule used by GPT-3, LLaMA, and most modern LLMs.
"""

import math


def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """
    Compute learning rate for a given step.
    
    Args:
        step: Current training step (0-indexed).
        max_lr: Peak learning rate (reached after warmup).
        min_lr: Minimum learning rate (floor during cosine decay).
        warmup_steps: Number of steps for linear warmup.
        total_steps: Total number of training steps.
    
    Returns:
        Learning rate for this step.
    """
    # Phase 1: Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # Phase 2: Cosine decay
    if step >= total_steps:
        return min_lr
    
    # Progress through the decay phase (0.0 at warmup_steps, 1.0 at total_steps)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    
    # Cosine decay: starts at max_lr, ends at min_lr
    # cos(0) = 1, cos(pi) = -1
    # We map progress [0,1] to angle [0, pi]
    # Then scale: (1 + cos(angle)) / 2 maps [0,pi] to [1,0]
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return min_lr + coeff * (max_lr - min_lr)