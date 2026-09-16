"""
Text generation for Pebble.

Supports multiple sampling strategies:
- Greedy (temperature=0)
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling
- Repetition penalty
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    prompt_tokens: list[int],
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    eos_token_id: int | None = None,
    suppress_tokens: list[int] | None = None,
) -> list[int]:
    """
    Generate text autoregressively.
    
    Args:
        model: The Pebble model (in eval mode, on GPU).
        prompt_tokens: List of token IDs for the prompt.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy, 1 = normal, >1 = creative).
        top_k: Keep only top-k most likely tokens (0 = disabled).
        top_p: Keep smallest set of tokens with cumulative prob >= top_p (1.0 = disabled).
        repetition_penalty: Penalize tokens already in the sequence (1.0 = disabled).
        eos_token_id: Stop generation when this token is produced.
        suppress_tokens: List of token IDs to suppress (set to -inf before sampling).
            Used to prevent the model from generating tool-call tokens when
            a direct response is desired, or vice versa.
    
    Returns:
        List of ALL token IDs (prompt + generated).
    """
    model.eval()
    device = next(model.parameters()).device
    
    tokens = list(prompt_tokens)
    
    for _ in range(max_new_tokens):
        # Crop to max_seq_len if needed (sliding window)
        max_seq_len = model.config.max_seq_len
        input_ids = tokens[-max_seq_len:]
        
        # Forward pass
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(x)                    # (1, seq_len, vocab_size)
        logits = logits[0, -1, :]            # (vocab_size,) — last position only
        
        # === Suppress tokens ===
        if suppress_tokens:
            for tid in suppress_tokens:
                logits[tid] = float("-inf")
        
        # === Repetition penalty ===
        if repetition_penalty != 1.0:
            for prev_token in set(tokens):
                if logits[prev_token] > 0:
                    logits[prev_token] /= repetition_penalty
                else:
                    logits[prev_token] *= repetition_penalty
        
        # === Temperature ===
        if temperature == 0.0:
            # Greedy: pick the highest probability token
            next_token = logits.argmax().item()
            tokens.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break
            continue
        
        logits = logits / temperature
        
        # === Top-k filtering ===
        if top_k > 0:
            # Zero out everything below the top-k threshold
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            threshold = top_k_vals[-1]
            logits[logits < threshold] = float("-inf")
        
        # === Top-p (nucleus) filtering ===
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            # Shift right so that the first token above threshold is kept
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = float("-inf")
            
            # Scatter back to original indexing
            logits = torch.zeros_like(logits).scatter(0, sorted_indices, sorted_logits)
        
        # === Sample ===
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        tokens.append(next_token)
        
        # Stop at EOS
        if eos_token_id is not None and next_token == eos_token_id:
            break
    
    return tokens