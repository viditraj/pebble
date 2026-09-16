# Pebble: Building a Tool-Calling SLM from Scratch

## Project Goal
Build "Pebble" — a small language model (~25M parameters) from absolute zero that can call tools (function calling), trained entirely on a single machine with an 8GB GPU.

## Hardware Constraints
- GPU: NVIDIA RTX PRO 2000 (8 GB VRAM)
- RAM: 16 GB
- CPU: Intel Core Ultra 7 265H (16 cores)
- Disk: ~890 GB free
- Target model: ~25M params (expandable to ~100M later)

---

## Phase 0: Environment Setup (Day 1)
**Goal**: Get the development environment ready.

### Tasks
- [ ] Install CUDA toolkit (matching driver 13.2)
- [ ] Create Python virtual environment
- [ ] Install PyTorch with CUDA support
- [ ] Install essential libraries: transformers, datasets, tokenizers, wandb, einops, tiktoken
- [ ] Verify GPU works with PyTorch (`torch.cuda.is_available()`)
- [ ] Set up project structure

### What You'll Learn
- CUDA/GPU ecosystem basics
- Python environment management for ML
- How the software stack fits together (CUDA → cuDNN → PyTorch → your code)

### Project Structure
```
pebble/
├── PLAN.md                  # This file
├── AGENTS.md                # Build/run commands
├── configs/                 # Model & training configs
│   ├── pebble_25m.yaml
│   └── pebble_100m.yaml
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py     # Multi-head attention from scratch
│   │   ├── transformer.py   # Transformer block
│   │   ├── embedding.py     # Token + positional embeddings
│   │   ├── pebble.py        # Full model assembly
│   │   └── rope.py          # Rotary positional embeddings
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── bpe.py           # BPE tokenizer from scratch
│   │   └── trainer.py       # Tokenizer training
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # Dataset classes
│   │   ├── preprocessing.py # Data cleaning & formatting
│   │   └── tool_format.py   # Tool-calling data formatting
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop from scratch
│   │   ├── optimizer.py     # AdamW + LR schedulers
│   │   ├── loss.py          # Loss functions
│   │   └── distributed.py   # (future) multi-GPU support
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generate.py      # Text generation (greedy, top-k, top-p)
│   │   ├── kv_cache.py      # KV-cache for fast inference
│   │   └── tool_executor.py # Tool calling execution engine
│   └── evaluation/
│       ├── __init__.py
│       ├── perplexity.py    # Perplexity measurement
│       └── tool_eval.py     # Tool-calling accuracy evaluation
├── data/
│   ├── raw/                 # Raw downloaded data
│   ├── processed/           # Tokenized & ready data
│   └── tool_data/           # Tool-calling training data
├── checkpoints/             # Saved model checkpoints
├── logs/                    # Training logs
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Evaluation script
│   ├── generate.py          # Interactive generation
│   └── export.py            # Model export (ONNX, GGUF)
└── tests/
    ├── test_attention.py
    ├── test_tokenizer.py
    └── test_model.py
```

---

## Phase 1: Foundations — Understanding Transformers (Days 2-5)
**Goal**: Deeply understand every component of a Transformer by implementing each piece from scratch.

### 1.1 Token Embeddings (Day 2)
**Theory to learn**:
- What are embeddings? Why do we need them?
- How does a lookup table (nn.Embedding) work?
- Embedding dimension choices and trade-offs
- Weight tying (sharing embedding and output projection weights)

**Implementation**:
- Build `src/model/embedding.py`
- Create token embedding layer
- Understand the math: one-hot → dense vector mapping
- Write tests to verify dimensions and gradients

**Concepts**:
```
Token "hello" → ID 4523 → Vector [0.12, -0.34, 0.56, ...] (d_model dimensions)
```

### 1.2 Positional Encoding — RoPE (Day 2-3)
**Theory to learn**:
- Why do transformers need position information? (permutation invariance)
- Evolution: sinusoidal → learned → ALiBi → RoPE
- RoPE (Rotary Position Embeddings): how rotation in 2D encodes position
- Why RoPE is the modern standard (used in LLaMA, Mistral, etc.)

**Implementation**:
- Build `src/model/rope.py`
- Implement rotary embedding computation
- Apply RoPE to query and key vectors
- Verify that relative position information is preserved

**Key math**:
```
RoPE rotates pairs of dimensions by angle θ * position:
q_rotated = q * cos(θ*pos) + rotate_half(q) * sin(θ*pos)
```

### 1.3 Self-Attention Mechanism (Day 3)
**Theory to learn**:
- Attention as "soft dictionary lookup": Query, Key, Value
- Scaled dot-product attention: why we scale by √d_k
- Causal (autoregressive) masking for language modeling
- Multi-Head Attention: why multiple heads? what do different heads learn?
- Grouped Query Attention (GQA): memory-efficient alternative used in modern models

**Implementation**:
- Build `src/model/attention.py`
- Implement single-head attention first (for understanding)
- Extend to multi-head attention
- Implement GQA (fewer KV heads than Q heads)
- Add causal mask
- Integrate RoPE into Q, K projections

**Key formula**:
```
Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
```

### 1.4 Feed-Forward Network with SwiGLU (Day 3-4)
**Theory to learn**:
- Role of FFN in transformers (information processing after attention)
- Activation functions: ReLU → GELU → SwiGLU
- Why SwiGLU works better (gated linear units)
- FFN hidden dimension sizing (typically 4x or 8/3x model dim)

**Implementation**:
```python
# SwiGLU: output = (xW₁ * sigmoid(xW₁)) * xW₂  (simplified)
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

### 1.5 Layer Normalization — RMSNorm (Day 4)
**Theory to learn**:
- Why normalization is needed (training stability)
- LayerNorm vs BatchNorm vs RMSNorm
- Pre-norm vs post-norm architecture (pre-norm is modern standard)
- RMSNorm: simpler, faster, works just as well

**Implementation**:
```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight
```

### 1.6 Full Transformer Block & Model Assembly (Day 4-5)
**Theory to learn**:
- Residual connections: why they're critical (gradient flow)
- Stacking transformer blocks
- Output head: projecting back to vocabulary
- Weight initialization strategies (Xavier, Kaiming, scaled init)

**Implementation**:
- Build `src/model/transformer.py` (single block)
- Build `src/model/pebble.py` (full model)
- Write comprehensive tests
- Count parameters and verify architecture

### Pebble-25M Architecture
```
┌─────────────────────────────┐
│ Token Embedding (vocab → d) │  vocab_size=16384, d_model=512
├─────────────────────────────┤
│ Transformer Block ×8        │  8 layers
│ ├─ RMSNorm                  │
│ ├─ GQA Attention            │  8 Q-heads, 4 KV-heads, head_dim=64
│ ├─ Residual Connection      │
│ ├─ RMSNorm                  │
│ ├─ SwiGLU FFN               │  d_ff=1376 (≈ 8/3 × 512)
│ └─ Residual Connection      │
├─────────────────────────────┤
│ RMSNorm                     │
│ Output Projection (d → vocab)│  weight tied with embedding
└─────────────────────────────┘

Total: ~25M parameters
Context length: 1024 tokens (expandable to 2048)
```

---

## Phase 2: Tokenizer — BPE from Scratch (Days 6-8)
**Goal**: Build a Byte-Pair Encoding tokenizer from the ground up.

### 2.1 Understanding Tokenization (Day 6)
**Theory to learn**:
- Why tokenization matters (vocabulary size vs sequence length trade-off)
- Character-level vs word-level vs subword tokenization
- BPE algorithm: how it works step by step
- Vocabulary size considerations (we'll use 16,384 for our small model)
- Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`, `<tool_call>`, `<tool_result>`

### 2.2 BPE Implementation (Day 6-7)
**Implementation**:
- Build `src/tokenizer/bpe.py`
  - Start from UTF-8 bytes (256 base tokens)
  - Iteratively find most frequent byte pair
  - Merge pairs to build vocabulary
  - Encode: text → token IDs
  - Decode: token IDs → text
- Build `src/tokenizer/trainer.py`
  - Train tokenizer on our dataset
  - Save/load vocabulary and merge rules

**Algorithm**:
```
1. Start with character-level tokens
2. Count all adjacent pairs
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches target size
5. Result: merge rules that define the tokenizer
```

### 2.3 Special Tokens for Tool Calling (Day 7-8)
**Implementation**:
- Add special tokens for tool-calling format:
  ```
  <|tool_call_start|>
  <|tool_call_end|>
  <|tool_result_start|>
  <|tool_result_end|>
  <|function_name|>
  <|parameters|>
  ```
- Test encoding/decoding with tool-call formatted text
- Compare our tokenizer against HuggingFace's tokenizers library

---

## Phase 3: Data Pipeline (Days 9-14)
**Goal**: Collect, clean, and prepare training data.

### 3.1 Understanding Training Data (Day 9)
**Theory to learn**:
- What makes good training data for LLMs?
- Data quality vs quantity trade-offs
- Deduplication and filtering
- Data mixing strategies
- Tokenized data storage formats

### 3.2 Dataset Collection (Day 9-10)
**Datasets to use** (all open-source, feasible for our GPU):

| Dataset | Size (tokens) | Purpose |
|---------|---------------|---------|
| TinyStories | ~500M tokens | Basic language understanding |
| OpenWebText subset | ~200M tokens | General knowledge |
| The Stack (Python subset, small) | ~100M tokens | Code understanding |
| **Total target** | **~500M-800M tokens** | Multiple epochs over this |

**Note**: We'll train for ~2-5B tokens total (multiple epochs). At ~25M params, the Chinchilla-optimal would be ~500M tokens, but we'll go a bit over since data is "free" to reuse.

### 3.3 Data Preprocessing Pipeline (Day 10-12)
**Implementation**:
- Build `src/data/preprocessing.py`
  - Download and extract datasets
  - Clean text (remove HTML, normalize unicode, etc.)
  - Filter low-quality documents
  - Deduplicate
- Build `src/data/dataset.py`
  - Tokenize all data with our BPE tokenizer
  - Pack sequences to max context length (1024)
  - Create efficient memory-mapped dataset (numpy memmap)
  - Train/validation split

### 3.4 Tool-Calling Data Creation (Day 12-14)
**Theory to learn**:
- How tool-calling works in modern LLMs
- Function calling format (OpenAI-style, Anthropic-style)
- Synthetic data generation for tool calling

**Implementation**:
- Build `src/data/tool_format.py`
- Design Pebble's tool-calling format:
  ```
  System: You have access to the following tools:
  - calculator(expression: str) -> float
  - weather(city: str) -> str
  - search(query: str) -> str

  User: What's 25 * 47?
  Assistant: <|tool_call_start|>calculator<|parameters|>{"expression": "25 * 47"}<|tool_call_end|>
  <|tool_result_start|>1175<|tool_result_end|>
  The result of 25 × 47 is 1,175.
  ```
- Create ~50K-100K synthetic tool-calling examples
- Include: calculator, unit conversion, date/time, simple lookup tools
- Mix tool-calling data with general language data (~10-20% tool data)

---

## Phase 4: Training Loop from Scratch (Days 15-22)
**Goal**: Build the complete training infrastructure.

### 4.1 Understanding Training (Day 15-16)
**Theory to learn**:
- Loss function: Cross-entropy for next-token prediction
- How backpropagation works through a transformer
- Gradient accumulation (simulate larger batch sizes)
- Mixed precision training (FP16/BF16) — critical for 8GB VRAM
- Gradient clipping and why it's needed
- Learning rate schedules: warmup + cosine decay

### 4.2 Optimizer — AdamW (Day 16-17)
**Theory to learn**:
- SGD → Momentum → Adam → AdamW
- Why weight decay is important (regularization)
- AdamW: decoupled weight decay
- Hyperparameter choices: lr, betas, eps, weight_decay
- Learning rate warmup: why and how

**Implementation**:
- Build `src/training/optimizer.py`
- Implement AdamW from scratch (for learning)
- Implement cosine annealing with warmup schedule
- Then switch to `torch.optim.AdamW` for actual training (better fused kernels)

### 4.3 Training Loop (Day 17-20)
**Implementation**:
- Build `src/training/trainer.py`
  - Data loading with proper batching
  - Forward pass → loss computation
  - Backward pass → gradient computation
  - Gradient clipping
  - Optimizer step
  - Learning rate scheduling
  - Mixed precision with `torch.cuda.amp`
  - Gradient accumulation
  - Checkpointing (save/resume)
  - Logging (loss, learning rate, throughput, GPU memory)
  - Validation loop
  - WandB integration for experiment tracking

### Training Hyperparameters (Pebble-25M)
```yaml
# configs/pebble_25m.yaml
model:
  vocab_size: 16384
  d_model: 512
  n_layers: 8
  n_heads: 8
  n_kv_heads: 4
  d_ff: 1376
  max_seq_len: 1024
  dropout: 0.0        # Modern practice: no dropout for pretraining

training:
  batch_size: 32       # Per-GPU micro batch
  gradient_accumulation_steps: 4  # Effective batch: 32 * 4 = 128
  max_lr: 3e-4
  min_lr: 3e-5
  warmup_steps: 1000
  total_steps: 50000   # ~6.5B tokens seen
  weight_decay: 0.1
  grad_clip: 1.0
  precision: "bf16"    # or fp16 if bf16 not supported
  
  # Checkpointing
  save_every: 5000
  eval_every: 500
  log_every: 10
```

### 4.4 Memory Optimization (Day 20-22)
**Theory to learn**:
- GPU memory breakdown: model weights + activations + gradients + optimizer states
- Gradient checkpointing (recompute vs store activations)
- Mixed precision: FP32 master weights, BF16 compute
- Memory-efficient attention (Flash Attention)

**Implementation**:
- Profile GPU memory usage at each stage
- Implement gradient checkpointing
- Install and integrate Flash Attention 2
- Verify training fits in 8GB VRAM

**Memory Budget (~8GB)**:
```
Model weights (BF16):     ~50 MB  (25M × 2 bytes)
Optimizer states (FP32):  ~200 MB (25M × 8 bytes for AdamW)
Gradients (BF16):         ~50 MB
Activations:              ~2-4 GB (depends on batch size & seq len)
CUDA overhead:            ~500 MB
─────────────────────────────────
Total:                    ~3-5 GB ✓ (fits with room to spare)
```

---

## Phase 5: Pretraining Run (Days 23-30)
**Goal**: Actually train Pebble on general text data.

### 5.1 Initial Training Run (Day 23-25)
- Start with a small test run (100 steps) to verify everything works
- Monitor loss curve, learning rate, GPU utilization
- Debug any issues (NaN loss, OOM, slow throughput)
- Target throughput: ~5,000-10,000 tokens/second on RTX PRO 2000

### 5.2 Full Pretraining (Day 25-30)
- Run full 50K step training (~24-48 hours of GPU time)
- Monitor via WandB dashboard
- Expected loss curve:
  ```
  Step 0:      ~9.7 (ln(16384) ≈ random)
  Step 1000:   ~5.5
  Step 5000:   ~4.0
  Step 10000:  ~3.5
  Step 25000:  ~3.0
  Step 50000:  ~2.7-2.9
  ```
- Save checkpoints at regular intervals

### 5.3 Evaluation During Training (Day 28-30)
**Implementation**:
- Build `src/evaluation/perplexity.py`
- Track validation perplexity
- Generate sample text at checkpoints to qualitatively assess
- Compare with known benchmarks for models of this size

---

## Phase 6: Supervised Fine-Tuning (SFT) for Tool Calling (Days 31-38)
**Goal**: Fine-tune pretrained Pebble to follow instructions and call tools.

### 6.1 Understanding Fine-Tuning (Day 31-32)
**Theory to learn**:
- Pretraining vs Fine-tuning: what changes?
- Full fine-tuning vs parameter-efficient methods (LoRA, QLoRA)
- Chat/instruction format templates
- Loss masking: only compute loss on assistant responses
- Overfitting risks in fine-tuning

### 6.2 Instruction Dataset Preparation (Day 32-34)
**Data format**:
```json
{
  "messages": [
    {"role": "system", "content": "You are Pebble, a helpful assistant with tool access."},
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"name": "weather", "arguments": {"city": "Tokyo"}}
    ]},
    {"role": "tool", "content": "72°F, partly cloudy"},
    {"role": "assistant", "content": "The weather in Tokyo is currently 72°F and partly cloudy."}
  ]
}
```

**Datasets**:
- Create ~50K instruction-following examples
- Create ~30K tool-calling examples (synthetic)
- Include multi-turn conversations
- Include "no tool needed" examples (so model doesn't always call tools)

### 6.3 SFT Training (Day 34-38)
**Implementation**:
- Modify trainer for SFT:
  - Chat template formatting
  - Loss masking (only on assistant tokens)
  - Lower learning rate (1e-5 to 5e-5)
  - Fewer steps (~2000-5000)
  - Evaluate tool-calling accuracy

### SFT Hyperparameters
```yaml
sft:
  learning_rate: 2e-5
  epochs: 3
  batch_size: 16
  gradient_accumulation_steps: 4
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_seq_len: 1024
  loss_on_assistant_only: true
```

---

## Phase 7: RLHF / DPO Alignment (Days 39-45)
**Goal**: Align Pebble to be more helpful and produce better tool calls.

### 7.1 Understanding Alignment (Day 39-40)
**Theory to learn**:
- Why alignment matters (helpful, harmless, honest)
- RLHF: Reward model → PPO training
- DPO (Direct Preference Optimization): simpler alternative
- Why DPO is preferred for small-scale (no separate reward model needed)
- Preference data format: chosen vs rejected responses

### 7.2 DPO Data Creation (Day 40-42)
**Format**:
```json
{
  "prompt": "User: What's 15% of 230?",
  "chosen": "<|tool_call_start|>calculator<|parameters|>{\"expression\": \"0.15 * 230\"}<|tool_call_end|>...",
  "rejected": "15% of 230 is approximately 35."  // wrong answer, no tool use
}
```
- Create ~10K preference pairs
- Focus on tool-calling quality

### 7.3 DPO Training (Day 42-45)
**Implementation**:
- Implement DPO loss function
- Train with preference data
- Evaluate improvement in tool-calling accuracy
- Compare before/after DPO

**DPO Loss**:
```python
# Simplified DPO loss
loss = -log_sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
# where log_ratio = log_pi(y|x) - log_pi_ref(y|x)
```

---

## Phase 8: Inference Engine (Days 46-52)
**Goal**: Build efficient inference for Pebble.

### 8.1 Text Generation (Day 46-48)
**Theory to learn**:
- Autoregressive generation: one token at a time
- Sampling strategies: greedy, top-k, top-p (nucleus), temperature
- Repetition penalty
- KV-Cache: why it makes inference O(n) instead of O(n²)

**Implementation**:
- Build `src/inference/generate.py`
  - Greedy decoding
  - Temperature scaling
  - Top-k sampling
  - Top-p (nucleus) sampling
  - Repetition penalty
- Build `src/inference/kv_cache.py`
  - Implement KV-cache for efficient autoregressive generation
  - Profile speedup vs naive generation

### 8.2 Tool Execution Engine (Day 48-50)
**Implementation**:
- Build `src/inference/tool_executor.py`
  - Parse tool calls from model output
  - Execute tool functions
  - Format tool results back into context
  - Handle multi-step tool chains
  - Build a simple tool registry

**Tool Registry Example**:
```python
tools = {
    "calculator": {
        "description": "Evaluate a math expression",
        "parameters": {"expression": "str"},
        "function": lambda expr: eval(expr)  # (use safe_eval in practice)
    },
    "weather": {
        "description": "Get weather for a city",
        "parameters": {"city": "str"},
        "function": get_weather
    }
}
```

### 8.3 Interactive Chat Interface (Day 50-52)
**Implementation**:
- Build `scripts/generate.py` — terminal chat interface
- Support conversation history
- Tool calling in the loop
- Streaming output (token by token)

---

## Phase 9: Optimization & Export (Days 53-58)
**Goal**: Make Pebble fast and portable.

### 9.1 Quantization (Day 53-55)
**Theory to learn**:
- What is quantization? FP32 → FP16 → INT8 → INT4
- Post-training quantization (PTQ) vs Quantization-aware training (QAT)
- GPTQ, AWQ, GGML/GGUF quantization methods
- Quality vs speed trade-offs at different bit widths

**Implementation**:
- Implement simple INT8 quantization from scratch
- Use `bitsandbytes` for INT4 quantization
- Export to GGUF format (for llama.cpp compatibility)
- Benchmark: speed and quality at each quantization level

### 9.2 Model Export (Day 55-57)
**Implementation**:
- Export to HuggingFace format (config.json + model weights)
- Export to ONNX for cross-platform inference
- Export to GGUF for llama.cpp
- Create a HuggingFace model card

### 9.3 Benchmarking (Day 57-58)
- Measure tokens/second at various precisions
- Measure memory usage
- Compare with similar-sized models
- Document results

---

## Phase 10: Advanced Topics & Scaling Up (Days 59-70+)
**Goal**: Explore advanced concepts and potentially train a larger model.

### 10.1 Scale to Pebble-100M (Day 59-63)
- Increase model dimensions:
  ```
  d_model: 768, n_layers: 12, n_heads: 12, n_kv_heads: 4
  ```
- Use gradient checkpointing to fit in 8GB VRAM
- Train on larger dataset (~2B tokens)
- Compare quality improvement from scaling

### 10.2 LoRA & QLoRA (Day 63-65)
**Theory to learn**:
- Low-Rank Adaptation: why it works
- QLoRA: 4-bit quantized base + LoRA adapters
- When to use LoRA vs full fine-tuning

**Implementation**:
- Implement LoRA from scratch
- Fine-tune Pebble-100M with LoRA
- Compare parameter count and quality

### 10.3 Speculative Decoding (Day 65-67)
**Theory to learn**:
- Draft model + verifier model
- How it achieves 2-3x speedup
- When it works well vs poorly

### 10.4 Knowledge Distillation (Day 67-69)
**Theory to learn**:
- Teacher-student framework
- Distilling from a larger model (e.g., use a 7B model as teacher)
- KL-divergence loss between teacher and student logits

### 10.5 Deployment (Day 69-70+)
- Build a simple REST API (FastAPI)
- Add streaming support (Server-Sent Events)
- Containerize with Docker
- Document everything

---

## Key Concepts You'll Master

By the end of this project, you will understand:

### Architecture
- [x] Transformer architecture (every component)
- [x] Attention mechanisms (MHA, GQA, MQA)
- [x] Positional encodings (sinusoidal, learned, RoPE, ALiBi)
- [x] Normalization (LayerNorm, RMSNorm)
- [x] Activation functions (ReLU, GELU, SwiGLU)

### Training
- [x] Loss functions (cross-entropy, DPO loss)
- [x] Optimizers (SGD, Adam, AdamW)
- [x] Learning rate schedules (warmup, cosine decay)
- [x] Mixed precision training (FP16/BF16)
- [x] Gradient accumulation & checkpointing
- [x] Training stability techniques

### Data
- [x] Tokenization (BPE from scratch)
- [x] Data preprocessing and cleaning
- [x] Dataset creation and formatting
- [x] Tool-calling data synthesis

### Fine-tuning & Alignment
- [x] Supervised Fine-Tuning (SFT)
- [x] DPO alignment
- [x] LoRA / QLoRA
- [x] Knowledge distillation

### Inference
- [x] Autoregressive generation
- [x] Sampling strategies (top-k, top-p, temperature)
- [x] KV-Cache
- [x] Quantization (INT8, INT4, GGUF)
- [x] Tool calling and execution

### Engineering
- [x] GPU memory management
- [x] Training infrastructure
- [x] Experiment tracking
- [x] Model export and deployment

---

## Resources

### Papers to Read (in order)
1. "Attention Is All You Need" (Vaswani et al., 2017) — The OG Transformer paper
2. "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) — GPT-2
3. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
4. "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
5. "Training Compute-Optimal LLMs" (Hoffmann et al., 2022) — Chinchilla scaling laws
6. "Direct Preference Optimization" (Rafailov et al., 2023) — DPO
7. "LoRA: Low-Rank Adaptation of LLMs" (Hu et al., 2021)
8. "Toolformer: LLMs Can Teach Themselves to Use Tools" (Schick et al., 2023)

### Code References
- Karpathy's nanoGPT: Minimal GPT implementation
- LitGPT: Clean LLaMA-style implementation
- llama.cpp: Efficient inference reference

---

## Timeline Summary

| Phase | Topic | Days | Cumulative |
|-------|-------|------|------------|
| 0 | Environment Setup | 1 | 1 |
| 1 | Transformer Components | 4 | 5 |
| 2 | BPE Tokenizer | 3 | 8 |
| 3 | Data Pipeline | 6 | 14 |
| 4 | Training Loop | 8 | 22 |
| 5 | Pretraining | 8 | 30 |
| 6 | SFT for Tool Calling | 8 | 38 |
| 7 | DPO Alignment | 7 | 45 |
| 8 | Inference Engine | 7 | 52 |
| 9 | Optimization & Export | 6 | 58 |
| 10 | Advanced Topics | 12+ | 70+ |

**Total estimated time: ~10 weeks** (working a few hours daily)

---

## Backlog (Post-Course Improvements)

These are improvements to revisit after completing all phases:

### 1. Remove Tool Calling Completely (Simplify)
- Strip tool-calling from the model entirely
- Focus on pure instruction-following and natural language
- Simpler task = better performance at 31M params
- Re-add tool calling only when model is scaled up

### 2. Scale Up to 60-80M Parameters
- Increase model dimensions (e.g., d_model=768, n_layers=12)
- More capacity = better at juggling tool calls + natural language
- Requires full pretraining + SFT again (~2-3 days on current GPU)
- Should resolve the core issue: model can't reliably distinguish
  "use tool" vs "respond directly" at 31M params

### Current SFT Results (31M params)
- Tool-call routing accuracy: 70% (7/10 test cases)
- Model correctly produces JSON tool calls with right tool names and arguments
- Model struggles with: (1) over-eager tool calling for simple queries,
  (2) garbled natural language responses after tool results
- Root cause: 31M params insufficient for multi-mode generation
  (tool JSON + natural language in same conversation)

---

*Let's build something beautiful, one pebble at a time.* 🪨
