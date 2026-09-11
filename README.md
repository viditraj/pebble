<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/pebble-banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/pebble-banner-light.svg">
    <img alt="Pebble" src="assets/pebble-banner-dark.svg" width="600">
  </picture>
</p>

<h3 align="center">Build a tool-calling language model from absolute zero.</h3>

<p align="center">
  <em>No frameworks. No magic. Just math, code, and a single GPU.</em>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-5_min-blue?style=flat-square" alt="Quick Start"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.0+"></a>
</p>

<p align="center">
  <a href="#what-is-pebble">What is Pebble</a> &bull;
  <a href="#what-youll-build">What You'll Build</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#the-curriculum">Curriculum</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#tool-calling">Tool Calling</a> &bull;
  <a href="#faq">FAQ</a>
</p>

---

## What is Pebble

Pebble is an open-source, from-scratch implementation of a **25M-parameter language model that can call tools** — built as a hands-on tutorial for anyone who wants to truly understand how LLMs work, not just use them.

No wrappers around HuggingFace. No fine-tuning someone else's model and calling it yours. You write **every layer**, train on **your own data**, and watch a pile of random weights learn to use a calculator.

```
You:     What's 47 * 89?
Pebble:  <|tool_call|> calculator(expression="47 * 89") <|end|>
         <|result|> 4183 <|end|>
         47 multiplied by 89 is 4,183.
```

### Why Pebble exists

There are mass of blog posts explaining attention "intuitively." There are massive repos with 10,000-line training frameworks. Pebble sits in the gap between the two: **production-grade concepts, implemented simply enough to fit in your head.**

After completing this project, you will understand:
- How a transformer works — not the hand-wavy version, the actual matrix multiplications
- How BPE tokenization breaks text into tokens (you'll build one)
- How pretraining, SFT, and DPO alignment work end-to-end
- How tool calling is just clever formatting + fine-tuning
- How to train, evaluate, quantize, and deploy a model on consumer hardware

You'll go from `torch.randn` to a working tool-calling chatbot.

---

## What You'll Build

Every component is implemented from scratch in clean, documented PyTorch:

| Component | What You'll Implement | Key Concepts |
|-----------|----------------------|--------------|
| **Tokenizer** | Byte-Pair Encoding (BPE) from raw bytes | Subword tokenization, merge rules, special tokens |
| **Embeddings** | Token + Rotary Position Embeddings (RoPE) | Lookup tables, rotation matrices, relative position |
| **Attention** | Grouped Query Attention (GQA) with causal mask | Q/K/V projections, scaled dot-product, KV-head sharing |
| **FFN** | SwiGLU feed-forward network | Gated linear units, activation functions |
| **Normalization** | RMSNorm (pre-norm architecture) | Training stability, why RMSNorm > LayerNorm |
| **Training Loop** | AdamW + cosine schedule + mixed precision | Gradient accumulation, checkpointing, AMP |
| **Data Pipeline** | Preprocessing, packing, memory-mapped datasets | Data quality, deduplication, sequence packing |
| **SFT** | Supervised fine-tuning with loss masking | Chat templates, instruction following |
| **DPO** | Direct Preference Optimization alignment | Preference learning, reward-free alignment |
| **Inference** | KV-cache + top-k/top-p sampling | Autoregressive generation, sampling strategies |
| **Tool Calling** | Function parsing + execution engine | Tool registry, structured output, multi-step chains |
| **Quantization** | INT8/INT4 post-training quantization | Model compression, GGUF export |

---

## Quick Start

### Requirements

- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM (tested on RTX PRO 2000)
- CUDA 12.0+
- ~10 GB disk space for datasets

> **No GPU?** The model architecture and tokenizer code works on CPU.  
> Pretraining will be slow but possible for small experiments.

### Setup

```bash
git clone https://github.com/viditraj/pebble.git
cd pebble
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### Train Pebble (25M)

```bash
# 1. Train the tokenizer
python scripts/train_tokenizer.py --vocab-size 16384 --data data/raw/

# 2. Preprocess data
python scripts/preprocess.py --tokenizer checkpoints/tokenizer.json --out data/processed/

# 3. Pretrain
python scripts/train.py --config configs/pebble_25m.yaml

# 4. Fine-tune for tool calling
python scripts/finetune.py --config configs/sft_tools.yaml --checkpoint checkpoints/pretrain/latest.pt

# 5. Chat with Pebble
python scripts/generate.py --checkpoint checkpoints/sft/latest.pt --tools
```

---

## The Curriculum

Pebble is structured as a learning journey. Each module builds on the last, with heavily commented code that explains the *why*, not just the *how*.

### 1. Tokenizer &mdash; `src/tokenizer/`

> *"You can't feed text to a neural network. You feed numbers."*

Build a BPE tokenizer from raw bytes. No libraries. You'll understand why "tokenization" shows up as 3 tokens, why LLMs struggle with counting letters, and why vocabulary size is a critical design choice.

```python
tokenizer = BPETokenizer.train(corpus, vocab_size=16384)
tokens = tokenizer.encode("Hello, world!")  # [1762, 44, 1452, 33]
text = tokenizer.decode(tokens)             # "Hello, world!"
```

**You'll implement**: byte-level BPE training, merge rules, encode/decode, special tokens for tool calling (`<|tool_call|>`, `<|result|>`, etc.)

<details>
<summary><b>Key insight: Why BPE?</b></summary>

Character-level tokenization creates sequences that are too long. Word-level creates vocabularies that are too large and can't handle new words. BPE finds the sweet spot: common words become single tokens, rare words get split into meaningful subwords. The word "unhappiness" might become `["un", "happiness"]` — and the model learns that "un-" means negation.

</details>

---

### 2. Model Architecture &mdash; `src/model/`

> *"A transformer is just attention + feed-forward, repeated."*

Build a modern LLaMA-style transformer. Not GPT-2's architecture from 2019 — the architecture actually used in today's best open models.

```
Input Token IDs
       |
  [ Embedding ] ──────────────────────────────────────────┐
       |                                                   |
       v                                            (weight tying)
  ┌─────────────────────┐                                  |
  │  Transformer Block  │ x8                               |
  │  ┌───────────────┐  │                                  |
  │  │    RMSNorm     │  │                                  |
  │  │       |        │  │                                  |
  │  │  GQA Attention │  │  8 Q-heads, 4 KV-heads          |
  │  │  + RoPE        │  │  head_dim = 64                  |
  │  │       |        │  │                                  |
  │  │   + Residual   │  │                                  |
  │  │       |        │  │                                  |
  │  │    RMSNorm     │  │                                  |
  │  │       |        │  │                                  |
  │  │  SwiGLU FFN    │  │  d_ff = 1376                    |
  │  │       |        │  │                                  |
  │  │   + Residual   │  │                                  |
  │  └───────────────┘  │                                  |
  └─────────────────────┘                                  |
       |                                                   |
  [ RMSNorm ]                                              |
       |                                                   |
  [ Linear Head ] ─────────────────────────────────────────┘
       |
  Output Logits (vocab_size)
```

**You'll implement:**
- **RoPE** (Rotary Positional Embeddings) — how rotation encodes position
- **GQA** (Grouped Query Attention) — the same trick LLaMA 2/3 and Mistral use to cut memory usage
- **SwiGLU** — the gated activation that outperforms ReLU and GELU
- **RMSNorm** — simpler and faster than LayerNorm
- **Weight tying** — sharing the embedding and output projection saves millions of parameters

<details>
<summary><b>Key insight: Why GQA instead of standard Multi-Head Attention?</b></summary>

Standard MHA uses separate Key and Value projections for every head. With 8 heads, that's 8 sets of K and V matrices. GQA shares K/V across groups of heads — our model uses 8 query heads but only 4 KV heads. This halves the KV-cache memory during inference with negligible quality loss. It's the same reason LLaMA 2 70B can fit on 2 GPUs instead of 4.

</details>

---

### 3. Training &mdash; `src/training/`

> *"Training is just: predict the next token, compute how wrong you were, adjust weights. Repeat 50,000 times."*

Build the complete training pipeline from scratch — the optimizer, the scheduler, the training loop, mixed-precision, gradient accumulation, checkpointing, and logging.

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

training:
  batch_size: 32
  gradient_accumulation_steps: 4    # effective batch = 128
  max_lr: 3e-4
  min_lr: 3e-5
  warmup_steps: 1000
  total_steps: 50000
  weight_decay: 0.1
  grad_clip: 1.0
  precision: bf16
```

**You'll implement:**
- **AdamW** from scratch (then use PyTorch's for speed)
- **Cosine annealing** with warmup — the standard LR schedule
- **Mixed precision** (BF16) — halves memory, doubles throughput
- **Gradient accumulation** — simulate large batches on small GPUs
- **Gradient checkpointing** — trade compute for memory

<details>
<summary><b>Key insight: Memory budget on 8GB VRAM</b></summary>

```
Model weights (BF16):     ~50 MB   (25M x 2 bytes)
Optimizer states (FP32):  ~200 MB  (25M x 8 bytes — AdamW stores m and v)
Gradients (BF16):         ~50 MB
Activations:              ~2-4 GB  (batch_size x seq_len x d_model x layers)
CUDA overhead:            ~500 MB
─────────────────────────────────────
Total:                    ~3-5 GB   (fits comfortably in 8GB)
```

The model is small. The activations are what eat your memory. That's why gradient checkpointing and batch size tuning matter more than model size at this scale.

</details>

---

### 4. Data &mdash; `src/data/`

> *"Your model is only as good as the data you feed it."*

Build the full data pipeline: download, clean, tokenize, pack, and serve sequences efficiently.

| Dataset | Tokens | Purpose |
|---------|--------|---------|
| TinyStories | ~500M | Coherent narrative and basic grammar |
| OpenWebText (subset) | ~200M | General world knowledge |
| The Stack (Python, subset) | ~100M | Code structure and logic |
| Synthetic tool-calling | ~20M | Function calling capability |
| **Total** | **~800M** | |

**You'll implement:**
- Data downloading and cleaning (HTML removal, unicode normalization, deduplication)
- Efficient tokenization with your BPE tokenizer
- Sequence packing to minimize padding waste
- Memory-mapped datasets (numpy memmap) — load TB-scale data without RAM limits
- Data mixing strategies (how much code vs text vs tool-calling data)

---

### 5. Fine-Tuning &mdash; SFT + DPO

> *"Pretraining teaches the model language. Fine-tuning teaches it to be useful."*

#### Supervised Fine-Tuning (SFT)

Turn the base model into a chat assistant that follows instructions and calls tools.

```json
{
  "messages": [
    {"role": "system", "content": "You are Pebble. Use tools when helpful."},
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "tool_calls": [{"name": "weather", "args": {"city": "Tokyo"}}]},
    {"role": "tool", "content": "72F, partly cloudy"},
    {"role": "assistant", "content": "It's 72F and partly cloudy in Tokyo right now."}
  ]
}
```

**You'll implement**: chat template formatting, loss masking (train only on assistant tokens), lower learning rate fine-tuning.

#### Direct Preference Optimization (DPO)

Align the model to prefer correct tool usage over hallucinated answers — without needing a separate reward model.

```python
# The DPO loss is elegant in its simplicity:
loss = -log_sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
```

**You'll implement**: preference data creation, reference model caching, DPO training loop.

---

### 6. Inference Engine &mdash; `src/inference/`

> *"A trained model is useless if you can't run it."*

Build a fast inference engine with KV-caching, multiple sampling strategies, and a tool execution runtime.

```python
from pebble import Pebble

model = Pebble.load("checkpoints/sft/latest.pt")

# Basic generation
print(model.generate("The capital of France is"))

# Chat with tool calling
response = model.chat(
    messages=[{"role": "user", "content": "What's 15% of 230?"}],
    tools=[calculator, weather, search],
    temperature=0.7,
    top_p=0.9,
)
```

**You'll implement:**
- **KV-Cache** — makes generation O(n) per token instead of O(n^2)
- **Sampling** — temperature, top-k, top-p (nucleus), repetition penalty
- **Tool executor** — parse model output, call functions, feed results back
- **Streaming** — token-by-token output for real-time chat

---

### 7. Optimization &mdash; Quantization & Export

> *"The best model is the one that actually runs on your hardware."*

Shrink Pebble from 50 MB to 12 MB with minimal quality loss.

```bash
# Quantize to INT4
python scripts/export.py --checkpoint checkpoints/sft/latest.pt --format gguf --bits 4

# Run with llama.cpp
./llama-cli -m pebble-25m-q4.gguf -p "Hello, Pebble!"
```

**You'll implement**: INT8/INT4 post-training quantization, GGUF export for llama.cpp, ONNX export for cross-platform inference.

---

## Architecture

### Pebble-25M

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| Parameters | ~25M | Trainable on a single 8GB GPU in ~24-48 hrs |
| Layers | 8 | Deep enough to learn language patterns |
| Hidden dim | 512 | Balanced compute-to-parameter ratio |
| Attention heads | 8 (Q) / 4 (KV) | GQA for memory efficiency |
| Head dim | 64 | Standard, works well with Flash Attention |
| FFN dim | 1376 | ~8/3 x hidden (SwiGLU sizing) |
| Vocab size | 16,384 | Small but effective for English + code |
| Context length | 1024 | Expandable to 2048 via RoPE |
| Positional encoding | RoPE | Modern standard, relative position aware |
| Normalization | RMSNorm (pre-norm) | Simpler, faster than LayerNorm |
| Activation | SwiGLU | Better than ReLU/GELU for transformers |

### Pebble-100M (Stretch Goal)

| Hyperparameter | Value |
|----------------|-------|
| Parameters | ~100M |
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 (Q) / 4 (KV) |
| FFN dim | 2048 |
| Context length | 2048 |

Trainable on the same hardware with gradient checkpointing.

---

## Tool Calling

Pebble's tool-calling system is designed to be simple and extensible.

### Defining Tools

```python
from pebble.tools import Tool

calculator = Tool(
    name="calculator",
    description="Evaluate a mathematical expression",
    parameters={"expression": {"type": "string", "description": "Math expression to evaluate"}},
    function=lambda expr: str(eval(expr))  # use safe_eval in production
)

weather = Tool(
    name="weather",
    description="Get current weather for a city",
    parameters={"city": {"type": "string"}},
    function=get_weather_api
)
```

### How It Works

```
                        ┌──────────────────────┐
   User message ──────> │                      │
                        │    Pebble Model       │
   Tool results ──────> │                      │
                        └──────────┬───────────┘
                                   │
                          Generates tokens
                                   │
                      ┌────────────┴────────────┐
                      │                         │
                Regular text              <|tool_call|>
                      │                         │
                      v                         v
                Print to user           ┌───────────────┐
                                        │ Tool Executor  │
                                        │ Parse function │
                                        │ + arguments    │
                                        └───────┬───────┘
                                                │
                                                v
                                        Execute function
                                                │
                                                v
                                        Format <|result|>
                                                │
                                        Feed back to model
```

The model learns to emit special tokens (`<|tool_call|>`, `<|end|>`) during SFT. The inference engine detects these tokens, pauses generation, executes the tool, injects the result, and resumes generation. It's the same pattern used by GPT-4, Claude, and Gemini — just transparent and simple.

---

## Project Structure

```
pebble/
├── configs/
│   ├── pebble_25m.yaml          # 25M model config
│   └── pebble_100m.yaml         # 100M model config (stretch)
├── src/
│   ├── model/
│   │   ├── attention.py         # GQA + causal masking
│   │   ├── embedding.py         # Token embeddings
│   │   ├── rope.py              # Rotary positional embeddings
│   │   ├── transformer.py       # Transformer block
│   │   └── pebble.py            # Full model assembly
│   ├── tokenizer/
│   │   ├── bpe.py               # BPE from scratch
│   │   └── trainer.py           # Tokenizer training
│   ├── data/
│   │   ├── dataset.py           # Memory-mapped dataset
│   │   ├── preprocessing.py     # Data cleaning pipeline
│   │   └── tool_format.py       # Tool-calling data synthesis
│   ├── training/
│   │   ├── trainer.py           # Training loop
│   │   ├── optimizer.py         # AdamW + LR scheduling
│   │   └── loss.py              # Cross-entropy, DPO loss
│   ├── inference/
│   │   ├── generate.py          # Text generation + sampling
│   │   ├── kv_cache.py          # KV-cache implementation
│   │   └── tool_executor.py     # Tool calling runtime
│   └── evaluation/
│       ├── perplexity.py        # Perplexity measurement
│       └── tool_eval.py         # Tool-calling accuracy
├── scripts/
│   ├── train.py                 # Pretraining entrypoint
│   ├── finetune.py              # SFT + DPO entrypoint
│   ├── generate.py              # Interactive chat
│   └── export.py                # Quantization + GGUF export
├── tests/                       # Unit tests for every module
├── notebooks/                   # Exploration & visualization
├── data/                        # Raw + processed datasets
└── checkpoints/                 # Saved models
```

Every file is a learning resource. Code is commented to explain *why*, not just *what*.

---

## Hardware Requirements

| Setup | What You Can Do | Training Time (25M) |
|-------|----------------|---------------------|
| **NVIDIA GPU, 8+ GB VRAM** | Full pretraining + SFT + DPO | ~24-48 hours |
| **NVIDIA GPU, 4 GB VRAM** | Reduced batch size, gradient checkpointing | ~48-96 hours |
| **CPU only** | Run architecture code, small experiments | Days (not recommended for full training) |
| **Google Colab (free)** | Small experiments, architecture exploration | Limited by session time |

Tested on: NVIDIA RTX PRO 2000 (8 GB), 16 GB RAM, Intel Core Ultra 7 265H.

---

## Papers Behind Pebble

These are the papers whose ideas you'll implement. Reading them is optional but recommended.

| Paper | Year | What You'll Use From It |
|-------|------|------------------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | The transformer architecture |
| [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | 2019 | GPT-2 pretraining approach |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 2021 | RoPE implementation |
| [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685) | 2021 | Parameter-efficient fine-tuning |
| [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) | 2022 | Chinchilla scaling laws |
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | 2023 | GQA, SwiGLU, RMSNorm architecture |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 2023 | DPO alignment |
| [Toolformer: LLMs Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | 2023 | Tool-calling approach |

---

## FAQ

<details>
<summary><b>Is 25M parameters enough to learn anything useful?</b></summary>

Yes. 25M parameters is enough to learn coherent English, basic reasoning, and structured tool calling. It won't write essays or pass the bar exam, but it will demonstrably learn language patterns and reliably call tools when prompted. The point isn't to compete with GPT-4 — it's to understand how GPT-4 works by building something real.
</details>

<details>
<summary><b>How is this different from nanoGPT?</b></summary>

nanoGPT is a minimal GPT-2 reimplementation optimized for speed and simplicity. Pebble is a learning-focused project that goes further:
- Modern architecture (RoPE, GQA, SwiGLU, RMSNorm) vs GPT-2's design
- Full fine-tuning pipeline (SFT + DPO), not just pretraining
- Tool calling capability — the model learns to use external functions
- BPE tokenizer built from scratch
- Quantization and export to GGUF
- Heavily commented code designed to teach
</details>

<details>
<summary><b>How is this different from rasbt/LLMs-from-scratch?</b></summary>

Raschka's excellent book builds a GPT-2 style model for educational purposes. Pebble differs in:
- **Modern architecture**: GQA, RoPE, SwiGLU instead of GPT-2's MHA, learned positional embeddings, GELU
- **Tool calling**: The model learns to call external functions — a capability not covered in the book
- **DPO alignment**: Full preference optimization pipeline
- **Production patterns**: Quantization, GGUF export, KV-cache inference
- **Single-GPU focus**: Everything is designed to run on a consumer GPU (8GB)
</details>

<details>
<summary><b>Can I run this without a GPU?</b></summary>

The architecture code, tokenizer, and data pipeline all work on CPU. Training on CPU is technically possible but will take days instead of hours for the full 25M model. For learning the concepts, you can train a tiny version (1-5M params) on CPU for small experiments.
</details>

<details>
<summary><b>What's the Chinchilla-optimal training budget for 25M params?</b></summary>

The Chinchilla scaling laws suggest ~500M tokens for a 25M parameter model (20x parameter count). We train on ~800M unique tokens over multiple epochs, slightly over-training to ensure the model fully converges. The total tokens seen during training is ~3-6B (with repetition).
</details>

<details>
<summary><b>Can I scale this up to a larger model?</b></summary>

Yes. The `configs/pebble_100m.yaml` config defines a 100M parameter variant that trains on the same hardware with gradient checkpointing. The code is written to be architecture-agnostic — change the config, and you get a bigger model. Beyond 100M, you'll want multiple GPUs or cloud compute.
</details>

---

## Contributing

Contributions are welcome. If you find a bug, have a clearer way to explain a concept, or want to add a new feature:

1. Fork the repo
2. Create a feature branch (`git checkout -b improve-attention-docs`)
3. Commit your changes
4. Open a PR

Please keep code comments educational. The goal is clarity, not cleverness.

---

## Acknowledgments

Pebble stands on the shoulders of giants:

- [Andrej Karpathy](https://github.com/karpathy) — nanoGPT, llm.c, and the Zero to Hero series
- [Sebastian Raschka](https://github.com/rasbt) — Build a Large Language Model (From Scratch)
- [Meta AI](https://github.com/meta-llama) — LLaMA architecture and open weights
- [Hugging Face](https://github.com/huggingface) — SmolLM and the open-source ML ecosystem
- The PyTorch team — for making this all possible

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Use it, learn from it, build on it. That's the point.

---

<p align="center">
  <sub>If this repo helped you understand LLMs better, a star would mean a lot.</sub>
</p>
