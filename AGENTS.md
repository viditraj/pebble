# Pebble - Project Commands & Info

## Project
Small Language Model (~25M params) built from scratch with tool-calling capability.

## Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Commands
```bash
# Training
python scripts/train.py --config configs/pebble_25m.yaml

# Evaluation
python scripts/evaluate.py --checkpoint checkpoints/latest.pt

# Interactive generation
python scripts/generate.py --checkpoint checkpoints/latest.pt

# Run tests
pytest tests/
```

## Hardware
- GPU: NVIDIA RTX PRO 2000 (8 GB VRAM)
- RAM: 16 GB
- CPU: Intel Core Ultra 7 265H (16 cores)
- CUDA: 13.2
