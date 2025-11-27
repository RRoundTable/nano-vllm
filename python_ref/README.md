# Phase 0: Python Reference and Binary Export

This directory contains Python scripts for loading pretrained Llama models from Hugging Face and converting them to raw binary format for C loading.

## Files

- `model.py`: Llama architecture reference and documentation
- `export_binary.py`: Converts Hugging Face model to binary format
- `requirements.txt`: Python dependencies

## Setup

Install dependencies using uv (recommended):

```bash
cd python_ref
uv sync
```

Or using Poetry:

```bash
cd python_ref
poetry install
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Export a model to binary

```bash
# Export TinyLlama (1.1B parameters, good for testing)
python export_binary.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output ../data/tinyllama.bin

# Export Llama-2-7B (requires ~13GB disk space)
python export_binary.py --model meta-llama/Llama-2-7b-hf --output ../data/llama2-7b.bin
```

### Binary Format

The exported `.bin` file contains:

**Header (36 bytes):**
- `dim` (int32): Model dimension
- `n_layers` (int32): Number of layers
- `n_heads` (int32): Number of attention heads
- `n_kv_heads` (int32): Number of KV heads (for GQA)
- `vocab_size` (int32): Vocabulary size
- `max_seq_len` (int32): Maximum sequence length
- `hidden_dim` (int32): FFN hidden dimension
- `head_dim` (int32): Attention head dimension (dim / n_heads)
- `rope_theta` (float32): RoPE base frequency

**Weights (all FP16, row-major):**
- Token embedding table
- For each layer:
  - Attention RMSNorm weights
  - Q, K, V, O projection matrices
  - FFN RMSNorm weights
  - Gate, Up, Down projection matrices
- Final RMSNorm weights
- LM head weights

All weights are stored as contiguous FP16 (float16) arrays in row-major order, ready to be loaded with a single `fread()` call in C.

## Recommended Models for Testing

1. **TinyLlama-1.1B** (Recommended for development):
   - Small and fast (~2GB)
   - Good for testing infrastructure
   - Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

2. **Llama-2-7B** (For benchmarking):
   - Standard size (~13GB)
   - Good for performance comparison with vLLM
   - Model: `meta-llama/Llama-2-7b-hf`
   - Note: Requires Hugging Face authentication

## Notes

- All models are converted to FP16 format
- HEAD_DIM must be 128 (enforced)
- Weights are transposed during export for row-major C layout
- No bias terms (Llama architecture doesn't use them)

