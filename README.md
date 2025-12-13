# nano-vllm

A minimalist, pure C implementation of LLM inference, designed to demystify modern serving systems like vLLM. This project implements the core components of Llama inference (Attention, RoPE, RMSNorm, MatMul) from scratch without complex C++ templates or external dependencies (except CUDA for GPU support).

**Philosophy:** "No C++, No STL, Just Raw C & CUDA"

## Features

- **Dual Backend Architecture**:
  - **CPU Backend**: Pure C implementation compatible with any system (Mac/Linux). Includes a terminal-based visualizer for Attention scores and KV Cache memory usage.
  - **GPU Backend**: CUDA implementation for NVIDIA GPUs with Paged Attention support.
- **Educational Focus**:
  - ASCII Art Visualization of internal model states.
  - Simple, readable implementations of complex kernels.
  - Direct memory management (malloc/cudaMalloc) to understand data flow.
- **HuggingFace Compatible**: Supports loading any Llama-architecture model from HuggingFace.

## Getting Started

### Prerequisites

- **C Compiler**: `gcc` or `clang`
- **Python 3**: For model download and conversion tools
- **NVIDIA Toolkit** (Optional): `nvcc` for GPU backend

### 1. Download & Prepare Model

This project uses HuggingFace models exported to a custom binary format (FP16).

```bash
# Easy setup script to download TinyStories (15M or 42M)
./setup_models.sh

# Or manually:
# Export a model from HuggingFace (example: TinyLlama)
cd python_ref
python export_binary.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output data/model.bin

# Download tokenizer (llama2.c format)
curl -L https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin -o ../data/tokenizer.bin
```

### 2. Build

Use `make` to compile the project.

```bash
# Build for CPU (Works on Mac/Linux)
make cpu

# Build for GPU (Requires NVIDIA GPU & NVCC)
make gpu
```

### 3. Run Inference

Run the inference engine with the model path. You can enable Paged Attention mode with `--paged`.

**CPU Inference:**
```bash
# Syntax: ./nano_vllm_cpu <model_path> [options]
./nano_vllm_cpu python_ref/data/model.bin -n 100 -i "Once upon a time" --paged
```

**GPU Inference:**
```bash
# Syntax: ./nano_vllm_gpu <model_path> [options]
./nano_vllm_gpu python_ref/data/model.bin -n 100 -i "Once upon a time" --paged
```

You will see the model generating text token-by-token, along with a visualization of the Attention mechanism!

```text
[Visualizer] Attention Patterns (Step 49)
Head:   H00     H01     H02     H03   
      ------- ------- ------- ------- 
t048: [..-] [.....] [.....] [.....] 
t049: [--*] [.....] [.....] [..-] 

[Visualizer] Memory Breakdown (Mode: Paged)
Reqs: [Req0:4 Blks] [Req1:5 Blks] ...

Memory Map (Total Pool: 64 blocks = 1024 slots):
[#####################........................._______]
1. Used (Actual Data)        : 450 (43.9%)
2. Internal Frag (Last Page) : 58 (5.7%)
3. Free (Available Blocks)   : 516 (50.4%)
```

## Project Structure

```
nano-vllm/
├── src/                # Core C source (main loop, memory management)
├── kernels/            # Kernel implementations
│   ├── cpu/            # Pure C kernels & Visualizer
│   └── gpu/            # CUDA kernels (Attention, Layers)
├── include/            # Header files
├── python_ref/         # Python reference implementation & export tools
├── data/               # Model binaries & tokenizer
└── Makefile            # Build configuration
```

## Roadmap

- [x] **Phase 1**: Naive CPU Inference & Visualization
- [x] **Phase 2**: CUDA Kernels (Naive)
- [x] **Phase 3**: PagedAttention (KV Cache Paging) - *Implemented for CPU & GPU*
- [ ] **Phase 4**: Continuous Batching
- [ ] **Phase 5**: Chunked Prefill

## License

MIT
