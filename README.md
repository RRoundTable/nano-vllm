# nano-vllm

A minimalist, pure C implementation of LLM inference, designed to demystify modern serving systems like vLLM. This project implements the core components of Llama inference (Attention, RoPE, RMSNorm, MatMul) from scratch without complex C++ templates or external dependencies (except CUDA for GPU support).

**Philosophy:** "No C++, No STL, Just Raw C & CUDA"

## Features

- **Dual Backend Architecture**:
  - **CPU Backend**: Pure C implementation compatible with any system (Mac/Linux). Includes a terminal-based visualizer for Attention scores and KV Cache memory usage.
  - **GPU Backend**: CUDA implementation for NVIDIA GPUs (Planned/Partial).
- **Educational Focus**:
  - ASCII Art Visualization of internal model states.
  - Simple, readable implementations of complex kernels.
  - Direct memory management (malloc/cudaMalloc) to understand data flow.
- **Llama-2 Compatible**: Supports loading `llama2.c` models (with conversion).

## Getting Started

### Prerequisites

- **C Compiler**: `gcc` or `clang`
- **Python 3**: For model download and conversion tools
- **NVIDIA Toolkit** (Optional): `nvcc` for GPU backend

### 1. Download & Prepare Model

This project uses the `stories15M` model from Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) as a lightweight test case.

```bash
# 1. Download model and tokenizer
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -O data/stories15M.bin
curl -L https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin -o data/tokenizer.bin

# 2. Convert model format
# (Our engine uses a slightly different binary layout for optimization)
python3 tools/convert_llama2c.py data/stories15M.bin data/stories15M_converted.bin
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

Run the inference engine with the model path and the number of tokens to generate.

```bash
# Syntax: ./nano_vllm_cpu <model_path> <steps>
./nano_vllm_cpu data/stories15M_converted.bin 50
```

You will see the model generating text token-by-token, along with a visualization of the Attention mechanism!

```text
[Visualizer] Attention Patterns (Step 49)
Head:   H00     H01     H02     H03   
      ------- ------- ------- ------- 
t048: [..-] [.....] [.....] [.....] 
t049: [--*] [.....] [.....] [..-] 

[Visualizer] KV Cache Usage: 50/256 tokens (19.5%)
Mem: [#####.............................................]
Once upon a time...
```

## Project Structure

```
nano-vllm/
├── src/                # Core C source (main loop, memory management)
├── kernels/            # Kernel implementations
│   ├── cpu/            # Pure C kernels & Visualizer
│   └── gpu/            # CUDA kernels (Attention, Layers)
├── include/            # Header files
├── tools/              # Python/C utilities (conversion, verification)
├── data/               # Model binaries
└── Makefile            # Build configuration
```

## Roadmap

- [x] **Phase 1**: Naive CPU Inference & Visualization
- [ ] **Phase 2**: CUDA Kernels (Naive)
- [ ] **Phase 3**: Tiled Attention Optimization
- [ ] **Phase 4**: PagedAttention (KV Cache Paging)
- [ ] **Phase 5**: Continuous Batching

## License

MIT
