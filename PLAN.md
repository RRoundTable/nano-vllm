# Project Refactoring Plan: CPU-Only Transition

## 1. Goal
Simplify the `nano-vllm` project by removing all GPU/CUDA support and focusing exclusively on the CPU implementation. This involves cleaning up the codebase, restructuring directories, and updating documentation.

## 2. Directory Structure Changes
- **Remove** `kernels/gpu/` directory.
- **Remove** `tests/test_matmul.cu`.
- **Relocate** CPU kernels:
  - Move `kernels/cpu/*.c` to `src/kernels/`.
  - Delete `kernels/` directory.

New Structure:
```
nano-vllm/
├── src/
│   ├── kernels/        # CPU Kernels (attention.c, layers.c)
│   ├── main.c
│   ├── model.c
│   ├── memory.c
│   ├── ...
├── include/            # Headers (backend.h updated)
├── docs/               # Updated documentation
├── python_ref/         # Python tools (unchanged)
├── Makefile            # Updated build config
└── README.md           # Updated project info
```

## 3. Code Modifications

### `include/backend.h`
- Remove `#ifdef NANO_CUDA` blocks.
- Permanently map `device_*` functions to standard C `malloc`/`free`/`memcpy`.

### `src/main.c` & `src/model.c`
- Remove conditional compilation for CUDA (`#ifdef NANO_CUDA`).
- Simplify memory management logic where backend-specific branches existed.

### `Makefile`
- Remove `TARGET_GPU` and `NVCC` configurations.
- Update source paths to include `src/kernels/*.c`.
- Set default target to CPU build.

## 4. Documentation
- **README.md**: Remove GPU setup/usage instructions. Focus on CPU features and visualization.
- **docs/**: Consolidate CPU-specific docs if necessary (or just leave them as they describe the core logic).

## 5. Execution Steps
1. Delete GPU files.
2. Move CPU kernels to `src/kernels/`.
3. Refactor `backend.h` and source files.
4. Update `Makefile`.
5. Update `README.md`.
