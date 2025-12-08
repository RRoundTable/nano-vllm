#ifndef NANO_CUDA_H
#define NANO_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaCheck(call)                                                         \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#endif


