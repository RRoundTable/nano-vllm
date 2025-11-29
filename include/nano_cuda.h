#ifndef NANO_CUDA_H
#define NANO_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define cudaCheck(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

#endif // NANO_CUDA_H

