#ifndef NANO_CUDA_H
#define NANO_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // Added for memcpy

// If compiling for CPU, define dummy macros
#ifndef __CUDACC__
    
    // Dummy cudaError_t enum
    typedef int cudaError_t;
    #define cudaSuccess 0
    
    // Dummy functions
    static inline const char* cudaGetErrorString(cudaError_t error) { return "No Error"; }
    
    // Mock cudaMalloc with malloc
    static inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
        *devPtr = malloc(size);
        return (*devPtr == NULL) ? 1 : cudaSuccess;
    }
    
    // Mock cudaFree with free
    static inline cudaError_t cudaFree(void* devPtr) {
        free(devPtr);
        return cudaSuccess;
    }
    
    // Mock cudaMemcpy with memcpy
    #define cudaMemcpyHostToDevice 1
    #define cudaMemcpyDeviceToHost 2
    #define cudaMemcpyDeviceToDevice 3
    
    static inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
        memcpy(dst, src, count);
        return cudaSuccess;
    }

#endif

// Common Macro
#define cudaCheck(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

#endif // NANO_CUDA_H
