#ifndef BACKEND_H
#define BACKEND_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h> // for memcpy

// ==========================================
// Backend Abstraction Layer
// Maps device_X functions to either CUDA or Standard C
// ==========================================

#ifdef __CUDACC__
    // --------------------------------------
    // GPU Backend (CUDA)
    // --------------------------------------
    #include <cuda_runtime.h>
    #include <cublas_v2.h>

    #define DEVICE_TO_HOST cudaMemcpyDeviceToHost
    #define DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
    #define HOST_TO_DEVICE cudaMemcpyHostToDevice

    // Macro for checking CUDA errors
    #define check_status(call)                                                     \
    do {                                                                           \
        cudaError_t err = call;                                                    \
        if (err != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__,         \
                    cudaGetErrorString(err));                                      \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

    // Wrappers
    #define device_malloc(ptr, size) cudaMalloc(ptr, size)
    #define device_free(ptr) cudaFree(ptr)
    #define device_memcpy(dst, src, size, kind) cudaMemcpy(dst, src, size, kind)
    #define device_sync() cudaDeviceSynchronize()

#else
    // --------------------------------------
    // CPU Backend (Standard C)
    // --------------------------------------
    
    // Mock enums for memcpy direction (ignored in CPU memcpy)
    typedef int DeviceMemcpyKind;
    #define DEVICE_TO_HOST 0
    #define DEVICE_TO_DEVICE 1
    #define HOST_TO_DEVICE 2

    // Mock status check (always success)
    #define check_status(stmt) stmt

    // Wrappers
    // cudaMalloc expects (void**, size), malloc returns void*
    static inline int device_malloc(void** ptr, size_t size) {
        *ptr = malloc(size);
        if (*ptr == NULL) return -1;
        return 0;
    }

    static inline void device_free(void* ptr) {
        free(ptr);
    }

    static inline int device_memcpy(void* dst, const void* src, size_t size, DeviceMemcpyKind kind) {
        memcpy(dst, src, size);
        return 0;
    }

    static inline void device_sync() {
        // No-op for CPU
    }

#endif // __CUDACC__

#endif // BACKEND_H

