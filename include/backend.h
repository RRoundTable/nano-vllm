#ifndef BACKEND_H
#define BACKEND_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h> // for memcpy

// ==========================================
// Backend Abstraction Layer
// CPU-Only Implementation
// ==========================================

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

#endif // BACKEND_H
