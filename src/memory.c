#include <stdlib.h>
#include <stdio.h>
#include "structs.h"
#include "ops.h"
#include "backend.h"

// Initialize the KV Cache Manager
void init_kv_cache_manager(KVCacheManager* mgr, int block_size, int num_blocks, int n_layers, int n_kv_heads, int head_dim) {
    mgr->block_size = block_size;
    mgr->num_blocks = num_blocks;
    mgr->free_blocks_count = num_blocks;
    
    // Allocate free block stack (Host Memory)
    mgr->free_block_indices = (int*)malloc(num_blocks * sizeof(int));
    for (int i = 0; i < num_blocks; i++) {
        mgr->free_block_indices[i] = num_blocks - 1 - i; // Reverse order (stack)
    }
    
    // Allocate Physical Memory Pool (Device Memory)
    /* 
     * Visualizing KV Cache Memory Pool Structure (5D Tensor):
     * 
     * pool_k / pool_v
     * ├── Layer 0
     * │   ├── Block 0
     * │   │   ├── Token 0
     * │   │   │   ├── Head 0: [float, float, ... (head_dim)]
     * │   │   │   ├── Head 1: [float, float, ... (head_dim)]
     * │   │   │   └── ...
     * │   │   ├── Token 1
     * │   │   └── ... (up to block_size)
     * │   ├── Block 1
     * │   └── ... (up to num_blocks)
     * ├── Layer 1
     * └── ... (up to n_layers)
     *
     * Total Size = n_layers * num_blocks * block_size * n_kv_heads * head_dim
     */
    // Size: [n_layers, num_blocks, block_size, n_kv_heads, head_dim]
    long total_elements = (long)n_layers * num_blocks * block_size * n_kv_heads * head_dim;
    
    check_status(device_malloc((void**)&mgr->pool_k, total_elements * sizeof(float)));
    check_status(device_malloc((void**)&mgr->pool_v, total_elements * sizeof(float)));
}

// Free the manager resources
void free_kv_cache_manager(KVCacheManager* mgr) {
    if (mgr->free_block_indices) free(mgr->free_block_indices);
    if (mgr->pool_k) check_status(device_free(mgr->pool_k));
    if (mgr->pool_v) check_status(device_free(mgr->pool_v));
}

// Allocate a new block
int alloc_block(KVCacheManager* mgr) {
    if (mgr->free_blocks_count <= 0) {
        return -1; // Out of memory
    }
    int block_idx = mgr->free_block_indices[mgr->free_blocks_count - 1];
    mgr->free_blocks_count--;
    return block_idx;
}

// Free a block (return to pool)
void free_block(KVCacheManager* mgr, int block_idx) {
    if (mgr->free_blocks_count >= mgr->num_blocks) {
        return; // Already full? Should not happen.
    }
    mgr->free_block_indices[mgr->free_blocks_count] = block_idx;
    mgr->free_blocks_count++;
}

// Helper: Get physical offset in the pool
// Layout: [layer, block_idx, block_offset, head, head_dim]
long get_physical_offset(KVCacheManager* mgr, int layer, int block_idx, int block_offset, int n_kv_heads, int head_dim) {
    long layer_stride = (long)mgr->num_blocks * mgr->block_size * n_kv_heads * head_dim;
    long block_stride = (long)mgr->block_size * n_kv_heads * head_dim;
    long offset_stride = (long)n_kv_heads * head_dim;
    
    return (long)layer * layer_stride + 
           (long)block_idx * block_stride + 
           (long)block_offset * offset_stride;
}

