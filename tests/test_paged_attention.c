#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "structs.h"
#include "ops.h"
#include "backend.h"

// Mock globals required by linked objects
int g_paged_mode = 1;
// KVCacheManager g_kv_manager; // Not needed if we don't use model.c logic that depends on it (we cleaned model.c)
// But model.c still refers to g_paged_mode.

// Mock backend functions for CPU test if needed (already in backend.h for CPU)

void test_kv_cache_allocation() {
    printf("Testing KV Cache Allocation...\n");
    
    KVCacheManager mgr;
    int block_size = 16;
    int num_blocks = 10;
    int n_layers = 2;
    int n_kv_heads = 4;
    int head_dim = 32;
    
    init_kv_cache_manager(&mgr, block_size, num_blocks, n_layers, n_kv_heads, head_dim);
    
    assert(mgr.block_size == block_size);
    assert(mgr.num_blocks == num_blocks);
    assert(mgr.free_blocks_count == num_blocks);
    
    // Test Allocation
    int b1 = alloc_block(&mgr);
    assert(b1 != -1);
    assert(mgr.free_blocks_count == num_blocks - 1);
    
    int b2 = alloc_block(&mgr);
    assert(b2 != -1);
    assert(b2 != b1);
    assert(mgr.free_blocks_count == num_blocks - 2);
    
    // Free block
    free_block(&mgr, b1);
    assert(mgr.free_blocks_count == num_blocks - 1);
    
    // Re-alloc should get the freed block (LIFO stack behavior usually, but implementation dependent)
    // Our implementation uses a stack: free_block pushes to top. alloc pops from top.
    int b3 = alloc_block(&mgr);
    assert(b3 == b1);
    
    free_kv_cache_manager(&mgr);
    printf("KV Cache Allocation Passed.\n");
}

void test_physical_address_mapping() {
    printf("Testing Physical Address Mapping...\n");
    
    KVCacheManager mgr;
    int block_size = 4;
    int num_blocks = 4;
    int n_layers = 1;
    int n_kv_heads = 1;
    int head_dim = 1; 
    
    init_kv_cache_manager(&mgr, block_size, num_blocks, n_layers, n_kv_heads, head_dim);
    
    // Layout: [layer, block, offset, head, dim]
    // Strides:
    //  layer: num_blocks * block_size * n_kv_heads * head_dim = 4 * 4 * 1 * 1 = 16
    //  block: block_size * n_kv_heads * head_dim = 4 * 1 * 1 = 4
    //  offset: n_kv_heads * head_dim = 1 * 1 = 1
    
    int layer = 0;
    int block_idx = 2;
    int offset = 3;
    
    long phys_addr = get_physical_offset(&mgr, layer, block_idx, offset, n_kv_heads, head_dim);
    long expected = (0 * 16) + (2 * 4) + (3 * 1); // 11
    
    assert(phys_addr == expected);
    
    free_kv_cache_manager(&mgr);
    printf("Physical Address Mapping Passed.\n");
}

int main() {
    init_ops(); // Init global if needed
    
    test_kv_cache_allocation();
    test_physical_address_mapping();
    
    printf("All Paged Attention tests passed!\n");
    return 0;
}

