#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "scheduler.h"
#include "structs.h"

int g_paged_mode = 1;

// Mock logging to avoid clutter
// #define log_printf printf

void test_chunked_prefill_logic() {
    printf("Testing Chunked Prefill Logic...\n");
    
    // Setup Dummy Sequence
    Sequence seq;
    seq.id = 0;
    seq.active = 0;
    seq.status = SEQ_WAITING;
    seq.arrival_step = 0;
    seq.pos = 0;
    seq.num_prompt_tokens = 20; // Length 20
    seq.prompt_tokens = (int*)malloc(20 * sizeof(int));
    for(int i=0; i<20; i++) seq.prompt_tokens[i] = i;
    seq.output_history = (int*)malloc(100 * sizeof(int)); // Dummy
    seq.state = NULL;
    
    // KV Manager (Dummy)
    KVCacheManager kv_mgr;
    kv_mgr.block_size = 4;
    // We won't actually alloc memory, just check logic calling alloc_block
    // But alloc_block needs initialized mgr.
    // Let's rely on valid mgr but minimal size
    init_kv_cache_manager(&kv_mgr, 4, 100, 1, 1, 1);
    
    // Batch Data
    BatchData batch;
    init_batch_data(&batch, 100, 1);
    
    // Initialize Seq Table
    seq.table.block_indices = (int*)malloc(100 * sizeof(int));
    seq.table.num_blocks = 0;
    
    int chunk_size = 8;
    int global_step = 0;
    
    // Step 1: 0..7 (8 tokens)
    scheduler_step(&seq, 1, global_step, 100, chunk_size, &kv_mgr, &batch);
    assert(seq.status == SEQ_PREFILLING);
    assert(seq.pos == 8);
    assert(batch.batch_count == 8);
    // Verify tokens in batch
    assert(batch.tokens[0] == 0);
    assert(batch.tokens[7] == 7);
    assert(seq.current_token == 7);
    
    // Reset batch
    batch_reset(&batch);
    global_step++;
    
    // Step 2: 8..15 (8 tokens)
    scheduler_step(&seq, 1, global_step, 100, chunk_size, &kv_mgr, &batch);
    assert(seq.status == SEQ_PREFILLING);
    assert(seq.pos == 16);
    assert(batch.batch_count == 8);
    assert(batch.tokens[0] == 8);
    assert(seq.current_token == 15);
    
    // Reset batch
    batch_reset(&batch);
    global_step++;
    
    // Step 3: 16..19 (4 tokens) - Finish Prefill?
    // Prefill logic: (pos < num_prompt_tokens - 1)
    // Here pos=16, end=20. remaining=4.
    // Next pos will be 20.
    // Is it prefill or decode next?
    // If pos reached 20 (== num_prompt_tokens), next is decode.
    
    scheduler_step(&seq, 1, global_step, 100, chunk_size, &kv_mgr, &batch);
    assert(seq.status == SEQ_PREFILLING); // Still labeled prefill during this step
    assert(seq.pos == 20);
    assert(batch.batch_count == 4);
    assert(batch.tokens[0] == 16);
    assert(batch.tokens[3] == 19);
    assert(seq.current_token == 19);
    
    // Reset batch
    batch_reset(&batch);
    global_step++;
    
    // Step 4: Decode
    // pos=20, num_prompt=20.
    // is_prefill = (20 < 19) -> False.
    scheduler_step(&seq, 1, global_step, 100, chunk_size, &kv_mgr, &batch);
    assert(seq.status == SEQ_DECODING);
    assert(seq.pos == 21); // Generated 1 token
    assert(batch.batch_count == 1);
    // Decode uses current_token (19) as input
    assert(batch.tokens[0] == 19);
    
    printf("Chunked Prefill Logic Passed.\n");
    
    // Cleanup
    free_kv_cache_manager(&kv_mgr);
    free(seq.prompt_tokens);
    free(seq.output_history);
    free(seq.table.block_indices);
    free(batch.tokens);
    free(batch.pos);
    free(batch.seq_ids);
    free(batch.output_indices);
    free(batch.block_tables);
}

int main() {
    init_ops();
    test_chunked_prefill_logic();
    return 0;
}

