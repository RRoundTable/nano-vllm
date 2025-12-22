#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "scheduler.h"
#include "structs.h"

int g_paged_mode = 1;

void test_continuous_batching_scheduler() {
    printf("Testing Continuous Batching Scheduler...\n");

    // KV Manager
    KVCacheManager kv_mgr;
    init_kv_cache_manager(&kv_mgr, 4, 100, 1, 1, 1);
    
    // Batch Data
    BatchData batch;
    init_batch_data(&batch, 100, 2); // Cap 100, 2 seqs

    // Seq 0: Long Prompt (10 tokens), Arrives 0. Chunk 4.
    // Seq 1: Short Prompt (3 tokens), Arrives 2.
    
    Sequence seqs[2];
    
    // Seq 0
    seqs[0].id = 0;
    seqs[0].status = SEQ_WAITING;
    seqs[0].arrival_step = 0;
    seqs[0].pos = 0;
    seqs[0].num_prompt_tokens = 10;
    seqs[0].prompt_tokens = (int*)malloc(10 * sizeof(int));
    for(int i=0; i<10; i++) seqs[0].prompt_tokens[i] = 100+i;
    seqs[0].table.block_indices = (int*)malloc(20 * sizeof(int));
    seqs[0].table.num_blocks = 0;
    seqs[0].active = 0;
    seqs[0].output_history = (int*)malloc(100 * sizeof(int));
    
    // Seq 1
    seqs[1].id = 1;
    seqs[1].status = SEQ_WAITING;
    seqs[1].arrival_step = 2;
    seqs[1].pos = 0;
    seqs[1].num_prompt_tokens = 3;
    seqs[1].prompt_tokens = (int*)malloc(3 * sizeof(int));
    for(int i=0; i<3; i++) seqs[1].prompt_tokens[i] = 200+i;
    seqs[1].table.block_indices = (int*)malloc(20 * sizeof(int));
    seqs[1].table.num_blocks = 0;
    seqs[1].active = 0;
    seqs[1].output_history = (int*)malloc(100 * sizeof(int));

    int chunk_size = 4;
    int global_step = 0;

    // --- Step 0 ---
    // Seq 0 Arrives. Prefill 0..3 (4 tokens)
    // Seq 1 Waiting.
    printf("Step 0...\n");
    scheduler_step(seqs, 2, global_step, 100, chunk_size, &kv_mgr, &batch);
    
    assert(batch.batch_count == 4);
    assert(batch.seq_ids[0] == 0); // All Seq 0
    assert(seqs[0].status == SEQ_PREFILLING);
    assert(seqs[0].pos == 4);
    
    batch_reset(&batch);
    global_step++;

    // --- Step 1 ---
    // Seq 0 Prefill 4..7 (4 tokens)
    // Seq 1 Waiting.
    printf("Step 1...\n");
    scheduler_step(seqs, 2, global_step, 100, chunk_size, &kv_mgr, &batch);
    
    assert(batch.batch_count == 4);
    assert(seqs[0].pos == 8);
    
    batch_reset(&batch);
    global_step++;
    
    // --- Step 2 ---
    // Seq 0 Prefill 8..9 (2 tokens) -> Finish Prefill next
    // Seq 1 Arrives. Prefill 0..2 (3 tokens) -> Finish Prefill next
    // Batch should contain BOTH.
    printf("Step 2...\n");
    scheduler_step(seqs, 2, global_step, 100, chunk_size, &kv_mgr, &batch);
    
    assert(seqs[0].pos == 10); // Finished prefill
    assert(seqs[1].status == SEQ_PREFILLING);
    assert(seqs[1].pos == 3); // Finished prefill
    
    // Verify Batch Content
    // Seq 0 adds 2 tokens (108, 109)
    // Seq 1 adds 3 tokens (200, 201, 202)
    // Total 5
    assert(batch.batch_count == 5);
    
    // Check Seq IDs in batch
    // Implementation order: iterate seqs 0 then 1.
    // So 0's tokens then 1's tokens.
    assert(batch.seq_ids[0] == 0);
    assert(batch.seq_ids[1] == 0);
    assert(batch.seq_ids[2] == 1);
    assert(batch.seq_ids[3] == 1);
    assert(batch.seq_ids[4] == 1);
    
    batch_reset(&batch);
    global_step++;
    
    // --- Step 3 ---
    // Both Decoding.
    // Seq 0: Decode 1 token (input=109)
    // Seq 1: Decode 1 token (input=202)
    printf("Step 3...\n");
    scheduler_step(seqs, 2, global_step, 100, chunk_size, &kv_mgr, &batch);
    
    assert(seqs[0].status == SEQ_DECODING);
    assert(seqs[1].status == SEQ_DECODING);
    assert(batch.batch_count == 2);
    
    assert(batch.seq_ids[0] == 0);
    assert(batch.tokens[0] == 109);
    
    assert(batch.seq_ids[1] == 1);
    assert(batch.tokens[1] == 202);
    
    printf("Continuous Batching Scheduler Passed.\n");
    
    // Cleanup
    free_kv_cache_manager(&kv_mgr);
    // ... frees ...
}

int main() {
    init_ops();
    test_continuous_batching_scheduler();
    return 0;
}

