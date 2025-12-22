#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "scheduler.h"
#include "log.h"

void init_batch_data(BatchData* batch, int max_batch_capacity, int max_sequences) {
    batch->tokens = (int*)malloc(max_batch_capacity * sizeof(int));
    batch->pos = (int*)malloc(max_batch_capacity * sizeof(int));
    batch->seq_ids = (int*)malloc(max_batch_capacity * sizeof(int));
    batch->output_indices = (int*)malloc(max_sequences * sizeof(int));
    batch->block_tables = (BlockTable**)malloc(max_sequences * sizeof(BlockTable*));
    
    batch->batch_count = 0;
    batch->output_count = 0;
}

void batch_reset(BatchData* batch) {
    batch->batch_count = 0;
    batch->output_count = 0;
}

int scheduler_step(Sequence* seqs, int num_seqs, int global_step, 
                   int max_batch_capacity, int chunk_size, 
                   KVCacheManager* kv_mgr, BatchData* batch) {
    
    int all_finished = 1;
    
    for (int i = 0; i < num_seqs; i++) {
        // Set block table pointer for this sequence (ensure it's linked)
        batch->block_tables[i] = &seqs[i].table;

        // Check Arrival
        if (global_step < seqs[i].arrival_step) {
            all_finished = 0;
            continue;
        }
        
        // Activate
        if (seqs[i].status == SEQ_WAITING) {
            seqs[i].active = 1;
            seqs[i].status = SEQ_PREFILLING;
            log_printf(">>> [Scheduler] Seq %d ARRIVED!\n", i);
        }
        
        if (seqs[i].status == SEQ_FINISHED) {
            continue;
        }
        
        all_finished = 0; // At least one running
        
        // Decide how many tokens to add
        int is_prefill = (seqs[i].pos < seqs[i].num_prompt_tokens - 1);
        int n_tokens = 0;
        int start_pos = seqs[i].pos;
        int* token_ptr = NULL;
        
        if (is_prefill) {
            seqs[i].status = SEQ_PREFILLING;
            
            int end_pos = seqs[i].num_prompt_tokens;
            int remaining = end_pos - start_pos;
            n_tokens = (remaining > chunk_size) ? chunk_size : remaining;
            token_ptr = seqs[i].prompt_tokens + start_pos;
            
        } else {
            seqs[i].status = SEQ_DECODING;
            n_tokens = 1;
            token_ptr = &seqs[i].current_token;
        }
        
        // Check Capacity
        if (batch->batch_count + n_tokens > max_batch_capacity) {
            log_printf("WARNING: Batch capacity reached! Skipping Seq %d this step.\n", i);
            continue; // Wait for next step
        }
        
        // Alloc Blocks if needed (for Paged Attention)
        if (kv_mgr) {
            int block_size = kv_mgr->block_size;
            for (int t = 0; t < n_tokens; t++) {
                int current_pos = start_pos + t;
                if (current_pos % block_size == 0) {
                    int new_block = alloc_block(kv_mgr);
                    if (new_block == -1) {
                        printf("Error: Out of KV Cache blocks!\n");
                        exit(1);
                    }
                    int logical_idx = current_pos / block_size;
                    seqs[i].table.block_indices[logical_idx] = new_block;
                    seqs[i].table.num_blocks++;
                }
            }
        }
        
        // Add to Batch
        for (int t = 0; t < n_tokens; t++) {
            batch->tokens[batch->batch_count + t] = token_ptr[t];
            batch->pos[batch->batch_count + t] = start_pos + t;
            batch->seq_ids[batch->batch_count + t] = i;
        }
        
        batch->output_indices[batch->output_count] = batch->batch_count + n_tokens - 1;
        batch->output_count++;
        batch->batch_count += n_tokens;
        
        seqs[i].pos += n_tokens;
        
        if (is_prefill) {
                seqs[i].current_token = token_ptr[n_tokens - 1]; // Last token of chunk
        }
    }
    
    // Return 0 if all finished, 1 if running
    return !all_finished;
}

