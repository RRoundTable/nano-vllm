#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "structs.h"
#include "ops.h"

typedef struct {
    int* tokens;          // [MAX_BATCH_CAPACITY]
    int* pos;             // [MAX_BATCH_CAPACITY]
    int* seq_ids;         // [MAX_BATCH_CAPACITY]
    int* output_indices;  // [BATCH_SIZE]
    BlockTable** block_tables; // [BATCH_SIZE]
    
    int batch_count;
    int output_count;
} BatchData;

// Initialize BatchData with pre-allocated buffers
void init_batch_data(BatchData* batch, int max_batch_capacity, int max_sequences);

// Reset counts for new step
void batch_reset(BatchData* batch);

// Scheduler Step: Selects tokens to process from sequences
// Returns: number of active sequences processed (0 if all waiting/finished)
int scheduler_step(Sequence* seqs, int num_seqs, int global_step, 
                   int max_batch_capacity, int chunk_size, 
                   KVCacheManager* kv_mgr, BatchData* batch);

#endif // SCHEDULER_H

