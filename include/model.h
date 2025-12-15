#ifndef MODEL_H
#define MODEL_H

#include "structs.h"

// Model Loading & Memory Management
void load_model(Weights* w, Config* p, const char* checkpoint_path);
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);

// Inference (Single Sequence - Deprecated/Reference)
void transformer(int* tokens, int num_tokens, int pos, Config* p, RunState* s, Weights* w, BlockTable* bt);

// Inference (Batched)
void transformer_batch(int* tokens, int num_tokens, int* pos_arr, int* seq_ids, 
                       int* output_indices, int num_outputs,
                       Config* p, RunState* s, Weights* w, BlockTable** block_tables);

#endif // MODEL_H

