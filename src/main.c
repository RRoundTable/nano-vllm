#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include "structs.h"
#include "ops.h"
#include "backend.h"
#include "log.h"
#include <limits.h>

// Forward declarations
void load_model(Weights* w, Config* p, const char* checkpoint_path);
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);
void visualize_final_timeline(char* history, int num_seqs, int total_steps, int max_steps_capacity);

typedef struct {
    char *str;
    int id;
} TokenIndex;

// Tokenizer (Minimal)
typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
const char* decode_token(Tokenizer* t, int token);
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);


// Sampler
typedef struct {
    unsigned long long state;
    float temperature;
    float topp;
    ProbIndex* probindex; // buffer used in top-p sampling
    int vocab_size;
} Sampler;

void build_sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long seed);
int sample(Sampler* s, float* logits);

// Global config for visualization
int g_visualize_paged = 0;
int g_paged_mode = 0; // Actual Logic Mode

// PagedAttention Globals
KVCacheManager g_kv_manager;

// Refactored Transformer to support Batched Processing
void transformer(int* tokens, int num_tokens, int pos, Config* p, RunState* s, Weights* w, BlockTable* bt) {
    // Deprecated for continuous batching, but kept for reference or single-seq fallback
    // ... (Original logic omitted for brevity if not used) ...
    printf("Error: Old transformer called in Continuous Batching mode.\n");
    exit(1);
}

// Batched Transformer
void transformer_batch(int* tokens, int num_tokens, int* pos_arr, int* seq_ids, 
                       int* output_indices, int num_outputs,
                       Config* p, RunState* s, Weights* w, BlockTable** block_tables) {
    
    // 1. Embedding
    for (int t = 0; t < num_tokens; t++) {
        float* content_row = w->token_embedding_table + tokens[t] * p->dim;
        check_status(device_memcpy(s->x + t * p->dim, content_row, p->dim * sizeof(float), DEVICE_TO_DEVICE));
    }
    
    // 2. Layers
    for(int i = 0; i < p->n_layers; i++) {
        LayerWeights* l = &w->layers[i];
        
        rms_norm(s->xb, s->x, l->rms_att_weight, p->dim, num_tokens, 1e-5f);
        
        matmul(s->q, s->xb, l->wq, num_tokens, p->dim, p->n_heads * p->head_dim);
        matmul(s->k, s->xb, l->wk, num_tokens, p->dim, p->n_kv_heads * p->head_dim);
        matmul(s->v, s->xb, l->wv, num_tokens, p->dim, p->n_kv_heads * p->head_dim);
        
        // Batched RoPE with pos_arr
        apply_rope_batch(s->q, s->k, pos_arr, p->rope_theta, p->head_dim, num_tokens, p->n_heads, p->n_kv_heads);
        
        // KV Update Paged Batch
        if (g_paged_mode) {
             update_kv_cache_paged_batch(&g_kv_manager, block_tables, seq_ids, s->k, s->v, 
                                  i, pos_arr, p->n_kv_heads, p->head_dim, num_tokens);
        } else {
             printf("Error: Continuous batching requires paged mode.\n");
             exit(1);
        }
        
        // Attention Batch
        if (g_paged_mode) {
            paged_attention_batch(s->xb2, s->q, &g_kv_manager, block_tables, seq_ids, s->att,
                            i, pos_arr, p->max_seq_len, p->n_heads, p->n_kv_heads, p->head_dim, num_tokens);
        }
        
        matmul(s->xb, s->xb2, l->wo, num_tokens, p->n_heads * p->head_dim, p->dim);
        accum(s->x, s->xb, num_tokens * p->dim);
        
        rms_norm(s->xb, s->x, l->rms_ffn_weight, p->dim, num_tokens, 1e-5f);
        matmul(s->hb, s->xb, l->w_gate, num_tokens, p->dim, p->hidden_dim);
        matmul(s->hb2, s->xb, l->w_up, num_tokens, p->dim, p->hidden_dim);
        swiglu(s->hb, s->hb, s->hb2, p->hidden_dim, num_tokens);
        matmul(s->xb, s->hb, l->w_down, num_tokens, p->hidden_dim, p->dim);
        accum(s->x, s->xb, num_tokens * p->dim);
    }
    
    rms_norm(s->x, s->x, w->rms_final_weight, p->dim, num_tokens, 1e-5f);

    // Compute Logits ONLY for output indices
    // We reuse s->logits buffer. It is size [max_batch_size * vocab_size].
    // We map output index k (0..num_outputs-1) to the location in s->logits.
    // The source embedding is at s->x + output_indices[k] * dim.
    
    // We can do a batched matmul if we gather the embeddings?
    // Or just loop. Since num_outputs is usually small (BATCH_SIZE), loop is fine.
    // OR we can use the fact that `matmul` supports batching.
    // If output_indices are contiguous at the end (often true for decode), we could optimize.
    // But for general ragged batch, they might be scattered.
    // Let's gather embeddings into s->xb (reuse buffer) or s->xb2?
    // s->xb is [max_batch * dim]. Safe to use for first num_outputs.
    
    for(int k=0; k<num_outputs; k++) {
        int token_idx = output_indices[k];
        check_status(device_memcpy(s->xb + k * p->dim, s->x + token_idx * p->dim, p->dim * sizeof(float), DEVICE_TO_DEVICE));
    }
    
    // Matmul: [num_outputs, dim] @ [vocab, dim]^T -> [num_outputs, vocab]
    // s->logits will store the result.
    matmul(s->logits, s->xb, w->lm_head, num_outputs, p->dim, p->vocab_size);
}

void print_usage(char *prog_name) {
    printf("Usage: %s <model_path> [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -t <temp>    Temperature (default: 1.0)\n");
    printf("  -p <topp>    Top-p value (default: 0.9)\n");
    printf("  -n <steps>   Number of generation steps (default: 64)\n");
    printf("  -i <prompt>  Input prompt (default: \"Hello, my name is\")\n");
    printf("  --paged      Enable paged attention mode (Required for Continuous Batching)\n");
    printf("  --chunk-size <N>  Chunk size for prefill (default: 10 if not set)\n");
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    int steps = 1; // Default reduced for multi-seq demo
    float temperature = 1.0f;
    float topp = 0.9f;
    // char *user_prompt = NULL; // If user provides specific prompt
    int chunk_size = INT_MAX; // Default: Full Sequence Prefill

    // Argument parsing
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) { temperature = atof(argv[++i]); }
        }
        else if (strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) { topp = atof(argv[++i]); }
        }
        else if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) { steps = atoi(argv[++i]); }
        }
        else if (strcmp(argv[i], "-i") == 0) {
            // if (i + 1 < argc) { user_prompt = argv[++i]; }
            i++; // skip
        }
        else if (strcmp(argv[i], "--paged") == 0) {
            g_visualize_paged = 1;
            g_paged_mode = 1;
        }
        else if (strcmp(argv[i], "--chunk-size") == 0) {
            if (i + 1 < argc) { chunk_size = atoi(argv[++i]); }
        }
        else if (i == 2 && isdigit(argv[i][0])) {
            steps = atoi(argv[i]);
        }
        else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (!g_paged_mode) {
        printf("WARNING: Enabling Paged Attention automatically for Continuous Batching.\n");
        g_visualize_paged = 1;
        g_paged_mode = 1;
    }
    
    // Validation
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;
    if (chunk_size <= 0) chunk_size = 1;


    // Initialize libraries
    log_init("nano_vllm.log");
    init_ops();

    Config config;
    Weights weights;
    Tokenizer tokenizer;
    Sampler sampler;

    printf("Initializing...\n");
    load_model(&weights, &config, model_path);
    
    // Batch State (Global Workspace)
    RunState batch_state;
    malloc_run_state(&batch_state, &config); // Allocates max_seq_len capacity
    int MAX_BATCH_CAPACITY = config.max_seq_len; // Safe upper bound
    
    // Define Batches: Long vs Short Race
    int BATCH_SIZE = 2;
    
    // Construct Long Prompt (Realistic Story)
    char* long_prompt = (char*)malloc(4096);
    strcpy(long_prompt, "Once upon a time, there was a little bird named Tweety. Tweety lived in a big tree with his family. One day, the sun was shining bright and the sky was blue. Tweety wanted to fly high in the sky. He flapped his little wings and jumped from the branch. Down, down, down he went! He was scared but he kept flapping. Suddenly, he felt the wind under his wings. He was flying! He flew over the green grass and the blue river. He saw a big cow eating grass. \"Hello Cow!\" chirped Tweety. The cow looked up and said \"Moo!\". Tweety was so happy. He flew higher and higher until he saw a white cloud. He wanted to touch it. But then, he saw something else. It was a big, scary hawk! The hawk was looking at Tweety with hungry eyes. Tweety was afraid. He flew as fast as he could back to his tree. He hid under a big leaf. The hawk flew away. Tweety was safe. He promised his mom he would be careful. His mom gave him a big hug and a worm. Tweety was happy again. The next day, Tweety ");
    
    char* prompts[2] = {
        long_prompt,      // Seq 0: Long
        "One day, a cat"  // Seq 1: Short
    };
    
    int arrival_steps[2] = { 0, 5 }; // Seq 1 arrives late
    int steps_per_seq[2] = { steps, steps }; // Just generate a few tokens
    
    // Initialize Sequences
    Sequence seqs[2];
    
    for(int i=0; i<BATCH_SIZE; i++) {
        seqs[i].id = i;
        seqs[i].active = 0; // Starts Inactive (Waiting)
        seqs[i].status = SEQ_WAITING;
        seqs[i].arrival_step = arrival_steps[i];
        seqs[i].pos = 0;
        seqs[i].seq_len = steps_per_seq[i]; // Total steps to generate
        
        // We do NOT use seqs[i].state for computation anymore.
        // We only use it to store minimal state if needed, or just NULL it.
        // But the struct expects it? Let's allocate it just to avoid null pointers in cleanup if we want, 
        // but for now we won't use it.
        seqs[i].state = NULL; 
        
        // Alloc History
        seqs[i].output_history = (int*)malloc((2048 + steps) * sizeof(int)); // Safe large size
    }
    
    // Phase 3: Initialize PagedAttention
    if (g_paged_mode) {
        int block_size = 16;
        int total_needed_blocks = 0;
        for(int i=0; i<BATCH_SIZE; i++) {
            int needed = (config.max_seq_len + block_size - 1) / block_size;
            total_needed_blocks += needed;
        }
        
        init_kv_cache_manager(&g_kv_manager, block_size, total_needed_blocks, config.n_layers, config.n_kv_heads, config.head_dim);
        
        // Init Block Tables
        for(int i=0; i<BATCH_SIZE; i++) {
            int max_blocks = (config.max_seq_len + block_size - 1) / block_size;
            seqs[i].table.block_indices = (int*)malloc(max_blocks * sizeof(int));
            seqs[i].table.num_blocks = 0;
        }
        
        log_printf("PagedAttention Initialized (Block Size: %d, Pool: %d blocks)\n", block_size, total_needed_blocks);
    }
    
    // Build tokenizer
    char tokenizer_path[1024];
    const char* last_slash = strrchr(model_path, '/');
    if (last_slash != NULL) {
        size_t dir_len = last_slash - model_path + 1;
        strncpy(tokenizer_path, model_path, dir_len);
        tokenizer_path[dir_len] = '\0';
        strcat(tokenizer_path, "tokenizer.bin");
    } else {
        strcpy(tokenizer_path, "tokenizer.bin");
    }
    
    build_tokenizer(&tokenizer, tokenizer_path, config.vocab_size);
    build_sampler(&sampler, config.vocab_size, temperature, topp, (unsigned long long)time(NULL)); 
    
    // Encode Prompts
    for(int i=0; i<BATCH_SIZE; i++) {
        seqs[i].prompt_tokens = (int*)malloc((strlen(prompts[i]) + 3) * sizeof(int)); 
        encode(&tokenizer, prompts[i], 1, 0, seqs[i].prompt_tokens, &seqs[i].num_prompt_tokens);
        seqs[i].current_token = seqs[i].prompt_tokens[0]; 
        
        for(int j=0; j<seqs[i].num_prompt_tokens; j++) {
            seqs[i].output_history[j] = seqs[i].prompt_tokens[j];
        }
        seqs[i].seq_len = seqs[i].num_prompt_tokens + steps; // input length + steps to generate
    }

    log_printf("Starting Continuous Batching (Ragged) Demo (Chunk Size: %d)\n", chunk_size);
    log_printf("Seq 0 (Long): %d tokens (Arrives Step 0)\n", seqs[0].num_prompt_tokens);
    log_printf("Seq 1 (Short): %d tokens (Arrives Step 5)\n", seqs[1].num_prompt_tokens);
    
    clock_t start = clock();
    
    int global_step = 0;
    int all_finished = 0;
    
    // Timeline History
    int history_capacity = 1024;
    char* history_log = (char*)malloc(BATCH_SIZE * history_capacity * sizeof(char));
    for(int i=0; i<BATCH_SIZE * history_capacity; i++) history_log[i] = ' ';
    
    // Batch Buffers
    int* batch_tokens = (int*)malloc(MAX_BATCH_CAPACITY * sizeof(int));
    int* batch_pos = (int*)malloc(MAX_BATCH_CAPACITY * sizeof(int));
    int* batch_seq_ids = (int*)malloc(MAX_BATCH_CAPACITY * sizeof(int));
    int* batch_output_indices = (int*)malloc(BATCH_SIZE * sizeof(int)); // Max 1 output per seq per step
    BlockTable** batch_block_tables = (BlockTable**)malloc(BATCH_SIZE * sizeof(BlockTable*));
    
    // Initialize BlockTable ptrs
    for(int i=0; i<BATCH_SIZE; i++) batch_block_tables[i] = &seqs[i].table;

    // Main Batch Loop
    while (!all_finished) {
        all_finished = 1;
        
        // Reset Batch Counters
        int batch_count = 0;
        int output_count = 0;
        
        log_printf("\n--- Step %d ---\n", global_step);
        
        // 1. Scheduler Phase: Collect Tokens
        for (int i = 0; i < BATCH_SIZE; i++) {
             // Check Arrival
            if (global_step < seqs[i].arrival_step) {
                if (global_step < history_capacity) history_log[i*history_capacity + global_step] = 'W';
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
                if (global_step < history_capacity) history_log[i*history_capacity + global_step] = 'F';
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
                if (global_step < history_capacity) history_log[i*history_capacity + global_step] = 'P';
                
                int end_pos = seqs[i].num_prompt_tokens;
                int remaining = end_pos - start_pos;
                n_tokens = (remaining > chunk_size) ? chunk_size : remaining;
                token_ptr = seqs[i].prompt_tokens + start_pos;
                
            } else {
                seqs[i].status = SEQ_DECODING;
                if (global_step < history_capacity) history_log[i*history_capacity + global_step] = 'D';
                
                n_tokens = 1;
                token_ptr = &seqs[i].current_token;
            }
            
            // Check Capacity
            if (batch_count + n_tokens > MAX_BATCH_CAPACITY) {
                log_printf("WARNING: Batch capacity reached! Skipping Seq %d this step.\n", i);
                continue; // Wait for next step
            }
            
            // Alloc Blocks if needed (for Paged Attention)
            // We check each token position
            int block_size = g_kv_manager.block_size;
            for (int t = 0; t < n_tokens; t++) {
                int current_pos = start_pos + t;
                if (current_pos % block_size == 0) {
                    int new_block = alloc_block(&g_kv_manager);
                    if (new_block == -1) {
                        printf("Error: Out of KV Cache blocks!\n");
                        exit(1);
                    }
                    int logical_idx = current_pos / block_size;
                    seqs[i].table.block_indices[logical_idx] = new_block;
                    seqs[i].table.num_blocks++;
                }
            }
            
            // Add to Batch
            for (int t = 0; t < n_tokens; t++) {
                batch_tokens[batch_count + t] = token_ptr[t];
                batch_pos[batch_count + t] = start_pos + t;
                batch_seq_ids[batch_count + t] = i;
            }
            
            batch_output_indices[output_count] = batch_count + n_tokens - 1;
            // batch_output_seq_ids[output_count] = i; // implicit if we iterate active seqs again? No.
            // Let's use `batch_seq_ids` of the output token to identify the sequence.
            
            output_count++;
            batch_count += n_tokens;
            
            seqs[i].pos += n_tokens;
            
            if (is_prefill) {
                 seqs[i].current_token = token_ptr[n_tokens - 1]; // Last token of chunk
            }
        }
        
        if (batch_count == 0 && !all_finished) {
            // Everyone waiting?
            global_step++;
            continue;
        }
        
        if (all_finished) break;
        
        // 2. Inference
        transformer_batch(batch_tokens, batch_count, batch_pos, batch_seq_ids, 
                          batch_output_indices, output_count,
                          &config, &batch_state, &weights, batch_block_tables);
                          
        // 3. Sampling & Update
        for (int k = 0; k < output_count; k++) {
            int batch_idx = batch_output_indices[k];
            int seq_id = batch_seq_ids[batch_idx];
            
            // Logits are in batch_state.logits[k * vocab_size]
            float* logits = batch_state.logits + k * config.vocab_size;
            
            // Sample
            float* host_logits;
            #if defined(__CUDACC__) || defined(NANO_CUDA)
                host_logits = (float*)malloc(config.vocab_size * sizeof(float));
                check_status(device_memcpy(host_logits, logits, config.vocab_size * sizeof(float), DEVICE_TO_HOST));
            #else
                host_logits = logits;
            #endif
            
            int next_token = sample(&sampler, host_logits);
            
            #if defined(__CUDACC__) || defined(NANO_CUDA)
                free(host_logits);
            #endif
            
            // Process Result
            // Check if we just finished a prompt
            // int was_prefill = (seqs[seq_id].pos <= seqs[seq_id].num_prompt_tokens); 
            // actually we already advanced `pos` by `n_tokens`.
            
            // Log
            // const char* text = decode_token(&tokenizer, next_token);
            // log_printf("[Seq %d] Gen: %s\n", seq_id, text);
            
            // Store
            if (seqs[seq_id].pos < seqs[seq_id].seq_len) {
                 seqs[seq_id].output_history[seqs[seq_id].pos] = next_token;
            }
            
            // Update current token for next step
            seqs[seq_id].current_token = next_token;
            
             // Check Finish
            if (seqs[seq_id].pos >= seqs[seq_id].seq_len) {
                seqs[seq_id].active = 0;
                seqs[seq_id].status = SEQ_FINISHED;
                log_printf("[Seq %d] FINISHED.\n", seq_id);
            }
        }
        
        // Visualize Memory State
        if (batch_count > 0 || global_step < 10) { 
             visualize_kv_cache_usage(seqs, BATCH_SIZE, &g_kv_manager, config.max_seq_len, g_visualize_paged);
        }
        global_step++;
        
        if (global_step > 500) break;
    }
    
    log_printf("\nAll sequences finished.\n");
    
    // VISUALIZE TIMELINE
    visualize_final_timeline(history_log, BATCH_SIZE, global_step, history_capacity);
    
    // Print Final Summaries
    log_printf("\n=== Final Generated Sequences ===\n");
    for(int i=0; i<BATCH_SIZE; i++) {
        log_printf("[Seq %d]: ", i);
        int max_print = seqs[i].pos; 
        if (max_print > seqs[i].seq_len) max_print = seqs[i].seq_len;
        
        for(int j=0; j<max_print; j++) {
            const char* text = decode_token(&tokenizer, seqs[i].output_history[j]);
            log_printf("%s", text);
        }
        log_printf("\n");
    }
    log_printf("=================================\n\n");
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    log_printf("Time: %.2fs\n", time_spent);

    // Cleanup
    free(history_log);
    free(long_prompt);
    free_tokenizer(&tokenizer);
    
    free(batch_tokens);
    free(batch_pos);
    free(batch_seq_ids);
    free(batch_output_indices);
    free(batch_block_tables);
    free_run_state(&batch_state);
    
    for(int i=0; i<BATCH_SIZE; i++) {
        if (seqs[i].state) free(seqs[i].state); // If we alloced it
        free(seqs[i].prompt_tokens);
        free(seqs[i].output_history);
        if (g_paged_mode) {
            free(seqs[i].table.block_indices);
        }
    }
    free_ops();
    
    if (g_paged_mode) {
        free_kv_cache_manager(&g_kv_manager);
    }
    
    log_close();
    
    return 0;
}
