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
    
    // Scheduler Logic: Allocate new block if needed (Only in Paged Mode)
    // NOTE: For batched processing, we might need multiple blocks.
    // For now, let's assume one block allocation per call or enough capacity.
    // In strict PagedAttention, we check block boundaries for EACH token.
    if (g_paged_mode) {
        int block_size = g_kv_manager.block_size;
        for (int t = 0; t < num_tokens; t++) {
            int current_pos = pos + t;
            if (current_pos % block_size == 0) {
                // Need a new block for this position
                int new_block = alloc_block(&g_kv_manager);
                if (new_block == -1) {
                    printf("Error: Out of KV Cache blocks!\n");
                    exit(1);
                }
                int logical_idx = current_pos / block_size;
                bt->block_indices[logical_idx] = new_block;
                bt->num_blocks++;
            }
        }
    }

    // 1. Embedding (Batched)
    for (int t = 0; t < num_tokens; t++) {
        float* content_row = w->token_embedding_table + tokens[t] * p->dim;
        // Copy to s->x + t * dim
        check_status(device_memcpy(s->x + t * p->dim, content_row, p->dim * sizeof(float), DEVICE_TO_DEVICE));
    }
    
    // 2. Forward layers
    for(int i = 0; i < p->n_layers; i++) {
        LayerWeights* l = &w->layers[i];
        
        // Attention Block
        // a. RMSNorm (Batched)
        rms_norm(s->xb, s->x, l->rms_att_weight, p->dim, num_tokens, 1e-5f);
        
        // b. QKV Matmuls (Batched: [num_tokens, dim] @ [dim, head_dim...])
        matmul(s->q, s->xb, l->wq, num_tokens, p->dim, p->n_heads * p->head_dim);
        matmul(s->k, s->xb, l->wk, num_tokens, p->dim, p->n_kv_heads * p->head_dim);
        matmul(s->v, s->xb, l->wv, num_tokens, p->dim, p->n_kv_heads * p->head_dim);
        
        // c. RoPE (Batched)
        apply_rope(s->q, s->k, pos, p->rope_theta, p->head_dim, num_tokens, p->n_heads, p->n_kv_heads);
        
        // d. KV Cache Update (Batched)
        if (g_paged_mode) {
            update_kv_cache_paged(&g_kv_manager, bt, s->k, s->v, 
                                  i, pos, p->n_kv_heads, p->head_dim, num_tokens);
        } else {
            update_kv_cache(s->key_cache, s->value_cache, s->k, s->v, 
                            i, pos, p->max_seq_len, p->n_kv_heads, p->head_dim, num_tokens);
        }
        
        // e. Multi-Head Attention (Batched)
        // out = xb2 (reusing buffer)
        if (g_paged_mode) {
            paged_attention(s->xb2, s->q, &g_kv_manager, bt, s->att,
                            i, pos, p->max_seq_len, p->n_heads, p->n_kv_heads, p->head_dim, num_tokens);
        } else {
            multi_head_attention(s->xb2, s->q, s->key_cache, s->value_cache, s->att,
                                 i, pos, p->max_seq_len, p->n_heads, p->n_kv_heads, p->head_dim, num_tokens);
        }
        
        // f. Output Projection (Batched)
        matmul(s->xb, s->xb2, l->wo, num_tokens, p->n_heads * p->head_dim, p->dim);
        
        // g. Residual Connection (Batched)
        accum(s->x, s->xb, num_tokens * p->dim);
        
        // FFN Block
        // a. RMSNorm
        rms_norm(s->xb, s->x, l->rms_ffn_weight, p->dim, num_tokens, 1e-5f);
        
        // b. Gate & Up
        matmul(s->hb, s->xb, l->w_gate, num_tokens, p->dim, p->hidden_dim);
        matmul(s->hb2, s->xb, l->w_up, num_tokens, p->dim, p->hidden_dim);
        
        // c. SwiGLU
        swiglu(s->hb, s->hb, s->hb2, p->hidden_dim, num_tokens); // Result in hb
        
        // d. Down Projection
        matmul(s->xb, s->hb, l->w_down, num_tokens, p->hidden_dim, p->dim);
        
        // e. Residual Connection
        accum(s->x, s->xb, num_tokens * p->dim);
    }
    
    // 3. Final RMSNorm (Batched)
    rms_norm(s->x, s->x, w->rms_final_weight, p->dim, num_tokens, 1e-5f);
    
    // 4. Classifier (LM Head)
    // Optimization: We only need the logits for the LAST token in the batch to predict the next token.
    // However, if we wanted to support "prefill" phase output for all tokens, we would compute all.
    // Let's compute ONLY for the last token to save time, as we only sample from the last one.
    // The last token is at index (num_tokens - 1).
    float* last_token_embedding = s->x + (num_tokens - 1) * p->dim;
    
    // We treat this as a batch of size 1 for the classifier
    matmul(s->logits, last_token_embedding, w->lm_head, 1, p->dim, p->vocab_size);
}

void print_usage(char *prog_name) {
    printf("Usage: %s <model_path> [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -t <temp>    Temperature (default: 1.0)\n");
    printf("  -p <topp>    Top-p value (default: 0.9)\n");
    printf("  -n <steps>   Number of generation steps (default: 64)\n");
    printf("  -i <prompt>  Input prompt (default: \"Hello, my name is\")\n");
    printf("  --paged      Enable paged attention mode\n");
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
    char *user_prompt = NULL; // If user provides specific prompt
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
            if (i + 1 < argc) { user_prompt = argv[++i]; }
        }
        else if (strcmp(argv[i], "--paged") == 0) {
            g_visualize_paged = 1;
            g_paged_mode = 1;
        }
        else if (strcmp(argv[i], "--chunk-size") == 0) {
            if (i + 1 < argc) { chunk_size = atoi(argv[++i]); }
        }
        else if (i == 2 && isdigit(argv[i][0])) {
            // Compatibility with old positional arg [steps]
            steps = atoi(argv[i]);
        }
        else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
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
    // RunState state; // REPLACED by Sequence.state
    Tokenizer tokenizer;
    Sampler sampler;

    printf("Initializing...\n");
    load_model(&weights, &config, model_path);
    
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
        
        // Alloc State
        seqs[i].state = (RunState*)malloc(sizeof(RunState));
        malloc_run_state(seqs[i].state, &config);
        
        // Alloc History
        seqs[i].output_history = (int*)malloc((2048 + steps) * sizeof(int)); // Safe large size

        // BlockTable Init (Wait until paged manager init)
    }
    
    // Phase 3: Initialize PagedAttention
    if (g_paged_mode) {
        int block_size = 16;
        // Total blocks = Sum of max needs for all seqs
        int total_needed_blocks = 0;
        for(int i=0; i<BATCH_SIZE; i++) {
            // Approx max len = max_seq_len (or actual seq_len)
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
    
    // Build tokenizer path
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
    build_sampler(&sampler, config.vocab_size, temperature, topp, (unsigned long long)time(NULL)); // Seed with time
    
    // Encode Prompts
    for(int i=0; i<BATCH_SIZE; i++) {
        // +3 for '\0', ?BOS, ?EOS
        seqs[i].prompt_tokens = (int*)malloc((strlen(prompts[i]) + 3) * sizeof(int)); 
        encode(&tokenizer, prompts[i], 1, 0, seqs[i].prompt_tokens, &seqs[i].num_prompt_tokens);
        seqs[i].current_token = seqs[i].prompt_tokens[0]; // Start token
        
        // Init history with prompt
        for(int j=0; j<seqs[i].num_prompt_tokens; j++) {
            seqs[i].output_history[j] = seqs[i].prompt_tokens[j];
        }
        // Adjust seq_len to be prompt len + generated steps
        seqs[i].seq_len = seqs[i].num_prompt_tokens + steps;
    }

    log_printf("Starting Interactive Scheduler Demo (Chunk Size: %d)\n", chunk_size);
    log_printf("Seq 0 (Long): %d tokens (Arrives Step 0)\n", seqs[0].num_prompt_tokens);
    log_printf("Seq 1 (Short): %d tokens (Arrives Step 5)\n", seqs[1].num_prompt_tokens);
    
    if (g_visualize_paged) {
        log_printf("Visualization Mode: Paged KV Cache (REAL LOGIC)\n");
    } else {
        log_printf("Visualization Mode: Linear KV Cache (Naive)\n");
    }

    clock_t start = clock();
    
    int global_step = 0;
    int all_finished = 0;
    
    // Timeline History
    int history_capacity = 1024;
    char* history_log = (char*)malloc(BATCH_SIZE * history_capacity * sizeof(char));
    for(int i=0; i<BATCH_SIZE * history_capacity; i++) history_log[i] = ' ';
    
    // Main Batch Loop
    while (!all_finished) {
        all_finished = 1;
        log_printf("\n--- Step %d ---\n", global_step);
        
        int active_count = 0;
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            // Check Arrival
            if (global_step < seqs[i].arrival_step) {
                // Not arrived yet
                if (global_step < history_capacity) history_log[i*history_capacity + global_step] = 'W';
                all_finished = 0; // Keep loop running
                continue;
            }
            
            // Activate if waiting
            if (seqs[i].status == SEQ_WAITING) {
                seqs[i].active = 1;
                seqs[i].status = SEQ_PREFILLING;
                log_printf(">>> [Scheduler] Seq %d ARRIVED! (Prompt: %d tokens)\n", i, seqs[i].num_prompt_tokens);
            }
            
            if (seqs[i].status == SEQ_FINISHED) {
                if (global_step < history_capacity) history_log[i*history_capacity + global_step] = 'F';
                continue;
            }
            
            all_finished = 0;
            active_count++;
            
            // Check if we are in Prefill Phase or Decode Phase
            int is_prefill = (seqs[i].pos < seqs[i].num_prompt_tokens - 1); 
            
            if (is_prefill) {
                // --- CHUNKED PREFILL ---
                seqs[i].status = SEQ_PREFILLING;
                
                // Record to History
                if (global_step < history_capacity) history_log[i*history_capacity + global_step] = 'P';

                int start_pos = seqs[i].pos;
                int end_pos = seqs[i].num_prompt_tokens; 
                int remaining = end_pos - start_pos;
                int n_tokens = (remaining > chunk_size) ? chunk_size : remaining;
                
                // Point to the chunk of tokens
                int* chunk_ptr = seqs[i].prompt_tokens + start_pos;
                
                BlockTable* bt_ptr = g_paged_mode ? &seqs[i].table : NULL;
                transformer(chunk_ptr, n_tokens, start_pos, &config, seqs[i].state, &weights, bt_ptr);
                
                // Update position
                seqs[i].pos += n_tokens; 
                
                // Update current_token
                seqs[i].current_token = chunk_ptr[n_tokens - 1];
                
                int finished_prompt = (seqs[i].pos >= seqs[i].num_prompt_tokens);
                
                if (finished_prompt) {
                    // Sample next token (First Generation)
                    float* host_logits;
                    #if defined(__CUDACC__) || defined(NANO_CUDA)
                        host_logits = (float*)malloc(config.vocab_size * sizeof(float));
                        check_status(device_memcpy(host_logits, seqs[i].state->logits, config.vocab_size * sizeof(float), DEVICE_TO_HOST));
                    #else
                        host_logits = seqs[i].state->logits;
                    #endif
                    
                    int next_token = sample(&sampler, host_logits);
                    
                    #if defined(__CUDACC__) || defined(NANO_CUDA)
                        free(host_logits);
                    #endif
                    
                    // Log and Store
                    const char* text = decode_token(&tokenizer, next_token);
                    log_printf("[Seq %d] PREFILL COMPLETE (%d tokens). First Gen: %s\n", i, n_tokens, text);
                    
                    if (seqs[i].pos < seqs[i].seq_len) {
                        seqs[i].output_history[seqs[i].pos] = next_token;
                    }
                    
                    seqs[i].current_token = next_token;
                    seqs[i].status = SEQ_DECODING;
                    
                } else {
                    log_printf("[Seq %d] Processed Chunk (%d tokens). Pos: %d/%d\n", i, n_tokens, seqs[i].pos, seqs[i].num_prompt_tokens);
                }
                
            } else {
                // --- DECODE PHASE (Single Token) ---
                seqs[i].status = SEQ_DECODING;
                
                // Record to History
                if (global_step < history_capacity) history_log[i*history_capacity + global_step] = 'D';

                BlockTable* bt_ptr = g_paged_mode ? &seqs[i].table : NULL;
                
                int token_arr[1] = { seqs[i].current_token };
                transformer(token_arr, 1, seqs[i].pos, &config, seqs[i].state, &weights, bt_ptr);
                
                float* host_logits;
                #if defined(__CUDACC__) || defined(NANO_CUDA)
                    host_logits = (float*)malloc(config.vocab_size * sizeof(float));
                    check_status(device_memcpy(host_logits, seqs[i].state->logits, config.vocab_size * sizeof(float), DEVICE_TO_HOST));
                #else
                    host_logits = seqs[i].state->logits;
                #endif
                
                int next_token = sample(&sampler, host_logits);
                
                #if defined(__CUDACC__) || defined(NANO_CUDA)
                    free(host_logits);
                #endif
                
                const char* text = decode_token(&tokenizer, next_token);
                log_printf("[Seq %d]: %s\n", i, text);
                
                if (seqs[i].pos + 1 < seqs[i].seq_len) {
                    seqs[i].output_history[seqs[i].pos + 1] = next_token;
                }
                
                seqs[i].current_token = next_token;
                seqs[i].pos++;
            }
            
            // Check Finish
            if (seqs[i].pos >= seqs[i].seq_len) {
                seqs[i].active = 0;
                seqs[i].status = SEQ_FINISHED;
                log_printf("[Seq %d] FINISHED.\n", i);
            }
        }
        
        // Visualize Memory State
        if (active_count > 0 || global_step < 10) { // Keep visualizing for a bit
             visualize_kv_cache_usage(seqs, BATCH_SIZE, &g_kv_manager, config.max_seq_len, g_visualize_paged);
        }
        global_step++;
        
        // Safety break
        if (global_step > 500) break;
    }
    
    log_printf("\nAll sequences finished.\n");
    
    // VISUALIZE TIMELINE
    visualize_final_timeline(history_log, BATCH_SIZE, global_step, history_capacity);
    
    // Print Final Summaries
    log_printf("\n=== Final Generated Sequences ===\n");
    for(int i=0; i<BATCH_SIZE; i++) {
        log_printf("[Seq %d]: ", i);
        // Be careful with seq_len vs pos
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
    for(int i=0; i<BATCH_SIZE; i++) {
        free_run_state(seqs[i].state);
        free(seqs[i].state);
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
