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

// Forward declarations
void load_model(Weights* w, Config* p, const char* checkpoint_path);
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);

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

void transformer(int token, int pos, Config* p, RunState* s, Weights* w, BlockTable* bt) {
    
    // Scheduler Logic: Allocate new block if needed (Only in Paged Mode)
    if (g_paged_mode) {
        int block_size = g_kv_manager.block_size;
        if (pos % block_size == 0) {
            // Need a new block for this position
            int new_block = alloc_block(&g_kv_manager);
            if (new_block == -1) {
                printf("Error: Out of KV Cache blocks!\n");
                exit(1);
            }
            int logical_idx = pos / block_size;
            // Resize block_indices if needed? (In this simple C code, we assume fixed max size or pre-allocated enough)
            // For this demo, we pre-allocated enough in main.
            bt->block_indices[logical_idx] = new_block;
            bt->num_blocks++;
            // log_printf("Allocated Block %d for Logical %d\n", new_block, logical_idx);
        }
    }

    // 1. Embedding
    float* content_row = w->token_embedding_table + token * p->dim;
    check_status(device_memcpy(s->x, content_row, p->dim * sizeof(float), DEVICE_TO_DEVICE));
    
    // 2. Forward layers
    for(int i = 0; i < p->n_layers; i++) {
        LayerWeights* l = &w->layers[i];
        
        // Attention Block
        // a. RMSNorm
        rms_norm(s->xb, s->x, l->rms_att_weight, p->dim, 1e-5f);
        
        // b. QKV Matmuls
        // xb [dim] @ wq [dim, n_heads * head_dim] -> q [n_heads * head_dim]
        matmul(s->q, s->xb, l->wq, p->dim, p->n_heads * p->head_dim);
        matmul(s->k, s->xb, l->wk, p->dim, p->n_kv_heads * p->head_dim);
        matmul(s->v, s->xb, l->wv, p->dim, p->n_kv_heads * p->head_dim);
        
        // c. RoPE
        apply_rope(s->q, s->k, pos, p->rope_theta, p->head_dim, p->n_heads, p->n_kv_heads);
        
        // d. KV Cache Update
        if (g_paged_mode) {
            // Paged Update (Writes to non-contiguous memory)
            update_kv_cache_paged(&g_kv_manager, bt, s->k, s->v, 
                                  i, pos, p->n_kv_heads, p->head_dim);
        } else {
            // Naive Update (Writes to contiguous memory)
            update_kv_cache(s->key_cache, s->value_cache, s->k, s->v, 
                            i, pos, p->max_seq_len, p->n_kv_heads, p->head_dim);
        }
        
        // e. Multi-Head Attention
        // out = xb2 (reusing buffer)
        if (g_paged_mode) {
            // Paged Attention (Reads from non-contiguous memory)
            paged_attention(s->xb2, s->q, &g_kv_manager, bt, s->att,
                            i, pos, p->max_seq_len, p->n_heads, p->n_kv_heads, p->head_dim);
        } else {
            // Naive Attention
            multi_head_attention(s->xb2, s->q, s->key_cache, s->value_cache, s->att,
                                 i, pos, p->max_seq_len, p->n_heads, p->n_kv_heads, p->head_dim);
        }
        
        // f. Output Projection
        // xb2 [n_heads * head_dim] @ wo [n_heads * head_dim, dim] -> xb [dim] (reuse xb)
        matmul(s->xb, s->xb2, l->wo, p->n_heads * p->head_dim, p->dim);
        
        // g. Residual Connection
        accum(s->x, s->xb, p->dim);
        
        // FFN Block
        // a. RMSNorm
        rms_norm(s->xb, s->x, l->rms_ffn_weight, p->dim, 1e-5f);
        
        // b. Gate & Up
        matmul(s->hb, s->xb, l->w_gate, p->dim, p->hidden_dim);
        matmul(s->hb2, s->xb, l->w_up, p->dim, p->hidden_dim);
        
        // c. SwiGLU
        swiglu(s->hb, s->hb, s->hb2, p->hidden_dim); // Result in hb
        
        // d. Down Projection
        matmul(s->xb, s->hb, l->w_down, p->hidden_dim, p->dim);
        
        // e. Residual Connection
        accum(s->x, s->xb, p->dim);
    }
    
    // 3. Final RMSNorm
    rms_norm(s->x, s->x, w->rms_final_weight, p->dim, 1e-5f);
    
    // 4. Classifier (LM Head)
    matmul(s->logits, s->x, w->lm_head, p->dim, p->vocab_size);
}

void print_usage(char *prog_name) {
    printf("Usage: %s <model_path> [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -t <temp>    Temperature (default: 1.0)\n");
    printf("  -p <topp>    Top-p value (default: 0.9)\n");
    printf("  -n <steps>   Number of generation steps (default: 64)\n");
    printf("  -i <prompt>  Input prompt (default: \"Hello, my name is\")\n");
    printf("  --paged      Enable paged attention mode\n");
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    int steps = 64; // Default reduced for multi-seq demo
    float temperature = 1.0f;
    float topp = 0.9f;
    char *user_prompt = NULL; // If user provides specific prompt

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
    
    // Define Batches
    int BATCH_SIZE = 4;
    // If user provided a prompt, use it for first seq, else default.
    char* prompts[4] = {
        user_prompt ? user_prompt : "Hello, my name is",
        "The quick brown fox jumps over",
        "Once upon a time in a distant land",
        "To be or not to be, that is"
    };
    // Different lengths for interest
    int steps_per_seq[4] = { steps, steps + 20, steps - 10, steps + 50 }; 
    if (steps_per_seq[2] < 10) steps_per_seq[2] = 10;
    
    // Initialize Sequences
    Sequence seqs[4];
    
    for(int i=0; i<BATCH_SIZE; i++) {
        seqs[i].id = i;
        seqs[i].active = 1;
        seqs[i].pos = 0;
        seqs[i].seq_len = steps_per_seq[i]; // Total steps to generate
        
        // Alloc State
        seqs[i].state = (RunState*)malloc(sizeof(RunState));
        malloc_run_state(seqs[i].state, &config);
        
        // Alloc History
        seqs[i].output_history = (int*)malloc(seqs[i].seq_len * sizeof(int));

        // BlockTable Init (Wait until paged manager init)
    }
    
    // Phase 3: Initialize PagedAttention
    if (g_paged_mode) {
        int block_size = 16;
        // Total blocks = Sum of max needs for all seqs
        int total_needed_blocks = 0;
        for(int i=0; i<BATCH_SIZE; i++) {
            // Approx max len = max_seq_len (or actual seq_len)
            // Let's reserve enough for max_seq_len for safety in demo
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
        // Note: seq_len might be smaller than prompt if user sets very small steps, handle that?
        // Assuming seq_len >= num_prompt_tokens for now or just filling what fits.
        for(int j=0; j<seqs[i].num_prompt_tokens && j < seqs[i].seq_len; j++) {
            seqs[i].output_history[j] = seqs[i].prompt_tokens[j];
        }
    }

    log_printf("Starting Multi-Sequence Inference (%d sequences)...\n", BATCH_SIZE);
    if (g_visualize_paged) {
        log_printf("Visualization Mode: Paged KV Cache (REAL LOGIC)\n");
    } else {
        log_printf("Visualization Mode: Linear KV Cache (Naive)\n");
    }

    clock_t start = clock();
    
    int any_active = 1;
    int global_step = 0;
    
    // Main Batch Loop
    while (any_active) {
        any_active = 0;
        log_printf("\n--- Step %d ---\n", global_step);
        
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (!seqs[i].active) continue;
            any_active = 1;
            
            // Run Transformer
            // Note: For naive mode, update_kv_cache uses 's->key_cache' inside RunState, so no extra block table needed.
            // But for Paged, we pass seqs[i].table.
            BlockTable* bt_ptr = g_paged_mode ? &seqs[i].table : NULL;
            transformer(seqs[i].current_token, seqs[i].pos, &config, seqs[i].state, &weights, bt_ptr);
            
            // Next Token Logic
            int next_token;
            if (seqs[i].pos < seqs[i].num_prompt_tokens - 1) {
                next_token = seqs[i].prompt_tokens[seqs[i].pos + 1];
            } else {
                // Sample
                float* host_logits;
                #if defined(__CUDACC__) || defined(NANO_CUDA)
                    host_logits = (float*)malloc(config.vocab_size * sizeof(float));
                    check_status(device_memcpy(host_logits, seqs[i].state->logits, config.vocab_size * sizeof(float), DEVICE_TO_HOST));
                #else
                    host_logits = seqs[i].state->logits;
                #endif
                
                next_token = sample(&sampler, host_logits);
                
                #if defined(__CUDACC__) || defined(NANO_CUDA)
                    free(host_logits);
                #endif
            }
            
            // Print
            const char* text = decode_token(&tokenizer, next_token);
            log_printf("[Seq %d]: %s\n", i, text);
            
            // Update History
            if (seqs[i].pos + 1 < seqs[i].seq_len) {
                seqs[i].output_history[seqs[i].pos + 1] = next_token;
            }
            
            // Update State
            seqs[i].current_token = next_token;
            seqs[i].pos++;
            
            // Check Finish
            // Use seq_len as generation limit (simple)
            if (seqs[i].pos >= seqs[i].seq_len) {
                seqs[i].active = 0;
                log_printf("[Seq %d] FINISHED.\n", i);
            }
        }
        
        // Visualize Memory State (Once per step, after all seqs updated)
        if (any_active && global_step % 1 == 0) { // Visualize every step
             visualize_kv_cache_usage(seqs, BATCH_SIZE, &g_kv_manager, config.max_seq_len, g_visualize_paged);
        }
        global_step++;
    }
    
    log_printf("\nAll sequences finished.\n");
    
    // Print Final Summaries
    log_printf("\n=== Final Generated Sequences ===\n");
    for(int i=0; i<BATCH_SIZE; i++) {
        log_printf("[Seq %d]: ", i);
        for(int j=0; j<seqs[i].seq_len; j++) {
            // Note: decode_token returns a static buffer, so we must print immediately or copy
            // Also, some tokens might be partial or special, but decode_token handles basic piece lookup
            // If pos < seq_len (finished early?), we should use seqs[i].pos
            if (j > seqs[i].pos) break;
            
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
