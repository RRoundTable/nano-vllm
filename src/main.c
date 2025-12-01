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
BlockTable g_block_table;

void transformer(int token, int pos, Config* p, RunState* s, Weights* w) {
    
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
            g_block_table.block_indices[logical_idx] = new_block;
            g_block_table.num_blocks++;
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
            update_kv_cache_paged(&g_kv_manager, &g_block_table, s->k, s->v, 
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
            paged_attention(s->xb2, s->q, &g_kv_manager, &g_block_table, s->att,
                            i, pos, p->max_seq_len, p->n_heads, p->n_kv_heads, p->head_dim);
        } else {
            // Naive Attention
            multi_head_attention(s->xb2, s->q, s->key_cache, s->value_cache, s->att,
                                 i, pos, p->max_seq_len, p->n_heads, p->n_kv_heads, p->head_dim);
        }
        
        // [Visualizer] Show KV Cache Usage for Layer 0
        #ifndef __CUDACC__
        if (i == 0) {
            // visualize_attention(s->att, p->n_heads, pos, p->max_seq_len); // Disabled for cleaner output
            visualize_kv_cache_usage(i, pos, p->max_seq_len, g_visualize_paged);
        }
        #endif
        
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

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path> [steps] [--paged]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    int steps = 256;
    float temperature = 1.0f;
    float topp = 0.9f;

    // poor man's C argparse so we can override the defaults above from the command line
    // Usage: ./run <model_path> [options]
    // Options: -t <temp> -p <topp> -n <steps>
    
    char *prompt = NULL;
    
    for (int i = 2; i < argc; i++) {
        if (argv[i][0] == '-') {
            if (argv[i][1] == 't') { 
                if (i + 1 < argc) { temperature = atof(argv[i + 1]); i++; }
            }
            else if (argv[i][1] == 'p') { 
                if (i + 1 < argc) { topp = atof(argv[i + 1]); i++; }
            }
            else if (argv[i][1] == 'n') { 
                if (i + 1 < argc) { steps = atoi(argv[i + 1]); i++; }
            }
            else if (argv[i][1] == 'i') { 
                if (i + 1 < argc) { prompt = argv[i + 1]; i++; }
            }
            else if (strcmp(argv[i], "--paged") == 0) {
                g_visualize_paged = 1;
                g_paged_mode = 1;
            }
        } else {
            // Compatibility with old positional arg [steps]
            // If it's just a number and not an option value
             if (i == 2 && isdigit(argv[i][0])) {
                steps = atoi(argv[i]);
            }
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
    RunState state;
    Tokenizer tokenizer;
    Sampler sampler;

    printf("Initializing...\n");
    load_model(&weights, &config, model_path);
    malloc_run_state(&state, &config);
    
    // Phase 3: Initialize PagedAttention
    if (g_paged_mode) {
        int block_size = 16;
        // Total blocks = (max_seq_len / block_size) * something ? 
        // For this demo, let's allocate slightly more than needed for one seq
        int num_blocks = (config.max_seq_len + block_size - 1) / block_size + 8; 
        
        init_kv_cache_manager(&g_kv_manager, block_size, num_blocks, config.n_layers, config.n_kv_heads, config.head_dim);
        
        // Init Block Table for the single sequence
        g_block_table.block_indices = (int*)malloc(num_blocks * sizeof(int));
        g_block_table.num_blocks = 0;
        
        log_printf("PagedAttention Initialized (Block Size: %d, Pool: %d blocks)\n", block_size, num_blocks);
    }
    
    // Build tokenizer path from model path
    // e.g., "python_ref/data/model.bin" -> "python_ref/data/tokenizer.bin"
    char tokenizer_path[1024];
    const char* last_slash = strrchr(model_path, '/');
    if (last_slash != NULL) {
        // Copy directory path
        size_t dir_len = last_slash - model_path + 1;
        strncpy(tokenizer_path, model_path, dir_len);
        tokenizer_path[dir_len] = '\0';
        strcat(tokenizer_path, "tokenizer.bin");
    } else {
        // No directory separator, assume current directory
        strcpy(tokenizer_path, "tokenizer.bin");
    }
    
    build_tokenizer(&tokenizer, tokenizer_path, config.vocab_size);
    build_sampler(&sampler, config.vocab_size, temperature, topp, (unsigned long long)time(NULL)); // Seed with time
    
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    // +3 for '\0', ?BOS, ?EOS
    if (prompt == NULL) { prompt = ""; }
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int)); 
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;
    
    log_printf("Starting inference for %d steps (temp=%f, topp=%f)...\n", steps, temperature, topp);
    if (g_visualize_paged) {
        log_printf("Visualization Mode: Paged KV Cache (REAL LOGIC)\n");
    } else {
        log_printf("Visualization Mode: Linear KV Cache (Naive)\n");
    }

    clock_t start = clock();
    
    // Buffer for accumulating text
    char* text_buffer = (char*)malloc(steps * 100 + 1024);
    text_buffer[0] = '\0';
    
    int next_token;
    
    for (pos = 0; pos < steps; pos++) {
        transformer(token, pos, &config, &state, &weights);
        
        // Advance state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next_token = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            
            // Copy logits to host to find max
            float* host_logits;
            #ifdef __CUDACC__
                host_logits = (float*)malloc(config.vocab_size * sizeof(float));
                check_status(device_memcpy(host_logits, state.logits, config.vocab_size * sizeof(float), DEVICE_TO_HOST));
            #else
                // In CPU mode, state.logits is already on host
                host_logits = state.logits;
            #endif
            
            // Sample next token
            next_token = sample(&sampler, host_logits);
            
            #ifdef __CUDACC__
                free(host_logits);
            #endif
        }
        
        const char* text = decode_token(&tokenizer, next_token);
        // Print real-time token (optional, maybe just visualizer?)
        // We keep printing it so user sees progress.
        log_printf("%s", text);
        fflush(stdout);
        
        // Accumulate
        strcat(text_buffer, text);
        
        token = next_token;
    }
    log_printf("\n\n");
    
    // Print accumulated text
    log_printf("==================================================\n");
    log_printf("Generated Text:\n");
    log_printf("==================================================\n");
    log_printf("%s\n", text_buffer);
    log_printf("==================================================\n");
    
    free(text_buffer);
    free(prompt_tokens);

    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    log_printf("Time: %.2fs, %.2f tok/s\n", time_spent, steps / time_spent);

    free_tokenizer(&tokenizer);
    free_run_state(&state);
    free_ops();
    
    if (g_paged_mode) {
        free_kv_cache_manager(&g_kv_manager);
        free(g_block_table.block_indices);
    }
    
    log_close();
    // free weights...
    
    return 0;
}
