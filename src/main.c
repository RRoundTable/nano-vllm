#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "structs.h"
#include "ops.h"
#include "nano_cuda.h"
#include "log.h"

// Forward declarations
void load_model(Weights* w, Config* p, const char* checkpoint_path);
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);

// Tokenizer (Minimal)
typedef struct {
    char** vocab;
    float* vocab_scores;
    int vocab_size;
    unsigned char byte_pieces[512];
} Tokenizer;

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
const char* decode_token(Tokenizer* t, int token);

// Sampler
typedef struct {
    unsigned long long state;
} Sampler;

void build_sampler(Sampler* s, unsigned long long seed);
int sample(Sampler* s, float* logits, int vocab_size, float temperature);

// Global config for visualization
int g_visualize_paged = 0;

void transformer(int token, int pos, Config* p, RunState* s, Weights* w) {
    
    // 1. Embedding
    float* content_row = w->token_embedding_table + token * p->dim;
    cudaCheck(cudaMemcpy(s->x, content_row, p->dim * sizeof(float), cudaMemcpyDeviceToDevice));
    
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
        update_kv_cache(s->key_cache, s->value_cache, s->k, s->v, 
                        i, pos, p->max_seq_len, p->n_kv_heads, p->head_dim);
        
        // e. Multi-Head Attention
        // out = xb2 (reusing buffer)
        multi_head_attention(s->xb2, s->q, s->key_cache, s->value_cache, s->att,
                             i, pos, p->max_seq_len, p->n_heads, p->n_kv_heads, p->head_dim);
        
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
    int steps = 10;
    if (argc >= 3) {
        steps = atoi(argv[2]);
    }
    
    // Parse --paged flag
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--paged") == 0) {
            g_visualize_paged = 1;
        }
    }

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
    
    // Try to load tokenizer if it exists
    build_tokenizer(&tokenizer, "data/tokenizer.bin", config.vocab_size);
    build_sampler(&sampler, (unsigned long long)time(NULL)); // Seed with time
    
    int token = 1; // BOS token
    int pos = 0;
    
    log_printf("Starting inference for %d steps...\n", steps);
    if (g_visualize_paged) {
        log_printf("Visualization Mode: Paged KV Cache (Simulated)\n");
    } else {
        log_printf("Visualization Mode: Linear KV Cache (Naive)\n");
    }

    clock_t start = clock();
    
    // Buffer for accumulating text
    char* text_buffer = (char*)malloc(steps * 100 + 1024);
    text_buffer[0] = '\0';
    
    for (pos = 0; pos < steps; pos++) {
        transformer(token, pos, &config, &state, &weights);
        
        // Copy logits to host to find max
        float* host_logits;
        #ifdef __CUDACC__
            host_logits = (float*)malloc(config.vocab_size * sizeof(float));
            cudaCheck(cudaMemcpy(host_logits, state.logits, config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        #else
            // In CPU mode, state.logits is already on host
            host_logits = state.logits;
        #endif
        
        // Sample next token (Temperature = 1.0f)
        int next_token = sample(&sampler, host_logits, config.vocab_size, 1.0f);
        
        #ifdef __CUDACC__
            free(host_logits);
        #endif
        
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
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    log_printf("Time: %.2fs, %.2f tok/s\n", time_spent, steps / time_spent);

    free_tokenizer(&tokenizer);
    free_run_state(&state);
    free_ops();
    log_close();
    // free weights...
    
    return 0;
}
