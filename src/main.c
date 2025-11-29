#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "structs.h"
#include "ops.h"
#include "nano_cuda.h"

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
        
        // [Visualizer] Show Attention Patterns for Layer 0
        #ifndef __CUDACC__
        if (i == 0) {
            visualize_attention(s->att, p->n_heads, pos, p->max_seq_len);
            visualize_kv_cache_usage(i, pos, p->max_seq_len);
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
        printf("Usage: %s <model_path> [steps]\n", argv[0]);
        return 1;
    }
    
    const char* model_path = argv[1];
    int steps = 10;
    if (argc >= 3) {
        steps = atoi(argv[2]);
    }

    // Initialize libraries
    init_ops();

    Config config;
    Weights weights;
    RunState state;
    Tokenizer tokenizer;

    printf("Initializing...\n");
    load_model(&weights, &config, model_path);
    malloc_run_state(&state, &config);
    
    // Try to load tokenizer if it exists
    build_tokenizer(&tokenizer, "data/tokenizer.bin", config.vocab_size);
    
    int token = 1; // BOS token
    int pos = 0;
    
    printf("Starting inference for %d steps...\n", steps);
    clock_t start = clock();
    
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
        
        int next_token = 0;
        float max_val = -INFINITY;
        for (int i = 0; i < config.vocab_size; i++) {
            if (host_logits[i] > max_val) {
                max_val = host_logits[i];
                next_token = i;
            }
        }
        
        #ifdef __CUDACC__
            free(host_logits);
        #endif
        
        const char* text = decode_token(&tokenizer, next_token);
        printf("%s", text);
        fflush(stdout);
        
        token = next_token;
    }
    printf("\n");
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2fs, %.2f tok/s\n", time_spent, steps / time_spent);

    free_tokenizer(&tokenizer);
    free_run_state(&state);
    free_ops();
    // free weights...
    
    return 0;
}
