#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "structs.h"
#include "ops.h"
#include "nano_cuda.h"

// Forward declarations
void load_model(Weights* w, Config* p, const char* checkpoint_path);
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);

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
        // Note: Our matmul is: out = in @ weight
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
    // x [dim] @ lm_head [dim, vocab_size] -> logits [vocab_size]
    // Wait, lm_head stored as [vocab_size, dim].
    // If we use our matmul, we expect [in_dim, out_dim].
    // So lm_head in file is [vocab_size, dim] (rows are tokens?).
    // Usually lm_head weights are [vocab_size, dim].
    // To project x [dim] to logits [vocab_size], we need W [dim, vocab_size].
    // The file has [vocab_size, dim] (transposed relative to what we need).
    // And export_binary.py writes `lm_head.weight.t()`.
    // PyTorch `lm_head` is `Linear(dim, vocab_size)`. Weights are `[vocab_size, dim]`.
    // `t()` makes it `[dim, vocab_size]`.
    // So file has `[dim, vocab_size]`.
    // This matches our matmul requirement: `matmul(out, in, w, dim, vocab_size)`.
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

    printf("Initializing...\n");
    load_model(&weights, &config, model_path);
    malloc_run_state(&state, &config);
    
    int token = 1; // BOS token (usually 1 for Llama)
    int pos = 0;
    
    printf("Starting inference for %d steps...\n", steps);
    
    // Simple generation loop
    // (We don't have a tokenizer in C yet, so we just print raw token IDs)
    
    // Timer
    clock_t start = clock();
    
    for (pos = 0; pos < steps; pos++) {
        transformer(token, pos, &config, &state, &weights);
        
        // Argmax to get next token
        // Copy logits to host to find max
        // This is slow but fine for Phase 1
        float* host_logits = (float*)malloc(config.vocab_size * sizeof(float));
        cudaCheck(cudaMemcpy(host_logits, state.logits, config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        int next_token = 0;
        float max_val = -INFINITY;
        for (int i = 0; i < config.vocab_size; i++) {
            if (host_logits[i] > max_val) {
                max_val = host_logits[i];
                next_token = i;
            }
        }
        free(host_logits);
        
        printf("%d ", next_token);
        fflush(stdout);
        
        token = next_token;
    }
    printf("\n");
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time: %.2fs, %.2f tok/s\n", time_spent, steps / time_spent);

    free_run_state(&state);
    free_ops();
    // free weights...
    
    return 0;
}

