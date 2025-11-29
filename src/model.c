#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structs.h"
#include "nano_cuda.h"

// Helper to allocate and read a tensor from file to GPU
float* load_tensor(FILE* f, size_t size) {
    float* host_ptr = (float*)malloc(size * sizeof(float));
    if (!host_ptr) {
        fprintf(stderr, "Failed to allocate host memory for tensor of size %zu\n", size);
        exit(1);
    }
    if (fread(host_ptr, sizeof(float), size, f) != size) {
        fprintf(stderr, "Failed to read tensor from file\n");
        exit(1);
    }
    
    float* device_ptr;
    cudaCheck(cudaMalloc((void**)&device_ptr, size * sizeof(float)));
    cudaCheck(cudaMemcpy(device_ptr, host_ptr, size * sizeof(float), cudaMemcpyHostToDevice));
    
    free(host_ptr);
    return device_ptr;
}

void load_model(Weights* w, Config* p, const char* checkpoint_path) {
    FILE* f = fopen(checkpoint_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open model file: %s\n", checkpoint_path);
        exit(1);
    }

    // Read header
    if (fread(p, sizeof(int), 7, f) != 7) { exit(1); }
    // Read head_dim (int) and rope_theta (float)
    if (fread(&p->head_dim, sizeof(int), 1, f) != 1) { exit(1); }
    if (fread(&p->rope_theta, sizeof(float), 1, f) != 1) { exit(1); }

    printf("Model Config:\n");
    printf("  dim: %d\n", p->dim);
    printf("  n_layers: %d\n", p->n_layers);
    printf("  n_heads: %d\n", p->n_heads);
    printf("  n_kv_heads: %d\n", p->n_kv_heads);
    printf("  vocab_size: %d\n", p->vocab_size);
    printf("  max_seq_len: %d\n", p->max_seq_len);
    printf("  hidden_dim: %d\n", p->hidden_dim);
    printf("  head_dim: %d\n", p->head_dim);
    printf("  rope_theta: %f\n", p->rope_theta);

    // Allocate layers struct
    w->layers = (LayerWeights*)malloc(p->n_layers * sizeof(LayerWeights));

    // Read weights
    printf("Loading weights...\n");
    
    w->token_embedding_table = load_tensor(f, (size_t)p->vocab_size * p->dim);
    
    for(int i=0; i<p->n_layers; i++) {
        LayerWeights* l = &w->layers[i];
        l->rms_att_weight = load_tensor(f, p->dim);
        l->wq = load_tensor(f, (size_t)p->dim * p->n_heads * p->head_dim);
        l->wk = load_tensor(f, (size_t)p->dim * p->n_kv_heads * p->head_dim);
        l->wv = load_tensor(f, (size_t)p->dim * p->n_kv_heads * p->head_dim);
        l->wo = load_tensor(f, (size_t)p->n_heads * p->head_dim * p->dim);
        
        l->rms_ffn_weight = load_tensor(f, p->dim);
        l->w_gate = load_tensor(f, (size_t)p->dim * p->hidden_dim);
        l->w_up = load_tensor(f, (size_t)p->dim * p->hidden_dim);
        l->w_down = load_tensor(f, (size_t)p->hidden_dim * p->dim);
    }
    
    w->rms_final_weight = load_tensor(f, p->dim);
    w->lm_head = load_tensor(f, (size_t)p->vocab_size * p->dim);
    
    fclose(f);
    printf("Model loaded.\n");
}

void malloc_run_state(RunState* s, Config* p) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int vocab_size = p->vocab_size;
    int max_seq_len = p->max_seq_len;
    int n_layers = p->n_layers;
    int n_kv_heads = p->n_kv_heads;
    int head_dim = p->head_dim;
    int n_heads = p->n_heads;

    cudaCheck(cudaMalloc((void**)&s->x, dim * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->xb, dim * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->xb2, dim * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->hb, hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->hb2, hidden_dim * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->q, dim * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->k, dim * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->v, dim * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->att, n_heads * max_seq_len * sizeof(float))); // Should be enough for one step
    cudaCheck(cudaMalloc((void**)&s->logits, vocab_size * sizeof(float)));
    
    // KV Cache
    size_t cache_size = (size_t)n_layers * max_seq_len * n_kv_heads * head_dim;
    cudaCheck(cudaMalloc((void**)&s->key_cache, cache_size * sizeof(float)));
    cudaCheck(cudaMalloc((void**)&s->value_cache, cache_size * sizeof(float)));
}

void free_run_state(RunState* s) {
    cudaCheck(cudaFree(s->x));
    cudaCheck(cudaFree(s->xb));
    cudaCheck(cudaFree(s->xb2));
    cudaCheck(cudaFree(s->hb));
    cudaCheck(cudaFree(s->hb2));
    cudaCheck(cudaFree(s->q));
    cudaCheck(cudaFree(s->k));
    cudaCheck(cudaFree(s->v));
    cudaCheck(cudaFree(s->att));
    cudaCheck(cudaFree(s->logits));
    cudaCheck(cudaFree(s->key_cache));
    cudaCheck(cudaFree(s->value_cache));
}

