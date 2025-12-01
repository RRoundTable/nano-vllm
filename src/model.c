#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "structs.h"
#include "backend.h"

// FP16 to FP32 conversion
// Reference: https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF) << 13;
    
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            result = sign;
        } else {
            // Denormalized number
            exponent = 1;
            while ((mantissa & 0x00800000) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= ~0x00800000;
            exponent = exponent + (127 - 15);
            result = sign | (exponent << 23) | mantissa;
        }
    } else if (exponent == 0x1F) {
        // Inf or NaN
        result = sign | 0x7F800000 | mantissa;
    } else {
        // Normalized number
        exponent = exponent + (127 - 15);
        result = sign | (exponent << 23) | mantissa;
    }
    
    return *(float*)&result;
}

// Helper to allocate and read a tensor from file to GPU
// Reads FP16 from file and converts to FP32
float* load_tensor(FILE* f, size_t size) {
    // Read FP16 data
    uint16_t* fp16_ptr = (uint16_t*)malloc(size * sizeof(uint16_t));
    if (!fp16_ptr) {
        fprintf(stderr, "Failed to allocate memory for FP16 tensor of size %zu\n", size);
        exit(1);
    }
    if (fread(fp16_ptr, sizeof(uint16_t), size, f) != size) {
        fprintf(stderr, "Failed to read FP16 tensor from file (expected %zu elements)\n", size);
        free(fp16_ptr);
        exit(1);
    }
    
    // Convert to FP32
    float* host_ptr = (float*)malloc(size * sizeof(float));
    if (!host_ptr) {
        fprintf(stderr, "Failed to allocate host memory for FP32 tensor of size %zu\n", size);
        free(fp16_ptr);
        exit(1);
    }
    
    for (size_t i = 0; i < size; i++) {
        host_ptr[i] = fp16_to_fp32(fp16_ptr[i]);
    }
    free(fp16_ptr);
    
    // Copy to device
    float* device_ptr;
    check_status(device_malloc((void**)&device_ptr, size * sizeof(float)));
    check_status(device_memcpy(device_ptr, host_ptr, size * sizeof(float), HOST_TO_DEVICE));
    
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

    check_status(device_malloc((void**)&s->x, dim * sizeof(float)));
    check_status(device_malloc((void**)&s->xb, dim * sizeof(float)));
    check_status(device_malloc((void**)&s->xb2, dim * sizeof(float)));
    check_status(device_malloc((void**)&s->hb, hidden_dim * sizeof(float)));
    check_status(device_malloc((void**)&s->hb2, hidden_dim * sizeof(float)));
    check_status(device_malloc((void**)&s->q, dim * sizeof(float)));
    check_status(device_malloc((void**)&s->k, dim * sizeof(float)));
    check_status(device_malloc((void**)&s->v, dim * sizeof(float)));
    check_status(device_malloc((void**)&s->att, n_heads * max_seq_len * sizeof(float))); // Should be enough for one step
    check_status(device_malloc((void**)&s->logits, vocab_size * sizeof(float)));
    
    // KV Cache
    size_t cache_size = (size_t)n_layers * max_seq_len * n_kv_heads * head_dim;
    check_status(device_malloc((void**)&s->key_cache, cache_size * sizeof(float)));
    check_status(device_malloc((void**)&s->value_cache, cache_size * sizeof(float)));
}

void free_run_state(RunState* s) {
    check_status(device_free(s->x));
    check_status(device_free(s->xb));
    check_status(device_free(s->xb2));
    check_status(device_free(s->hb));
    check_status(device_free(s->hb2));
    check_status(device_free(s->q));
    check_status(device_free(s->k));
    check_status(device_free(s->v));
    check_status(device_free(s->att));
    check_status(device_free(s->logits));
    check_status(device_free(s->key_cache));
    check_status(device_free(s->value_cache));
}
