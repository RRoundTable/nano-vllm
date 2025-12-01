#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "structs.h"
#include "backend.h"

extern int g_paged_mode;

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
float* load_tensor_fp16(FILE* f, size_t size) {
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

// Reads FP32 from file directly
float* load_tensor_fp32(FILE* f, size_t size) {
    // Read FP32 data directly
    float* host_ptr = (float*)malloc(size * sizeof(float));
    if (!host_ptr) {
        fprintf(stderr, "Failed to allocate host memory for FP32 tensor of size %zu\n", size);
        exit(1);
    }
    if (fread(host_ptr, sizeof(float), size, f) != size) {
        fprintf(stderr, "Failed to read FP32 tensor from file (expected %zu elements)\n", size);
        free(host_ptr);
        exit(1);
    }
    
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
    
    // Determine file format based on size
    // Save current position
    long header_size = ftell(f);
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, header_size, SEEK_SET); // Go back to weights
    
    long data_size = file_size - header_size;
    
    // Calculate expected sizes
    // Params count
    // Embed: vocab * dim
    // Layers: n_layers * (rms_att(dim) + wq(dim*n_heads*head_dim) + wk(...) + wv(...) + wo(...) + rms_ffn(dim) + w_gate(dim*hidden) + w_up + w_down)
    // Final: dim
    // Head: vocab * dim (if unshared)
    
    // Note: p->head_dim might not be set yet if we rely on calculating it. 
    // But header has 7 ints. 
    // Let's check if p->head_dim is valid or derived.
    // The file header has 7 ints. The 7th is max_seq_len?
    // struct: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, max_seq_len
    // src/structs.h has head_dim as 8th. It is NOT in the file.
    
    // Derive head_dim
    p->head_dim = p->dim / p->n_heads;
    // Default rope_theta for now (llama2.c legacy format doesn't have it)
    p->rope_theta = 10000.0f; 

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

    long long num_weights = 0;
    num_weights += (long long)p->vocab_size * p->dim; // embed
    num_weights += (long long)p->n_layers * (
        p->dim + // rms_att
        (long long)p->dim * p->n_heads * p->head_dim + // wq
        (long long)p->dim * p->n_kv_heads * p->head_dim + // wk
        (long long)p->dim * p->n_kv_heads * p->head_dim + // wv
        (long long)p->n_heads * p->head_dim * p->dim + // wo
        p->dim + // rms_ffn
        (long long)p->dim * p->hidden_dim + // w_gate
        (long long)p->dim * p->hidden_dim + // w_up
        (long long)p->hidden_dim * p->dim   // w_down
    );
    num_weights += p->dim; // final rms
    
    long long num_weights_head = (long long)p->vocab_size * p->dim;
    
    // RoPE tables (legacy llama2.c format includes these at the end)
    // 2 tables of shape (seq_len, head_dim/2) complex or float?
    // run.c: ptr += p->seq_len * head_size / 2; (real) + (imag)
    // Total floats: seq_len * head_dim
    long long rope_floats = (long long)p->max_seq_len * p->head_dim;
    long long rope_bytes = rope_floats * 4;

    // Check scenarios
    int is_fp32 = 0;
    int is_shared = 0;
    
    long long size_fp32_unshared = (num_weights + num_weights_head) * 4;
    long long size_fp32_shared   = num_weights * 4;
    long long size_fp32_shared_rope = size_fp32_shared + rope_bytes; // Legacy format often has this
    
    long long size_fp16_unshared = (num_weights + num_weights_head) * 2;
    long long size_fp16_shared   = num_weights * 2;
    
    printf("File data size: %ld bytes\n", data_size);
    printf("Expected FP32 Shared (Raw): %lld\n", size_fp32_shared);
    printf("Expected FP32 Shared (+RoPE): %lld\n", size_fp32_shared_rope);
    printf("Expected FP16 Unshared: %lld\n", size_fp16_unshared);
    
    if (data_size == size_fp32_shared || data_size == size_fp32_shared_rope) {
        is_fp32 = 1;
        is_shared = 1;
        printf("Detected format: FP32 Shared Weights (Legacy%s)\n", 
               data_size == size_fp32_shared_rope ? " + RoPE" : "");
    } else if (data_size == size_fp32_unshared) {
        is_fp32 = 1;
        is_shared = 0;
        printf("Detected format: FP32 Unshared Weights\n");
    } else if (data_size == size_fp16_unshared) {
        is_fp32 = 0;
        is_shared = 0;
        printf("Detected format: FP16 Unshared Weights (Nano Export)\n");
    } else if (data_size == size_fp16_shared) {
        is_fp32 = 0;
        is_shared = 1;
        printf("Detected format: FP16 Shared Weights\n");
    } else {
        printf("Warning: File size matches none of the expected formats! Trying FP16 Unshared as fallback...\n");
        is_fp32 = 0;
        is_shared = 0;
    }

    // Allocate layers struct
    w->layers = (LayerWeights*)malloc(p->n_layers * sizeof(LayerWeights));

    // Read weights
    printf("Loading weights...\n");
    
    float* (*load_func)(FILE*, size_t) = is_fp32 ? load_tensor_fp32 : load_tensor_fp16;
    
    w->token_embedding_table = load_func(f, (size_t)p->vocab_size * p->dim);
    
    // Llama 2 weights are grouped by parameter type, not by layer
    
    // rms_att_weight
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].rms_att_weight = load_func(f, p->dim);
    }
    // wq
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].wq = load_func(f, (size_t)p->dim * p->n_heads * p->head_dim);
    }
    // wk
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].wk = load_func(f, (size_t)p->dim * p->n_kv_heads * p->head_dim);
    }
    // wv
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].wv = load_func(f, (size_t)p->dim * p->n_kv_heads * p->head_dim);
    }
    // wo
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].wo = load_func(f, (size_t)p->n_heads * p->head_dim * p->dim);
    }
    // rms_ffn_weight
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].rms_ffn_weight = load_func(f, p->dim);
    }
    // w_gate
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].w_gate = load_func(f, (size_t)p->dim * p->hidden_dim);
    }
    // w_down
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].w_down = load_func(f, (size_t)p->hidden_dim * p->dim);
    }
    // w_up
    for(int i=0; i<p->n_layers; i++) {
        w->layers[i].w_up = load_func(f, (size_t)p->dim * p->hidden_dim);
    }
    
    w->rms_final_weight = load_func(f, p->dim);
    
    if (is_shared) {
        // Share pointer
        w->lm_head = w->token_embedding_table;
        printf("Weights shared: lm_head -> token_embedding_table\n");
    } else {
        w->lm_head = load_func(f, (size_t)p->vocab_size * p->dim);
    }
    
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
    if (!g_paged_mode) {
        size_t cache_size = (size_t)n_layers * max_seq_len * n_kv_heads * head_dim;
        check_status(device_malloc((void**)&s->key_cache, cache_size * sizeof(float)));
        check_status(device_malloc((void**)&s->value_cache, cache_size * sizeof(float)));
    } else {
        s->key_cache = NULL;
        s->value_cache = NULL;
    }
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
    if (s->key_cache) check_status(device_free(s->key_cache));
    if (s->value_cache) check_status(device_free(s->value_cache));
}
