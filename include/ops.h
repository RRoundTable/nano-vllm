#ifndef OPS_H
#define OPS_H

#include "structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialization
void init_ops();
void free_ops();

// Normalization
void rms_norm(float* out, float* in, float* weight, int size, float eps);

// Matrix Multiplication
// out = in @ weight
// in: [in_dim], weight: [in_dim, out_dim] (row-major)
// out: [out_dim]
void matmul(float* out, float* in, float* weight, int in_dim, int out_dim);

// RoPE
void apply_rope(float* q, float* k, int pos, float theta, int head_dim, int n_heads, int n_kv_heads);

// Activation
void swiglu(float* hb, float* gate, float* up, int hidden_dim);

// Softmax
void inplace_softmax(float* x, int size);

// Element-wise add
void accum(float* a, float* b, int size);

// Attention
void multi_head_attention(
    float* out, 
    float* q, 
    float* k_cache, 
    float* v_cache, 
    float* att, 
    int layer, 
    int pos, 
    int max_seq_len, 
    int n_heads, 
    int n_kv_heads, 
    int head_dim
);

// KV Cache Update
void update_kv_cache(
    float* k_cache, 
    float* v_cache, 
    float* k, 
    float* v, 
    int layer, 
    int pos, 
    int max_seq_len, 
    int n_kv_heads, 
    int head_dim
);

// Visualizer (CPU Only, but safe to declare)
void visualize_attention(float* att, int n_heads, int pos, int max_seq_len);
void visualize_kv_cache_usage(int layer, int pos, int max_seq_len);

#ifdef __cplusplus
}
#endif

#endif // OPS_H
