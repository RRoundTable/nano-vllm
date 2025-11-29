#ifndef OPS_H
#define OPS_H

#include "structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Normalization
void rms_norm(float* out, float* in, float* weight, int size, float eps);

// Matrix Multiplication (using cuBLAS)
// out = in @ weight
// in: [in_dim], weight: [in_dim, out_dim] (stored as [in_dim * out_dim])
// out: [out_dim]
void matmul(float* out, float* in, float* weight, int in_dim, int out_dim);

// Utils to init/free library resources
void init_ops();
void free_ops();

// RoPE
void apply_rope(float* q, float* k, int pos, float theta, int head_dim, int n_heads, int n_kv_heads);

// Activation
void swiglu(float* hb, float* gate, float* up, int hidden_dim);

// Softmax
void inplace_softmax(float* x, int size);

// Element-wise add
void accum(float* a, float* b, int size);

// Attention
// q: [dim], k_cache: [layer_offset + ...], v_cache: ...
// att: [n_heads, max_seq_len] - buffer for scores
// out: [dim]
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
// k, v: [dim] (current step) -> copy to cache at pos
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

#ifdef __cplusplus
}
#endif

#endif // OPS_H

