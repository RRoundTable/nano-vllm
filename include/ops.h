#ifndef OPS_H
#define OPS_H

#include "structs.h"

// Global init/free for libraries (like cuBLAS)
void init_ops();
void free_ops();

// Operations
void rms_norm(float* o, float* x, float* weight, int size, float eps);
void matmul(float* out, float* in, float* weight, int in_dim, int out_dim); 
void apply_rope(float* q, float* k, int pos, float theta, int head_dim, int n_heads, int n_kv_heads);
void swiglu(float* hb, float* h_gate, float* h_up, int hidden_dim);
void softmax(float* x, int size);
void accum(float* a, float* b, int size);

// Attention specific
void update_kv_cache(float* key_cache, float* value_cache, float* k, float* v, 
                     int layer, int pos, int max_seq_len, int n_kv_heads, int head_dim);

void multi_head_attention(float* out, float* q, float* key_cache, float* value_cache, float* att,
                          int layer, int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim);

// Visualization
void visualize_attention(float* att, int n_heads, int pos, int max_seq_len);
// mode: 0 = Linear (Naive), 1 = Paged (Block-based)
void visualize_kv_cache_usage(int layer, int pos, int max_seq_len, int mode);

#endif
