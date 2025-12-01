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

// Attention specific (Naive)
void update_kv_cache(float* key_cache, float* value_cache, float* k, float* v, 
                     int layer, int pos, int max_seq_len, int n_kv_heads, int head_dim);

void multi_head_attention(float* out, float* q, float* key_cache, float* value_cache, float* att,
                          int layer, int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim);

// ==========================================
// PagedAttention Logic (Phase 3)
// ==========================================

// Memory Manager
void init_kv_cache_manager(KVCacheManager* mgr, int block_size, int num_blocks, int n_layers, int n_kv_heads, int head_dim);
void free_kv_cache_manager(KVCacheManager* mgr);
int alloc_block(KVCacheManager* mgr);
void free_block(KVCacheManager* mgr, int block_idx);
long get_physical_offset(KVCacheManager* mgr, int layer, int block_idx, int block_offset, int n_kv_heads, int head_dim);

// Paged Kernels
void update_kv_cache_paged(KVCacheManager* mgr, BlockTable* block_table, float* k, float* v, 
                           int layer, int pos, int n_kv_heads, int head_dim);

void paged_attention(float* out, float* q, KVCacheManager* mgr, BlockTable* block_table, float* att,
                     int layer, int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim);

// Visualization
void visualize_attention(float* att, int n_heads, int pos, int max_seq_len);
// mode: 0 = Linear (Naive), 1 = Paged (Block-based)
void visualize_kv_cache_usage(Sequence* seqs, int num_seqs, KVCacheManager* mgr, int max_seq_len, int mode);

#endif
