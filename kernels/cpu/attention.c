#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===========================================================================
// KV Cache (Naive)
// ===========================================================================

void update_kv_cache(float* k_cache, float* v_cache, float* k, float* v, 
                                int layer, int pos, int max_seq_len, int n_kv_heads, int head_dim) {
    
    long layer_offset = (long)layer * max_seq_len * n_kv_heads * head_dim;
    long pos_offset = (long)pos * n_kv_heads * head_dim;
    long base_offset = layer_offset + pos_offset;
    
    int size = n_kv_heads * head_dim;
    memcpy(k_cache + base_offset, k, size * sizeof(float));
    memcpy(v_cache + base_offset, v, size * sizeof(float));
}

// ===========================================================================
// Attention (Naive)
// ===========================================================================

void multi_head_attention(float* out, float* q, float* key_cache, float* value_cache, float* att, 
                                     int layer, int pos, int max_seq_len, 
                                     int n_heads, int n_kv_heads, int head_dim) {
    
    float scale = 1.0f / sqrtf(head_dim);
    long layer_offset = (long)layer * max_seq_len * n_kv_heads * head_dim;
    
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / (n_heads / n_kv_heads); // GQA
        float* q_head = q + h * head_dim;
        float* att_head = att + h * max_seq_len;
        
        // 1. Score: Q @ K.T
        for (int t = 0; t <= pos; t++) {
            float* k_head = key_cache + layer_offset + (long)t * n_kv_heads * head_dim + kv_h * head_dim;
            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }
            score *= scale;
            att_head[t] = score;
        }
        
        // 2. Softmax
        float max_val = -INFINITY;
        for (int t = 0; t <= pos; t++) {
            if (att_head[t] > max_val) max_val = att_head[t];
        }
        
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float val = expf(att_head[t] - max_val);
            att_head[t] = val;
            sum += val;
        }
        
        for (int t = 0; t <= pos; t++) {
            att_head[t] /= sum;
        }
        
        // 3. Weighted Sum: Att @ V
        float* out_head = out + h * head_dim;
        // Init out_head to 0
        for(int i=0; i<head_dim; i++) out_head[i] = 0.0f;
        
        for (int t = 0; t <= pos; t++) {
            float prob = att_head[t];
            float* v_head = value_cache + layer_offset + (long)t * n_kv_heads * head_dim + kv_h * head_dim;
            for (int i = 0; i < head_dim; i++) {
                out_head[i] += prob * v_head[i];
            }
        }
    }
}

// ===========================================================================
// PagedAttention Kernels
// ===========================================================================

void update_kv_cache_paged(KVCacheManager* mgr, BlockTable* block_table, float* k, float* v, 
                           int layer, int pos, int n_kv_heads, int head_dim) {
    // Find physical location
    int block_size = mgr->block_size;
    int logical_block_idx = pos / block_size;
    int block_offset = pos % block_size;
    
    // Get physical block index from table
    if (logical_block_idx >= block_table->num_blocks) {
        printf("Error: Block table too small! logical_idx=%d, num_blocks=%d\n", logical_block_idx, block_table->num_blocks);
        return;
    }
    int physical_block_idx = block_table->block_indices[logical_block_idx];
    
    // Calculate physical offset
    long offset = get_physical_offset(mgr, layer, physical_block_idx, block_offset, n_kv_heads, head_dim);
    
    // Write to pool
    int size = n_kv_heads * head_dim;
    memcpy(mgr->pool_k + offset, k, size * sizeof(float));
    memcpy(mgr->pool_v + offset, v, size * sizeof(float));
}

void paged_attention(float* out, float* q, KVCacheManager* mgr, BlockTable* block_table, float* att,
                     int layer, int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim) {
    
    float scale = 1.0f / sqrtf(head_dim);
    int block_size = mgr->block_size; // the number of tokens in a block
    
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / (n_heads / n_kv_heads); // GQA
        float* q_head = q + h * head_dim;
        float* att_head = att + h * max_seq_len;
        
        // 1. Score: Q @ K.T (Iterate over all previous tokens)
        for (int t = 0; t <= pos; t++) {
            // Resolve Physical Address for Token t
            int logical_block = t / block_size;
            int block_offset = t % block_size;
            int physical_block = block_table->block_indices[logical_block];
            
            long offset = get_physical_offset(mgr, layer, physical_block, block_offset, n_kv_heads, head_dim);
            
            // Calculate k_head pointer address:
            // 1. mgr->pool_k: Base address of the entire Key cache memory pool
            // 2. offset: Distance to the specific [Layer, Physical Block, Token Offset]
            // 3. kv_h * head_dim: Start position of the specific KV Head vector within that token
            float* k_head = mgr->pool_k + offset + kv_h * head_dim;
            
            float score = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                score += q_head[i] * k_head[i];
            }
            score *= scale;
            att_head[t] = score;
        }
        
        // 2. Softmax
        float max_val = -INFINITY;
        for (int t = 0; t <= pos; t++) {
            if (att_head[t] > max_val) max_val = att_head[t];
        }
        
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float val = expf(att_head[t] - max_val);
            att_head[t] = val;
            sum += val;
        }
        
        for (int t = 0; t <= pos; t++) {
            att_head[t] /= sum;
        }
        
        // 3. Weighted Sum: Att @ V
        float* out_head = out + h * head_dim;
        for(int i=0; i<head_dim; i++) out_head[i] = 0.0f;
        
        for (int t = 0; t <= pos; t++) {
            float prob = att_head[t];
            
            // Resolve Physical Address again
            int logical_block = t / block_size;
            int block_offset = t % block_size;
            int physical_block = block_table->block_indices[logical_block];
            
            long offset = get_physical_offset(mgr, layer, physical_block, block_offset, n_kv_heads, head_dim);
            float* v_head = mgr->pool_v + offset + kv_h * head_dim;
            
            for (int i = 0; i < head_dim; i++) {
                out_head[i] += prob * v_head[i];
            }
        }
    }
}

