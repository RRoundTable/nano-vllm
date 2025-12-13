#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===========================================================================
// KV Cache (Naive)
// ===========================================================================

void update_kv_cache(float* k_cache, float* v_cache, float* k, float* v, 
                                int layer, int pos, int max_seq_len, int n_kv_heads, int head_dim, int num_tokens) {
    
    int head_block_size = n_kv_heads * head_dim;
    
    // Iterate over the batch of tokens
    for (int t = 0; t < num_tokens; t++) {
        int current_pos = pos + t;
        long layer_offset = (long)layer * max_seq_len * head_block_size;
        long pos_offset = (long)current_pos * head_block_size;
        long base_offset = layer_offset + pos_offset;
        
        // Copy K and V for this token
        float* k_src = k + t * head_block_size;
        float* v_src = v + t * head_block_size;
        
        memcpy(k_cache + base_offset, k_src, head_block_size * sizeof(float));
        memcpy(v_cache + base_offset, v_src, head_block_size * sizeof(float));
    }
}

// ===========================================================================
// Attention (Naive)
// ===========================================================================

void multi_head_attention(float* out, float* q, float* key_cache, float* value_cache, float* att, 
                                     int layer, int pos, int max_seq_len, 
                                     int n_heads, int n_kv_heads, int head_dim, int num_tokens) {
    
    float scale = 1.0f / sqrtf(head_dim);
    long layer_offset = (long)layer * max_seq_len * n_kv_heads * head_dim;
    int q_dim = n_heads * head_dim;
    
    // Loop over each token in the batch (chunk)
    for (int curr_t = 0; curr_t < num_tokens; curr_t++) {
        int current_global_pos = pos + curr_t;
        
        // Pointers for current token's Query and Output
        float* q_token = q + curr_t * q_dim;
        float* out_token = out + curr_t * q_dim;

        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / (n_heads / n_kv_heads); // GQA
            float* q_head = q_token + h * head_dim;
            float* att_head = att + h * max_seq_len; // Reuse att buffer per token/head
            
            // 1. Score: Q @ K.T
            // Causal Masking: Attend to [0 ... current_global_pos]
            for (int t = 0; t <= current_global_pos; t++) {
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
            for (int t = 0; t <= current_global_pos; t++) {
                if (att_head[t] > max_val) max_val = att_head[t];
            }
            
            float sum = 0.0f;
            for (int t = 0; t <= current_global_pos; t++) {
                float val = expf(att_head[t] - max_val);
                att_head[t] = val;
                sum += val;
            }
            
            for (int t = 0; t <= current_global_pos; t++) {
                att_head[t] /= sum;
            }
            
            // 3. Weighted Sum: Att @ V
            float* out_head = out_token + h * head_dim;
            // Init out_head to 0
            for(int i=0; i<head_dim; i++) out_head[i] = 0.0f;
            
            for (int t = 0; t <= current_global_pos; t++) {
                float prob = att_head[t];
                float* v_head = value_cache + layer_offset + (long)t * n_kv_heads * head_dim + kv_h * head_dim;
                for (int i = 0; i < head_dim; i++) {
                    out_head[i] += prob * v_head[i];
                }
            }
        }
    }
}

// ===========================================================================
// PagedAttention Kernels
// ===========================================================================

void update_kv_cache_paged(KVCacheManager* mgr, BlockTable* block_table, float* k, float* v, 
                           int layer, int pos, int n_kv_heads, int head_dim, int num_tokens) {
    
    int head_block_size = n_kv_heads * head_dim;

    for (int t = 0; t < num_tokens; t++) {
        int current_pos = pos + t;
        
        // Find physical location
        int block_size = mgr->block_size;
        int logical_block_idx = current_pos / block_size;
        int block_offset = current_pos % block_size;
        
        // Get physical block index from table
        if (logical_block_idx >= block_table->num_blocks) {
            printf("Error: Block table too small! logical_idx=%d, num_blocks=%d\n", logical_block_idx, block_table->num_blocks);
            return;
        }
        int physical_block_idx = block_table->block_indices[logical_block_idx];
        
        // Calculate physical offset
        long offset = get_physical_offset(mgr, layer, physical_block_idx, block_offset, n_kv_heads, head_dim);
        
        // Write to pool
        float* k_src = k + t * head_block_size;
        float* v_src = v + t * head_block_size;

        memcpy(mgr->pool_k + offset, k_src, head_block_size * sizeof(float));
        memcpy(mgr->pool_v + offset, v_src, head_block_size * sizeof(float));
    }
}

void paged_attention(float* out, float* q, KVCacheManager* mgr, BlockTable* block_table, float* att,
                     int layer, int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim, int num_tokens) {
    
    float scale = 1.0f / sqrtf(head_dim);
    int block_size = mgr->block_size; 
    int q_dim = n_heads * head_dim;

    for (int curr_t = 0; curr_t < num_tokens; curr_t++) {
        int current_global_pos = pos + curr_t;
        float* q_token = q + curr_t * q_dim;
        float* out_token = out + curr_t * q_dim;

        #if defined(_OPENMP)
        #pragma omp parallel for
        #endif
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / (n_heads / n_kv_heads); // GQA
            float* q_head = q_token + h * head_dim;
            float* att_head = att + h * max_seq_len;
            
            // 1. Score: Q @ K.T (Iterate over all previous tokens)
            for (int t = 0; t <= current_global_pos; t++) {
                // Resolve Physical Address for Token t
                int logical_block = t / block_size;
                int block_offset = t % block_size;
                int physical_block = block_table->block_indices[logical_block];
                
                long offset = get_physical_offset(mgr, layer, physical_block, block_offset, n_kv_heads, head_dim);
                
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
            for (int t = 0; t <= current_global_pos; t++) {
                if (att_head[t] > max_val) max_val = att_head[t];
            }
            
            float sum = 0.0f;
            for (int t = 0; t <= current_global_pos; t++) {
                float val = expf(att_head[t] - max_val);
                att_head[t] = val;
                sum += val;
            }
            
            for (int t = 0; t <= current_global_pos; t++) {
                att_head[t] /= sum;
            }
            
            // 3. Weighted Sum: Att @ V
            float* out_head = out_token + h * head_dim;
            for(int i=0; i<head_dim; i++) out_head[i] = 0.0f;
            
            for (int t = 0; t <= current_global_pos; t++) {
                float prob = att_head[t];
                
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
}
