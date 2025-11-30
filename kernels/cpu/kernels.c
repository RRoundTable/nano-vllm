#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===========================================================================
// Initialization
// ===========================================================================

void init_ops() {
    // Nothing to initialize for CPU
}

void free_ops() {
    // Nothing to free for CPU
}

// ===========================================================================
// Normalization
// ===========================================================================

void rms_norm(float* out, float* in, float* weight, int size, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += in[i] * in[i];
    }
    float rms = 1.0f / sqrtf(sum / size + eps);
    
    for (int i = 0; i < size; i++) {
        out[i] = in[i] * rms * weight[i];
    }
}

// ===========================================================================
// MatMul (Naive CPU)
// ===========================================================================

void matmul(float* out, float* in, float* weight, int in_dim, int out_dim) {
    // out = in @ weight
    // in: [in_dim], weight: [in_dim, out_dim] (row-major)
    // out: [out_dim]
    
    // Parallelize outer loop (output dimension)
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int i = 0; i < out_dim; i++) {
        float val = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            val += in[j] * weight[j * out_dim + i];
        }
        out[i] = val;
    }
}

// ===========================================================================
// RoPE
// ===========================================================================

void apply_rope(float* q, float* k, int pos, float theta, int head_dim, int n_heads, int n_kv_heads) {
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / head_dim);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            
            int q_idx = h * head_dim + i;
            float q0 = q[q_idx];
            float q1 = q[q_idx+1];
            q[q_idx]   = q0 * fcr - q1 * fci;
            q[q_idx+1] = q0 * fci + q1 * fcr;
        }
    }
    
    for (int h = 0; h < n_kv_heads; h++) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / head_dim);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            
            int k_idx = h * head_dim + i;
            float k0 = k[k_idx];
            float k1 = k[k_idx+1];
            k[k_idx]   = k0 * fcr - k1 * fci;
            k[k_idx+1] = k0 * fci + k1 * fcr;
        }
    }
}

// ===========================================================================
// Activation / Element-wise
// ===========================================================================

void swiglu(float* hb, float* gate, float* up, int hidden_dim) {
    for (int i = 0; i < hidden_dim; i++) {
        float g = gate[i];
        float val = g / (1.0f + expf(-g)); // SiLU
        hb[i] = val * up[i];
    }
}

void accum(float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void inplace_softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

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
    int block_size = mgr->block_size;
    
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
