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
    // in: [in_dim], weight: [out_dim, in_dim] (row-major)
    // out: [out_dim]
    
    // Parallelize outer loop (output dimension)
    #if defined(_OPENMP)
    #pragma omp parallel for
    #endif
    for (int i = 0; i < out_dim; i++) {
        float val = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            val += in[j] * weight[i * in_dim + j];
        }
        out[i] = val;
    }
}

// ===========================================================================
// RoPE
// ===========================================================================

// Apply Rotary Positional Embedding (RoPE)
// Formula:
// 1. Calculate frequency (freq): theta_i = 1.0 / (theta ^ (i / head_dim))
// 2. Calculate rotation angle (val): angle = pos * theta_i
// 3. Apply rotation matrix:
//    [ v_i'   ]   [ cos(angle)  -sin(angle) ] [ v_i   ]
//    [ v_i+1' ] = [ sin(angle)   cos(angle) ] [ v_i+1 ]
//
//    v_i'   = v_i * cos(angle) - v_i+1 * sin(angle)
//    v_i+1' = v_i * sin(angle) + v_i+1 * cos(angle)
void apply_rope(float* q, float* k, int pos, float theta, int head_dim, int n_heads, int n_kv_heads) {
    for (int i = 0; i < head_dim; i+=2) {
        // Calculate frequency and angle
        // freq = 1 / theta^(i/d)
        float freq = 1.0f / powf(theta, i / (float)head_dim);
        // val = pos * freq (rotation angle)
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        
        // Query 벡터에 RoPE 적용
        for (int h = 0; h < n_heads; h++) {
            float* vec = q + h * head_dim;
            float v0 = vec[i];
            float v1 = vec[i+1];
            // Apply rotation
            vec[i]   = v0 * fcr - v1 * fci; // real part calculation
            vec[i+1] = v0 * fci + v1 * fcr; // imaginary part calculation
        }
        
        // Key 벡터에 RoPE 적용
        for (int h = 0; h < n_kv_heads; h++) {
            float* vec = k + h * head_dim;
            float v0 = vec[i];
            float v1 = vec[i+1];
            // Apply rotation
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }
}

// ===========================================================================
// Activation / Element-wise
// ===========================================================================

// Apply SwiGLU (Swish-Gated Linear Unit) activation
// Formula: output = SiLU(gate) * up
// where SiLU(x) = x * sigmoid(x) = x / (1 + e^-x)
void swiglu(float* hb, float* gate, float* up, int hidden_dim) {
    for (int i = 0; i < hidden_dim; i++) {
        float g = gate[i];
        // Calculate SiLU activation: val = g * sigmoid(g)
        float val = g / (1.0f + expf(-g));
        // Element-wise multiplication
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
