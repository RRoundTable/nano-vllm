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

void rms_norm(float* out, float* in, float* weight, int size, int num_tokens, float eps) {
    for (int t = 0; t < num_tokens; t++) {
        float* in_ptr = in + t * size;
        float* out_ptr = out + t * size;
        
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += in_ptr[i] * in_ptr[i];
        }
        float rms = 1.0f / sqrtf(sum / size + eps);
        
        for (int i = 0; i < size; i++) {
            out_ptr[i] = in_ptr[i] * rms * weight[i];
        }
    }
}

// ===========================================================================
// MatMul (Naive CPU)
// ===========================================================================

void matmul(float* out, float* in, float* weight, int num_tokens, int in_dim, int out_dim) {
    // out = in @ weight^T
    // in: [num_tokens, in_dim]
    // weight: [out_dim, in_dim] (row-major)
    // out: [num_tokens, out_dim]
    
    // Matrix-Matrix Multiplication
    #if defined(_OPENMP)
    #pragma omp parallel for collapse(2)
    #endif
    for (int t = 0; t < num_tokens; t++) {
        for (int i = 0; i < out_dim; i++) {
            float val = 0.0f;
            float* in_vec = in + t * in_dim;
            float* w_vec = weight + i * in_dim;
            
            // Vector dot product
            for (int j = 0; j < in_dim; j++) {
                val += in_vec[j] * w_vec[j];
            }
            out[t * out_dim + i] = val;
        }
    }
}

// ===========================================================================
// RoPE
// ===========================================================================

void apply_rope(float* q, float* k, int pos, float theta, int head_dim, int num_tokens, int n_heads, int n_kv_heads) {
    int dim_q = n_heads * head_dim;
    int dim_k = n_kv_heads * head_dim;

    for (int t = 0; t < num_tokens; t++) {
        int current_pos = pos + t;
        float* q_ptr = q + t * dim_q;
        float* k_ptr = k + t * dim_k;

        for (int i = 0; i < head_dim; i+=2) {
            float freq = 1.0f / powf(theta, i / (float)head_dim);
            float val = current_pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            
            // Apply to Query
            for (int h = 0; h < n_heads; h++) {
                float* vec = q_ptr + h * head_dim;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
            
            // Apply to Key
            for (int h = 0; h < n_kv_heads; h++) {
                float* vec = k_ptr + h * head_dim;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
    }
}

void apply_rope_batch(float* q, float* k, int* pos_arr, float theta, int head_dim, int num_tokens, int n_heads, int n_kv_heads) {
    int dim_q = n_heads * head_dim;
    int dim_k = n_kv_heads * head_dim;

    for (int t = 0; t < num_tokens; t++) {
        int current_pos = pos_arr[t];
        float* q_ptr = q + t * dim_q;
        float* k_ptr = k + t * dim_k;

        for (int i = 0; i < head_dim; i+=2) {
            float freq = 1.0f / powf(theta, i / (float)head_dim);
            float val = current_pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            
            // Apply to Query
            for (int h = 0; h < n_heads; h++) {
                float* vec = q_ptr + h * head_dim;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
            
            // Apply to Key
            for (int h = 0; h < n_kv_heads; h++) {
                float* vec = k_ptr + h * head_dim;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
    }
}

// ===========================================================================
// Activation / Element-wise
// ===========================================================================

void swiglu(float* hb, float* gate, float* up, int hidden_dim, int num_tokens) {
    for (int t = 0; t < num_tokens; t++) {
        float* hb_ptr = hb + t * hidden_dim;
        float* gate_ptr = gate + t * hidden_dim;
        float* up_ptr = up + t * hidden_dim;

        for (int i = 0; i < hidden_dim; i++) {
            float g = gate_ptr[i];
            float val = g / (1.0f + expf(-g));
            hb_ptr[i] = val * up_ptr[i];
        }
    }
}

void accum(float* a, float* b, int size) {
    // Note: accum expects size to include num_tokens * dim if batched
    for (int i = 0; i < size; i++) {
        a[i] += b[i];
    }
}

void softmax(float* x, int size) {
    // Naive softmax for single vector
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
