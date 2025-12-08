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
