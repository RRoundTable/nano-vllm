#include "nano_cuda.h"
#include "ops.h"
#include <math.h>

extern "C" void init_ops() {
}

extern "C" void free_ops() {
}

// ===========================================================================
// RMSNorm
// ===========================================================================

__global__ void rms_norm_kernel(float* out, float* in, float* weight, int size, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    for (int i = tid; i < size; i += blockDim.x) {
        sum += in[i] * in[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    float rms = rsqrtf(sdata[0] / size + eps);
    
    for (int i = tid; i < size; i += blockDim.x) {
        out[i] = in[i] * rms * weight[i];
    }
}

extern "C" void rms_norm(float* out, float* in, float* weight, int size, float eps) {
    int threads = 1024;
    rms_norm_kernel<<<1, threads, threads * sizeof(float)>>>(out, in, weight, size, eps);
    cudaCheck(cudaGetLastError());
}

// ===========================================================================
// MatMul (Pure CUDA)
// ===========================================================================

__global__ void matmul_kernel(float* out, float* in, float* weight, int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_dim) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += in[i] * weight[i * out_dim + idx];
        }
        out[idx] = sum;
    }
}

extern "C" void matmul(float* out, float* in, float* weight, int in_dim, int out_dim) {
    int threads = 256;
    int blocks = (out_dim + threads - 1) / threads;
    matmul_kernel<<<blocks, threads>>>(out, in, weight, in_dim, out_dim);
    cudaCheck(cudaGetLastError());
}

// ===========================================================================
// RoPE
// ===========================================================================

__global__ void rope_kernel(float* q, float* k, int pos, float theta, int head_dim, int n_heads, int n_kv_heads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = (n_heads * head_dim + n_kv_heads * head_dim) / 2;
    
    if (idx >= total_pairs) return;

    // Determine if this thread handles Q or K
    int q_pairs = (n_heads * head_dim) / 2;
    
    float* target;
    int local_idx;
    
    if (idx < q_pairs) {
        target = q;
        local_idx = idx;
    } else {
        target = k;
        local_idx = idx - q_pairs;
    }
    
    int i = local_idx * 2;
    int val_idx = i % head_dim;
    float freq = 1.0f / powf(theta, (float)val_idx / (float)head_dim);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    
    float v0 = target[i];
    float v1 = target[i+1];
    
    target[i]   = v0 * fcr - v1 * fci;
    target[i+1] = v0 * fci + v1 * fcr;
}

extern "C" void apply_rope(float* q, float* k, int pos, float theta, int head_dim, int n_heads, int n_kv_heads) {
    int total_pairs = (n_heads * head_dim + n_kv_heads * head_dim) / 2;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    rope_kernel<<<blocks, threads>>>(q, k, pos, theta, head_dim, n_heads, n_kv_heads);
    cudaCheck(cudaGetLastError());
}

// ===========================================================================
// SwiGLU
// ===========================================================================

__global__ void swiglu_kernel(float* hb, float* gate, float* up, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        // SiLU: x * sigmoid(x)
        float val = g / (1.0f + expf(-g));
        hb[idx] = val * up[idx];
    }
}

extern "C" void swiglu(float* hb, float* gate, float* up, int hidden_dim) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(hb, gate, up, hidden_dim);
    cudaCheck(cudaGetLastError());
}

// ===========================================================================
// Softmax
// ===========================================================================

__global__ void softmax_kernel(float* x, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    float max_val = -INFINITY;
    for (int i = tid; i < size; i += blockDim.x) {
        if (x[i] > max_val) max_val = x[i];
    }
    sdata[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    max_val = sdata[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        float v = expf(x[i] - max_val);
        x[i] = v;
        sum += v;
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float total_sum = sdata[0];
    
    for (int i = tid; i < size; i += blockDim.x) {
        x[i] /= total_sum;
    }
}

extern "C" void inplace_softmax(float* x, int size) {
    int threads = 1024;
    softmax_kernel<<<1, threads, threads * sizeof(float)>>>(x, size);
    cudaCheck(cudaGetLastError());
}

// ===========================================================================
// Accumulate
// ===========================================================================

__global__ void accum_kernel(float* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

extern "C" void accum(float* a, float* b, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    accum_kernel<<<blocks, threads>>>(a, b, size);
    cudaCheck(cudaGetLastError());
}
