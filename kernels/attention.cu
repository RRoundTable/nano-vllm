#include "nano_cuda.h"
#include "ops.h"
#include <math.h>

// ===========================================================================
// KV Cache Update
// ===========================================================================

__global__ void update_kv_kernel(float* k_cache, float* v_cache, float* k, float* v, 
                                 size_t layer_offset, int n_kv_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = n_kv_heads * head_dim;
    
    if (idx < size) {
        // k_cache is flattened: [n_layers, max_seq_len, n_kv_heads, head_dim]
        // layer_offset points to the start of the "timestep slot" in the cache
        k_cache[layer_offset + idx] = k[idx];
        v_cache[layer_offset + idx] = v[idx];
    }
}

extern "C" void update_kv_cache(float* k_cache, float* v_cache, float* k, float* v, 
                                int layer, int pos, int max_seq_len, int n_kv_heads, int head_dim) {
    // Calculate offset for this specific position
    // layer_stride = max_seq_len * n_kv_heads * head_dim
    // pos_stride = n_kv_heads * head_dim
    size_t layer_stride = (size_t)max_seq_len * n_kv_heads * head_dim;
    size_t pos_stride = (size_t)n_kv_heads * head_dim;
    size_t offset = layer * layer_stride + pos * pos_stride;
    
    int size = n_kv_heads * head_dim;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    update_kv_kernel<<<blocks, threads>>>(k_cache, v_cache, k, v, offset, n_kv_heads, head_dim);
    cudaCheck(cudaGetLastError());
}

// ===========================================================================
// Naive Attention Kernel
// ===========================================================================

__global__ void attention_kernel(float* out, float* q, float* k_cache, float* v_cache, float* att,
                                 size_t layer_base_offset, int pos, int max_seq_len, 
                                 int n_heads, int n_kv_heads, int head_dim, float scale) {
    
    // One block per head
    int h = blockIdx.x;
    if (h >= n_heads) return;
    
    int tid = threadIdx.x;
    int kv_h = h / (n_heads / n_kv_heads); // GQA Mapping
    
    // Offsets
    int q_offset = h * head_dim;
    
    // Shared memory configuration:
    // [0..head_dim]: shared_q
    // [head_dim..head_dim + blockDim.x]: reduction buffer
    extern __shared__ float smem[];
    float* shared_q = smem;
    float* red_smem = &smem[head_dim];
    
    // Load Q into shared memory
    if (tid < head_dim) {
        shared_q[tid] = q[q_offset + tid];
    }
    __syncthreads();
    
    // 1. Compute Scores: q . k
    // Iterate over all timesteps 0..pos
    // Distribute timesteps among threads?
    // Yes. Thread t handles timesteps t, t+blockDim, t+2*blockDim...
    
    // Note: This loop structure is for computing Softmax denominator.
    // To perform Softmax, we need all scores. 
    // We should store scores in global memory 'att' first?
    // Yes, `att` buffer is [n_heads, max_seq_len].
    
    for (int t = tid; t <= pos; t += blockDim.x) {
        float score = 0.0f;
        size_t k_ptr_offset = layer_base_offset + (size_t)t * n_kv_heads * head_dim + kv_h * head_dim;
        
        for (int i = 0; i < head_dim; i++) {
            score += shared_q[i] * k_cache[k_ptr_offset + i];
        }
        score *= scale;
        att[h * max_seq_len + t] = score;
    }
    __syncthreads();
    
    // 2. Softmax
    // Find Max
    float max_val = -INFINITY;
    for (int t = tid; t <= pos; t += blockDim.x) {
        float val = att[h * max_seq_len + t];
        if (val > max_val) max_val = val;
    }
    
    // Block Reduce Max
    red_smem[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (red_smem[tid+s] > red_smem[tid]) red_smem[tid] = red_smem[tid+s];
        }
        __syncthreads();
    }
    max_val = red_smem[0];
    __syncthreads();
    
    // Exp and Sum
    float sum = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        float val = att[h * max_seq_len + t];
        val = expf(val - max_val);
        att[h * max_seq_len + t] = val; // Store probability
        sum += val;
    }
    
    // Block Reduce Sum
    red_smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red_smem[tid] += red_smem[tid+s];
        __syncthreads();
    }
    float total_sum = red_smem[0];
    __syncthreads();
    
    // 3. Compute Weighted Sum: sum(prob * v)
    // Output is vector [head_dim].
    // We can parallelize over head_dim elements?
    // Or parallelize over timesteps and reduce?
    // Parallel over output elements is simpler (no reduction needed across threads).
    // If tid < head_dim, compute output[tid].
    
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float prob = att[h * max_seq_len + t] / total_sum;
            size_t v_ptr_offset = layer_base_offset + (size_t)t * n_kv_heads * head_dim + kv_h * head_dim;
            acc += prob * v_cache[v_ptr_offset + tid];
        }
        out[h * head_dim + tid] = acc;
    }
}

extern "C" void multi_head_attention(float* out, float* q, float* k_cache, float* v_cache, float* att, 
                                     int layer, int pos, int max_seq_len, 
                                     int n_heads, int n_kv_heads, int head_dim) {
    size_t layer_stride = (size_t)max_seq_len * n_kv_heads * head_dim;
    size_t offset = layer * layer_stride;
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    int threads = 256; 
    // Shared mem: Q (head_dim) + Reduction (threads)
    size_t smem_size = (head_dim + threads) * sizeof(float);
    
    attention_kernel<<<n_heads, threads, smem_size>>>(out, q, k_cache, v_cache, att, 
                                                      offset, pos, max_seq_len, 
                                                      n_heads, n_kv_heads, head_dim, scale);
    cudaCheck(cudaGetLastError());
}

