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

// ===========================================================================
// Paged Attention Kernels
// ===========================================================================

__device__ long get_physical_offset_gpu(int num_blocks, int block_size, int layer, int physical_block, int block_offset, int n_kv_heads, int head_dim) {
    long layer_stride = (long)num_blocks * block_size * n_kv_heads * head_dim;
    long block_stride = (long)block_size * n_kv_heads * head_dim;
    long offset_stride = (long)n_kv_heads * head_dim;
    
    return (long)layer * layer_stride + 
           (long)physical_block * block_stride + 
           (long)block_offset * offset_stride;
}

__global__ void update_kv_paged_kernel(float* pool_k, float* pool_v, float* k, float* v,
                                       int num_blocks, int block_size, int layer, int physical_block, int block_offset,
                                       int n_kv_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = n_kv_heads * head_dim;
    
    if (idx < size) {
        long offset = get_physical_offset_gpu(num_blocks, block_size, layer, physical_block, block_offset, n_kv_heads, head_dim);
        pool_k[offset + idx] = k[idx];
        pool_v[offset + idx] = v[idx];
    }
}

extern "C" void update_kv_cache_paged(KVCacheManager* mgr, BlockTable* block_table, float* k, float* v, 
                                      int layer, int pos, int n_kv_heads, int head_dim) {
    
    int block_size = mgr->block_size;
    int logical_block_idx = pos / block_size;
    int block_offset = pos % block_size;
    
    // Resolve physical block on Host
    if (logical_block_idx >= block_table->num_blocks) {
        printf("Error: Block table too small! logical_idx=%d, num_blocks=%d\n", logical_block_idx, block_table->num_blocks);
        return;
    }
    int physical_block_idx = block_table->block_indices[logical_block_idx];
    
    int size = n_kv_heads * head_dim;
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    
    update_kv_paged_kernel<<<blocks, threads>>>(mgr->pool_k, mgr->pool_v, k, v, 
                                                mgr->num_blocks, block_size, layer, physical_block_idx, block_offset,
                                                n_kv_heads, head_dim);
    cudaCheck(cudaGetLastError());
}


__global__ void paged_attention_kernel(float* out, float* q, float* pool_k, float* pool_v, int* block_indices, float* att,
                                       int num_blocks, int block_size, int layer, int pos, int max_seq_len,
                                       int n_heads, int n_kv_heads, int head_dim, float scale) {
    
    // One block per head
    int h = blockIdx.x;
    if (h >= n_heads) return;
    
    int tid = threadIdx.x;
    int kv_h = h / (n_heads / n_kv_heads); // GQA
    
    // Offsets
    int q_offset = h * head_dim;
    
    extern __shared__ float smem[];
    float* shared_q = smem;
    float* red_smem = &smem[head_dim];
    
    // Load Q
    if (tid < head_dim) {
        shared_q[tid] = q[q_offset + tid];
    }
    __syncthreads();
    
    // 1. Compute Scores
    for (int t = tid; t <= pos; t += blockDim.x) {
        // Resolve Physical Address
        int logical_block = t / block_size;
        int block_offset = t % block_size;
        
        int physical_block = block_indices[logical_block];
        
        long offset = get_physical_offset_gpu(num_blocks, block_size, layer, physical_block, block_offset, n_kv_heads, head_dim);
        float* k_head = pool_k + offset + kv_h * head_dim;
        
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += shared_q[i] * k_head[i];
        }
        score *= scale;
        att[h * max_seq_len + t] = score;
    }
    __syncthreads();
    
    // 2. Softmax
    float max_val = -INFINITY;
    for (int t = tid; t <= pos; t += blockDim.x) {
        float val = att[h * max_seq_len + t];
        if (val > max_val) max_val = val;
    }
    
    red_smem[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && red_smem[tid+s] > red_smem[tid]) red_smem[tid] = red_smem[tid+s];
        __syncthreads();
    }
    max_val = red_smem[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int t = tid; t <= pos; t += blockDim.x) {
        float val = expf(att[h * max_seq_len + t] - max_val);
        att[h * max_seq_len + t] = val;
        sum += val;
    }
    
    red_smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red_smem[tid] += red_smem[tid+s];
        __syncthreads();
    }
    float total_sum = red_smem[0];
    __syncthreads();
    
    // 3. Weighted Sum
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            float prob = att[h * max_seq_len + t] / total_sum;
            
            int logical_block = t / block_size;
            int block_offset = t % block_size;
            int physical_block = block_indices[logical_block];
            long offset = get_physical_offset_gpu(num_blocks, block_size, layer, physical_block, block_offset, n_kv_heads, head_dim);
            
            float* v_head = pool_v + offset + kv_h * head_dim;
            acc += prob * v_head[tid]; // Parallelize over output dim (tid)
        }
        out[h * head_dim + tid] = acc;
    }
}

extern "C" void paged_attention(float* out, float* q, KVCacheManager* mgr, BlockTable* block_table, float* att,
                                int layer, int pos, int max_seq_len, int n_heads, int n_kv_heads, int head_dim) {
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Copy Block Table to Device
    int num_logical_blocks = (pos / mgr->block_size) + 1;
    int* d_block_indices;
    cudaCheck(cudaMalloc((void**)&d_block_indices, num_logical_blocks * sizeof(int)));
    cudaCheck(cudaMemcpy(d_block_indices, block_table->block_indices, num_logical_blocks * sizeof(int), cudaMemcpyHostToDevice));
    
    int threads = 256;
    size_t smem_size = (head_dim + threads) * sizeof(float);
    
    paged_attention_kernel<<<n_heads, threads, smem_size>>>(out, q, mgr->pool_k, mgr->pool_v, d_block_indices, att,
                                                            mgr->num_blocks, mgr->block_size, layer, pos, max_seq_len,
                                                            n_heads, n_kv_heads, head_dim, scale);
    cudaCheck(cudaGetLastError());
    
    cudaCheck(cudaFree(d_block_indices));
}
