#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../include/ops.h"
#include "../kernels/gpu/nano_cuda.h"

// CPU Reference implementation
void matmul_cpu(float* out, float* in, float* weight, int in_dim, int out_dim) {
    for (int j = 0; j < out_dim; j++) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; i++) {
            sum += in[i] * weight[i * out_dim + j];
        }
        out[j] = sum;
    }
}

int main() {
    // Set random seed for reproducibility
    srand(42);

    int in_dim = 256;
    int out_dim = 512;
    
    size_t size_in = in_dim * sizeof(float);
    size_t size_out = out_dim * sizeof(float);
    size_t size_w = in_dim * out_dim * sizeof(float);

    printf("Initializing memory...\n");

    // Host memory
    float *h_in = (float*)malloc(size_in);
    float *h_w = (float*)malloc(size_w);
    float *h_out_cuda = (float*)malloc(size_out);
    float *h_out_cpu = (float*)malloc(size_out);

    // Initialize
    for(int i=0; i<in_dim; i++) h_in[i] = (float)rand() / RAND_MAX;
    for(int i=0; i<in_dim*out_dim; i++) h_w[i] = (float)rand() / RAND_MAX / in_dim;

    // Device memory
    float *d_in, *d_w, *d_out;
    cudaCheck(cudaMalloc(&d_in, size_in));
    cudaCheck(cudaMalloc(&d_w, size_w));
    cudaCheck(cudaMalloc(&d_out, size_out));

    cudaCheck(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_w, h_w, size_w, cudaMemcpyHostToDevice));

    printf("Running CUDA kernel...\n");
    // Run Custom Kernel (via ops.h interface)
    init_ops(); 
    matmul(d_out, d_in, d_w, in_dim, out_dim);
    cudaCheck(cudaDeviceSynchronize());
    free_ops();

    cudaCheck(cudaMemcpy(h_out_cuda, d_out, size_out, cudaMemcpyDeviceToHost));

    printf("Running CPU reference...\n");
    // Run CPU Reference
    matmul_cpu(h_out_cpu, h_in, h_w, in_dim, out_dim);

    // Verify
    float max_diff = 0.0f;
    for(int i=0; i<out_dim; i++) {
        float diff = fabs(h_out_cuda[i] - h_out_cpu[i]);
        if(diff > max_diff) max_diff = diff;
    }

    printf("Matmul Test: in=%d, out=%d\n", in_dim, out_dim);
    printf("Max Difference: %e\n", max_diff);
    
    // Tolerance might need adjustment depending on accumulation error
    bool passed = max_diff < 1e-4;
    printf("Test %s\n", passed ? "PASSED" : "FAILED");

    cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
    free(h_in); free(h_w); free(h_out_cuda); free(h_out_cpu);
    
    return passed ? 0 : 1;
}

