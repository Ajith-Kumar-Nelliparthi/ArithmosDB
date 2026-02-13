#include <stdio.h>
#include <cuda_runtime.h>

// error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * Transpose matrix form [m, n] to [n, m] with shared memory tiling
 * Prevents bank conflicts with padding.
 */

 __global__ void transpose_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int m, int n) {
    __shared__ float tile[32][33]; // +1 padding

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // load into shared memory
    if (x < n && y < m) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x]; 
    }
    __syncthreads();

    // write transposed
    x = blockIdx.y * blockDim.y + threadIdx.y;
    y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < m && y < n) {
        output[y * m + x] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void transpose(const float* d_input,
                            float* d_output,
                            int m, int n){
    dim3 blockSize(32, 32);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                    (m + blockSize.y - 1) / blockSize.y);
    transpose_kernel<<<gridSize, blockSize>>>(d_input, d_output, m, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}