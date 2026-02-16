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

/**
 * Normalize vectors in place for cosine similarity
 */
__global__ void normalize_kernel(float* __restrict__ vectors,
                                 int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float norm = 0.0f;
        
        // Compute L2 norm
        for (int i = 0; i < d; i++) {
            float val = vectors[idx * d + i];
            norm += val * val;
        }
        norm = sqrtf(norm);
        
        // Normalize
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                vectors[idx * d + i] /= norm;
            }
        }
    }
}

extern "C" void normalise_vectors(float* d_vectors, int d, int n) {
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);
    normalize_kernel<<<gridSize, blockSize>>>(d_vectors, d, n);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

/**
 * Compute Cosine Similarity
 */
__global__ void cosine_similarity_kernel(const float* __restrict__ queries,
                                            const float* __restrict__ vectors,
                                            float* __restrict__ similarities,
                                            int n, int d, int nq) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (vec_idx < n && query_idx < nq) {
        extern __shared__ float query_shared[];

        // load query into shared memory
        for (int i=tid; i<d; i+= blockDim.x) {
            query_shared[i] = queries[query_idx * d + i];
        }
        __syncthreads();

        float dot_product = 0.0f;
        for (int i=0; i<d; i++) {
            dot_product += vectors[vec_idx * d + i] * query_shared[i];
        }

        // store as distance
        similarities[query_idx * n + vec_idx] = 1.0f - dot_product;
    }
}

extern "C" void cosine_similarity(const float* queries,
                                    const float* vectors,
                                float* similarities,
                                int n, int d, int nq) {
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, nq);
    size_t sharedMem = d * sizeof(float);

    cosine_similarity_kernel<<<gridSize, blockSize, sharedMem>>>(queries, vectors, similarities, n, d, nq);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}