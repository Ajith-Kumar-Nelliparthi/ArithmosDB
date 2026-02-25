// src/distance.cu
#include "../include/distance.cuh"

/**
 * Compute L2 distances between queries and vectors
 * Uses shared memory for query to reduce global memory loads.
 */

__global__ void distance_kernel(const float* __restrict__ queries,
                                const float* __restrict__ vectors,
                                float* __restrict__ distances,
                                int n_vectors, int dim, int n_queries){
    
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;

    if (vec_idx >= n_vectors || query_idx >= n_queries) return;

    extern __shared__ float s_query[];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        s_query[i] = queries[query_idx * dim + i];
    }
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = s_query[i] - vectors[vec_idx * dim + i];
        sum += diff * diff;
    }
    distances[query_idx * n_vectors + vec_idx] = sum;
}

extern  "C" void compute_distances(const float* d_queries, 
                                    const float* d_vectors, 
                                    float* d_distances, 
                                    int n_vectors, int dim, int n_queries){
    dim3 block(THREADS_PER_BLOCK);
    dim3 gridSize((n_vectors + block.x - 1) / block.x, n_queries);
    size_t sharedMemSize = dim * sizeof(float);

    distance_kernel<<<gridSize, block, sharedMemSize>>>(d_queries, d_vectors, d_distances, n_vectors, dim, n_queries);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}