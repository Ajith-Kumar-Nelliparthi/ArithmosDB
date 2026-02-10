#include <stdio.h>
#include <cuda_runtime.h>

__global__ void distance_kernel(const float* __restrict__ queries,
                                const float* __restrict__ vectors,
                                float* __restrict__ distances,
                                int n, int d, int nq){
    
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;

    if (vec_idx < n && query_idx < nq){
        float sum = 0.0f;

        // shared memory for the query vector
        extern __shared__ float query_shared[];
        if (threadIdx.x < d) {
            query_shared[threadIdx.x] = queries[query_idx * d + threadIdx.x];
        }
        __syncthreads();

        for (int i=0; i<d; i++){
            float diff = vectors[vec_idx * d + i] - query_shared[i];
            sum += diff * diff;
        }

        // Store the computed distance
        distances[query_idx * n + vec_idx] = sqrtf(sum);
    }
                                }

extern  "C" void compute_distances(const float* queries, const float* vectors, float* distances, int n, int d, int nq){
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, nq);
    size_t sharedMemSize = d * sizeof(float);

    distance_kernel<<<gridSize, blockSize, sharedMemSize>>>(queries, vectors, distances, n, d, nq);
    cudaDeviceSynchronize();
}