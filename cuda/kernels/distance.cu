#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Compute L2 distances between queries and vectors
 * Uses shared memory for query to reduce global memory loads.
 */

__global__ void distance_kernel(const float* __restrict__ queries,
                                const float* __restrict__ vectors,
                                float* __restrict__ distances,
                                int n, int d, int nq){
    
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;

    if (vec_idx < n && query_idx < nq){
        // shared memory for the query vector
        extern __shared__ float query_shared[];
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            query_shared[i] = queries[query_idx * d + i];
        }
        __syncthreads();

        float sum = 0.0f;
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}