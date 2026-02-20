/**
 * Find top-k smallest distances for each query using bubble sort.
 * Note: Bubble sort can be updated with bitonic sort or heap.
 */
#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void topk_selection(const float* __restrict__ distances,
                                int* __restrict__ indices,
                                float* __restrict__ top_distances,
                                int n, int k, int nq) {
    
    int query_idx = blockIdx.x;
    if (query_idx >= nq) return;

    extern __shared__ char shared_mem[];
    float* s_dist = (float*)shared_mem;
    int* s_idx = (int*)(shared_mem + blockDim.x * sizeof(float));
    
    int tid = threadIdx.x;
    const float* query_distances = distances + query_idx * n;

    // each thread loads one element
    if (tid < n){
        s_dist[tid] = query_distances[tid];
        s_idx[tid] = tid;
    } else {
        s_dist[tid] = FLT_MAX;
        s_idx[tid] = -1;
    }
    __syncthreads();

    // Bubble Sort 
    for (int i = 0; i < k; i++){
        for (int j = tid; j < n - 1 - i; j += blockDim.x) {
            if (s_dist[j] > s_dist[j + 1]){
                // swap distances
                float tmp_distance = s_dist[j];
                s_dist[j] = s_dist[j + 1];
                s_dist[j + 1] = tmp_distance;

                // swap indices
                int tmp_idx = s_idx[j];
                s_idx[j] = s_idx[j + 1];
                s_idx[j + 1] = tmp_idx;
            }
        }
        __syncthreads();
    }
    // write top-k results
    if (tid < k){
        indices[query_idx * k + tid] = s_idx[tid];
        top_distances[query_idx * k + tid] = s_dist[tid];
    }    
}

extern "C" void find_topk(const float* d_distances,
                        int* d_indices,
                        float* d_top_distances,
                        int n, int k, int nq){
    dim3 blockSize(256);
    dim3 gridSize(nq);
    size_t size = blockSize.x * (sizeof(float) + sizeof(int));

    topk_selection<<<gridSize, blockSize, size>>>(d_distances, d_indices, d_top_distances, n, k, nq);
    cudaDeviceSynchronize();
}

/**
 * Disadvantages of this approach:
 * 1. Inefficient for large n: Bubble sort has a time complexity of O(n^2), which can be very slow for large n.
 * 2. Not suitable for large k: If k is close to n, the performance will degrade significantly as the algorithm will need to perform more comparisons and swaps.
 * 3. High shared memory usage: This approach requires a large amount of shared memory to store the distances and indices for each query, which may limit the number of threads that can be launched and reduce occupancy.
 * 4. Not scalable: As the number of queries (nq) increases, the performance may degrade due to increased contention for shared memory and global memory bandwidth.
 */