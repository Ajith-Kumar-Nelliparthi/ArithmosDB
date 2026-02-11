// Top-K Selection
#include <stdio.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void topk_selection(const float* __restrict__ distances,
                                float* __restrict__ indices,
                                float* __restrict__ top_distances,
                                int n, int k, int nq) {
    
    int query_idx = blockIdx.x;
    if (query_idx >= n) return;

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
                        float* d_indices,
                        float* d_top_distances,
                        int n, int k, int nq){
    dim3 blockSize(256);
    dim3 gridSize(nq);
    size_t size = blockSize.x * (sizeof(float) + sizeof(int));

    topk_selection<<<gridSize, blockSize, size>>>(d_distances, d_indices, d_top_distances, n, k, nq);
    cudaDeviceSynchronize();
}