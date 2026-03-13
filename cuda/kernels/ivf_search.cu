#include "../include/common.cuh"
#include "../include/ivf_index.cuh"
#include "../include/distance.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

/**
 * Coarse Search KERNEL
 * This kernel calculates the distance from one query to all k centeroids.
 */

 __global__ void coarse_search_kernel(const float* __restrict__ d_queries,
                                        const float* __restrict__ d_centroids,
                                        float* __restrict__ d_coarse_dists,
                                        int nlist,
                                        int dim,
                                        int n_queries) {
    int centeroid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;

    if (centeroid_idx >= nlist || query_idx >= n_queries) {
        return;
    }

    // store the query in shared memory for faster access
    extern __shared__ float s_query[];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        s_query[i] = d_queries[query_idx * dim + i];
    }
    __syncthreads();

    float sum = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < dim; i++) {
        float diff = s_query[i] - d_centroids[centeroid_idx * dim + i];
        sum += diff * diff;
    }

    d_coarse_dists[query_idx * nlist + centeroid_idx] = sum;
}


__global__ void ivf_final_topk_kernel(const float* __restrict__ d_queries,
                                      const float* __restrict__ d_inverted_vectors,
                                      const int*   __restrict__ d_inverted_ids,
                                      const int*   __restrict__ d_list_offsets,
                                      const int*   __restrict__ d_selected_clusters,  // n_queries × nprobe
                                      int*         __restrict__ d_out_indices,
                                      float*       __restrict__ d_out_distances,
                                      int n_queries,
                                      int k,
                                      int nprobe,
                                      int dim) {
    int q = blockIdx.x;
    if (q >= n_queries) return;

    extern __shared__ float s_query[];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        s_query[i] = d_queries[q * dim + i];
    }
    __syncthreads();

    // Small heap in registers (assumes k ≤ 64)
    float heap_dist[64];
    int   heap_idx[64];
    for (int i = 0; i < k; ++i) {
        heap_dist[i] = FLT_MAX;
        heap_idx[i]  = -1;
    }

    // Process each probed cluster
    for (int p = 0; p < nprobe; ++p) {
        int cluster = d_selected_clusters[q * nprobe + p];
        int start   = d_list_offsets[cluster];
        int end     = d_list_offsets[cluster + 1];

        for (int v = start + threadIdx.x; v < end; v += blockDim.x) {
            float dist = 0.0f;
            #pragma unroll 4
            for (int j = 0; j < dim; ++j) {
                float diff = s_query[j] - d_inverted_vectors[v * dim + j];
                dist += diff * diff;
            }

            if (dist < heap_dist[k - 1]) {
                heap_dist[k - 1] = dist;
                heap_idx[k - 1]  = d_inverted_ids[v];

                // Bubble up (simple insertion sort style)
                for (int j = k - 1; j > 0; --j) {
                    if (heap_dist[j] < heap_dist[j - 1]) {
                        float td = heap_dist[j]; heap_dist[j] = heap_dist[j - 1]; heap_dist[j - 1] = td;
                        int   ti = heap_idx[j];  heap_idx[j]  = heap_idx[j - 1];  heap_idx[j - 1]  = ti;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    // Write result (only first k threads)
    if (threadIdx.x < k) {
        d_out_distances[q * k + threadIdx.x] = heap_dist[threadIdx.x];
        d_out_indices[q * k + threadIdx.x]   = heap_idx[threadIdx.x];
    }
}

// Host: search_ivf - main entry point
extern "C" void search_ivf(const float* d_queries,
                           const IVFIndex* index,
                           int* d_out_indices,
                           float* d_out_distances,
                           int n_queries,
                           int k,
                           int nprobe) {
    if (nprobe > index->nlist) nprobe = index->nlist;
    if (nprobe < 1) nprobe = 1;

    // 1. Coarse distances to all centroids
    float* d_coarse_dists = nullptr;
    CUDA_CHECK(cudaMalloc(&d_coarse_dists, n_queries * index->nlist * sizeof(float)));

    {
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((index->nlist + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, n_queries);
        size_t shared = index->dim * sizeof(float);

        coarse_search_kernel<<<grid, block, shared>>>(
            d_queries, index->d_centroids, d_coarse_dists,
            index->nlist, index->dim, n_queries);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 2. Select top-nprobe clusters per query
    std::vector<int> h_selected(n_queries * nprobe);

    {
        thrust::device_ptr<float> coarse_ptr(d_coarse_dists);
        thrust::device_vector<int> cluster_ids(index->nlist);
        thrust::sequence(cluster_ids.begin(), cluster_ids.end());

        for (int q = 0; q < n_queries; ++q) {
            auto query_start = coarse_ptr + q * index->nlist;

            thrust::device_vector<thrust::pair<float, int>> pairs(index->nlist);
            thrust::transform(thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(index->nlist),
                              query_start,
                              pairs.begin(),
                              [] __device__ (int id, float dist) {
                                  return thrust::make_pair(dist, id);
                              });

            thrust::sort(pairs.begin(), pairs.end());

            // Copy sorted pairs to host to access .second
            std::vector<thrust::pair<float, int>> h_pairs(index->nlist);
            thrust::copy(pairs.begin(), pairs.end(), h_pairs.begin());

            for (int p = 0; p < nprobe; ++p) {
                h_selected[q * nprobe + p] = h_pairs[p].second;
            }
        }
    }

    int* d_selected_clusters = nullptr;
    CUDA_CHECK(cudaMalloc(&d_selected_clusters, n_queries * nprobe * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_selected_clusters, h_selected.data(),
                          n_queries * nprobe * sizeof(int),
                          cudaMemcpyHostToDevice));

    // 3. Fine search + final top-k in one kernel (one block per query)
    {
        size_t shared_mem = index->dim * sizeof(float);
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid(n_queries);

        ivf_final_topk_kernel<<<grid, block, shared_mem>>>(
            d_queries,
            index->d_inverted_vectors,
            index->d_inverted_ids,
            index->d_list_offsets,
            d_selected_clusters,
            d_out_indices,
            d_out_distances,
            n_queries, k, nprobe, index->dim);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_coarse_dists));
    CUDA_CHECK(cudaFree(d_selected_clusters));

    printf("IVF search completed: %d queries, k=%d, nprobe=%d\n", n_queries, k, nprobe);
}