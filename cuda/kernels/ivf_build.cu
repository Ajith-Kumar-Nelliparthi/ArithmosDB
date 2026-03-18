// kernels/ivf_build.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <thrust/scatter.h>
#include <thrust/fill.h>

#include "../include/common.cuh"
#include "../include/ivf_index.cuh"
#include "../include/kmeans.cuh"

__global__ void reorder_kernel(const float* __restrict__ d_vectors,
                               const int*   __restrict__ d_assignments,
                               float* __restrict__ d_inverted_vectors,
                               int*   __restrict__ d_inverted_ids,
                               int*   __restrict__ d_working_offsets,
                               int n, int d) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n) return;

    int cluster_id       = d_assignments[vec_idx];
    int position_in_list = atomicAdd(&d_working_offsets[cluster_id], 1);

    d_inverted_ids[position_in_list] = vec_idx;
    for (int i = 0; i < d; i++)
        d_inverted_vectors[(size_t)position_in_list * d + i] =
            d_vectors[(size_t)vec_idx * d + i];
}

// BUG 4 FIX: remove default args from extern "C"
extern "C" void build_ivf(const float* d_vectors,
                          IVFIndex* index,
                          int n_vectors,
                          int dim,
                          int nlist,
                          int max_iter) {
    if (nlist <= 0) {
        nlist = (int)(sqrtf((float)n_vectors) * 1.2f) + 1;
        if (nlist < 16)   nlist = 16;
        if (nlist > 8192) nlist = 8192;
    }

    index->nlist     = nlist;
    index->n_vectors = n_vectors;
    index->dim       = dim;

    CUDA_CHECK(cudaMalloc(&index->d_centroids,        (size_t)nlist * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&index->d_inverted_vectors, (size_t)n_vectors * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&index->d_inverted_ids,     n_vectors * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&index->d_list_offsets,    (nlist + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&index->d_list_sizes,       nlist * sizeof(int)));

    int* d_assignments = nullptr;
    CUDA_CHECK(cudaMalloc(&d_assignments, n_vectors * sizeof(int)));

    run_kmeans(d_vectors, index->d_centroids, d_assignments,
               n_vectors, dim, nlist, max_iter);

    // ---- Build list sizes and offsets via Thrust ----
    thrust::device_ptr<int> sizes_ptr(index->d_list_sizes);
    thrust::fill(sizes_ptr, sizes_ptr + nlist, 0);

    thrust::device_vector<int> sorted_assigns(d_assignments, d_assignments + n_vectors);
    thrust::sort(sorted_assigns.begin(), sorted_assigns.end());

    thrust::device_vector<int> ones(n_vectors, 1);
    thrust::device_vector<int> unique_clusters(nlist);
    thrust::device_vector<int> list_sizes_tmp(nlist, 0);

    auto new_end = thrust::reduce_by_key(
        sorted_assigns.begin(), sorted_assigns.end(),
        ones.begin(),
        unique_clusters.begin(),
        list_sizes_tmp.begin());

    int num_nonempty = (int)(new_end.first - unique_clusters.begin());

    thrust::scatter(list_sizes_tmp.begin(),
                    list_sizes_tmp.begin() + num_nonempty,
                    unique_clusters.begin(),
                    sizes_ptr);

    // prefix sum → offsets[0..nlist-1]
    thrust::device_ptr<int> offsets_ptr(index->d_list_offsets);
    thrust::exclusive_scan(sizes_ptr, sizes_ptr + nlist, offsets_ptr);

    // offsets[nlist] = total n_vectors
    CUDA_CHECK(cudaMemcpy(index->d_list_offsets + nlist,
                          &n_vectors, sizeof(int),
                          cudaMemcpyHostToDevice));

    int* d_working_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_working_offsets, nlist * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_working_offsets,
                          index->d_list_offsets,
                          nlist * sizeof(int),
                          cudaMemcpyDeviceToDevice));

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n_vectors + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    reorder_kernel<<<grid, block>>>(d_vectors, d_assignments,
                                    index->d_inverted_vectors,
                                    index->d_inverted_ids,
                                    d_working_offsets,
                                    n_vectors, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_working_offsets));

    printf("IVF index built: %d vectors, dim=%d, nlist=%d\n",
           n_vectors, dim, nlist);
}
