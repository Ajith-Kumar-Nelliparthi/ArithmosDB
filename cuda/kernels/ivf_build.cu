#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "../include/common.cuh"
#include "../include/ivf_index.cuh"
#include "../include/kmeans.cuh"

/**
 * REORDER  KERNEL
 * This kernel moves vectors into their respective inverted lists based on the list indices computed in the previous step.
 * It also keeps track of the original indices of the vectors for later retrieval during search.
 */

 __global__ void reorder_kernel(const float* __restrict__ d_vectors,
                               const int* __restrict__ d_assignments,
                               float* __restrict__ d_inverted_vectors,
                               int* __restrict__ d_inverted_ids,
                               int* __restrict__ d_working_offsets, // out: original indices of vectors in the new order
                                int n, int d) {
                                    
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_idx >= n) return;
    
    // find which cluster this vector belongs to
    int cluster_id = d_assignments[vec_idx];

    // a unique position id inside the cluster's list
    int position_in_list = atomicAdd(&d_working_offsets[cluster_id], 1);

    // store the original id at that position
    d_inverted_ids[position_in_list] = vec_idx;

    // copy the vector to the new position in the inverted list
    #pragma unroll
    for (int i=0; i<d; i++) {
        d_inverted_vectors[position_in_list * d + i] = d_vectors[vec_idx * d + i];
    }
}

extern "C" void build_ivf(const float* d_vectors,
                            IVFIndex* index,
                            int n_vectors,
                            int dim,
                            int nlist /* = 0 */,
                            int max_iter /* = 25 */) {
    if (nlist <= 0) {
        nlist = (int)(sqrtf((float)n_vectors) * 1.2f) + 1;
        if (nlist < 16) nlist = 16;
        if (nlist > 8192) nlist = 8192;
    }

    index->nlist = nlist;
    index->n_vectors = n_vectors;
    index->dim = dim;

    // allocate index components
    CUDA_CHECK(cudaMalloc(&index->d_centeroids, nlist * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&index->d_inverted_vectors, n_vectors * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&index->d_inverted_ids, n_vectors * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&index->d_list_offsets, (nlist + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&index->d_list_sizes, nlist * sizeof(int)));

    // temp
    int *d_assignments = nullptr;
    CUDA_CHECK(cudaMalloc(&d_assignments, n_vectors * sizeof(int)));

    // 1. RUN K-MEANS TO FIND CENTROIDS AND ASSIGNMENTS
    run_kmeans(d_vectors, index->d_centeroids, d_assignments, n_vectors, dim, nlist, max_iter);

    // 2. REORDER VECTORS INTO INVERTED LISTS
    int* d_working_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_working_offsets, nlist * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_working_offsets, 0, nlist * sizeof(int)));

    {
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((n_vectors + block.x - 1) / block.x);

        reorder_kernel<<<grid, block>>>(d_vectors, d_assignments,
                                        index->d_inverted_vectors,
                                        index->d_inverted_ids,
                                        d_working_offsets,
                                        n_vectors, dim);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 3. Build list_sizes and list_offsets using thrust
    thrust::device_ptr<int> assign_ptr(d_assignments);
    thrust::device_vector<int> keys(n_vectors);
    thrust::copy(assign_ptr, assign_ptr + n_vectors, keys.begin());

    // sort assignments to group by cluster
    thrust::sort(keys.begin(), keys.end());

    // count how many vectors in each cluster
    thrust::device_vector<int> unique_clusters(nlist);
    thrust::device_vector<int> list_sizes(nlist);

    auto new_end = thrust::reduce_by_key(keys.begin(), keys.end(),
                                    thrust::constant_iterator<int>(1),
                                    unique_clusters.begin(),
                                    list_sizes.begin());
    
    int num_nonempty = new_end.first - unique_clusters.begin();

    // fill list_sizes
    thrust::device_ptr<int> sizes_ptr(index->d_list_sizes);
    thrust::fill(sizes_ptr, sizes_ptr + nlist, 0);

    thrust::scatter(list_sizes.begin(), list_sizes.begin() + num_nonempty,
                    unique_clusters.begin(),
                    sizes_ptr);

    // Build prefix sums → list_offsets
    thrust::device_ptr<int> offsets_ptr(index->d_list_offsets);
    thrust::exclusive_scan(sizes_ptr, sizes_ptr + nlist,
                           offsets_ptr);
    // Last element should be total n_vectors
    CUDA_CHECK(cudaMemcpy((offsets_ptr + nlist).get(),
                          &n_vectors,
                          sizeof(int),
                          cudaMemcpyHostToDevice));

    // Cleanup
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_working_offsets));

    printf("IVF index built: %d vectors, dim=%d, nlist=%d\n", n_vectors, dim, nlist);
}