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
