#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>


/**
 * REORDER  KERNEL
 * This kernel moves vectors into their respective inverted lists based on the list indices computed in the previous step.
 * It also keeps track of the original indices of the vectors for later retrieval during search.
 */

 __global__ void reorder_kernel(const float* __restrict__ d_vectors, // original n x d matrix of vectors
                                const int* d_assignments, // cluster id for each vector
                                const int* d_offsets, // starting position of each cluster
                                int* d_working_offsets, // current offset for each cluster (updated atomically
                                float* d_inverted_vectors, // out: reordered n x d matrix of vectors
                                int* d_inverted_ids, // out: original indices of vectors in the new order
                                int n, int d) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < n) {
        // find which cluster this vector belongs to
        int cluster_id = d_assignments[vec_idx];

        // a unique position id inside the cluster's list
        int position_in_list = atomicAdd(&d_working_offsets[cluster_id], 1);

        // store the original id at that position
        d_inverted_ids[position_in_list] = vec_idx;

        // copy the vector to the new position in the inverted list
        for (int i=0; i<d; i++) {
            d_inverted_vectors[position_in_list * d + i] = d_vectors[vec_idx * d + i];
        }
    }
}

/**
 * Coarse Search KERNEL
 * This kernel calculates the distance from one query to all k centeroids.
 */

 __global__ void coarse_search_kernel(const float* d_query, // query vector (size d)
                                        const float* d_centeroids, // all k centeroids (size k x d)
                                        float* d_coarse_distances, // out: distances from the query to each centeroid (size k)
                                        int k, // no.of centeroids
                                        int d) {
    int centeroid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;

    if (centeroid_idx < k) {
        // store the query in shared memory for faster access
        extern __shared__ float s_query[];
        for (int i=threadIdx.x; i<d; i+=blockDim.x) {
            s_query[i] = d_query[query_idx * d + i];
        }
        __syncthreads();

        // compute the distance from the query to the centeroid
        float sum = 0.0f;
        for (int i=0; i<d; i++) {
            float diff = s_query[i] - d_centeroids[centeroid_idx * d + i];
            sum += diff * diff;
        }
        d_coarse_distances[query_idx * k + centeroid_idx] = sum;
    }
}

/**
 * FINE SEARCH KERNEL
 * This kernel computes the distances from one query to all vectors 
 * and store in the "shortlisted cluster".
 */

__global__ void fine_search_kernel(const float* d_query,
                                    const float* d_inverted_vectors,
                                    const int* d_inverted_ids,
                                    const int* d_list_offsets,
                                    const int* d_list_sizes,
                                    const int* d_selected_cluster_ids,
                                    float* d_neighbor_distances,
                                    int* d_neighbor_ids,
                                    int nprobe,
                                    int d) {
    int probe_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (probe_idx < nprobe) {
        int cluster_id = d_selected_cluster_ids[probe_idx];
        int start_offset = d_list_offsets[cluster_id];
        int list_size = d_list_sizes[cluster_id];

        // store the query in shared memory for faster access
        extern __shared__ float s_query[];
        int query_idx = blockIdx.y;
        for (int i= threadIdx.x; i<d; i+=blockDim.x) {
            s_query[i] = d_query[query_idx * d + i];
        }
        __syncthreads();

        // compute distances from the query to all vectors in this cluster
        for (int i=0; i<list_size; i++) {
            int vec_idx = d_inverted_ids[start_offset + i];
            float sum = 0.0f;
            for (int j=0; j<d; j++) {
                float diff = s_query[j] - d_inverted_vectors[(start_offset + i) * d + j];
                sum += diff * diff;
            }
            d_neighbor_distances[probe_idx * list_size + i] = sum;
            d_neighbor_ids[probe_idx * list_size + i] = vec_idx;
        }
    }
}