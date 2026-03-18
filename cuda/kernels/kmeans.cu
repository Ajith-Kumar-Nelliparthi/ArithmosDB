// kernels/kmeans.cu
#include "../include/kmeans.cuh"
#include "../include/common.cuh"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <float.h>

// ---- k-means++ min-distance kernel ----
__global__ void kmeanspp_min_dist_kernel(const float* vectors,
                                          const float* centroids,
                                          float* min_dists,
                                          int n, int d, int cur_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float best = FLT_MAX;
    for (int c = 0; c < cur_k; c++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = vectors[idx * d + j] - centroids[c * d + j];
            sum += diff * diff;
        }
        if (sum < best) best = sum;
    }
    min_dists[idx] = best;
}

extern "C" void init_kmeanspp(const float* d_vectors,
                              float* d_centroids,
                              int n_vectors, int dim, int n_clusters) {
    if (n_clusters <= 0) return;

    int first = rand() % n_vectors;

    CUDA_CHECK(cudaMemcpy(d_centroids,
                          d_vectors + (size_t)first * dim,
                          dim * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    float* d_min_dists;
    CUDA_CHECK(cudaMalloc(&d_min_dists, n_vectors * sizeof(float)));

    for (int c = 1; c < n_clusters; c++) {
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((n_vectors + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        kmeanspp_min_dist_kernel<<<grid, block>>>(
            d_vectors, d_centroids, d_min_dists, n_vectors, dim, c);

        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::device_ptr<float> ptr(d_min_dists);
        int chosen = (int)(thrust::max_element(ptr, ptr + n_vectors) - ptr);

        CUDA_CHECK(cudaMemcpy(d_centroids + (size_t)c * dim,
                              d_vectors   + (size_t)chosen * dim,
                              dim * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaFree(d_min_dists));
}

// ---- assign each vector to its nearest centroid ----
__global__ void assign_clusters(const float* __restrict__ vectors,
                                const float* __restrict__ centroids,
                                int* assignments,
                                int n_vectors, int dim, int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    float min_dist = FLT_MAX;
    int best = 0;
    for (int c = 0; c < n_clusters; c++) {
        float dist = 0.0f;
        for (int j = 0; j < dim; j++) {
            float diff = vectors[idx * dim + j] - centroids[c * dim + j];
            dist += diff * diff;
        }
        if (dist < min_dist) { min_dist = dist; best = c; }
    }
    assignments[idx] = best;
}

// ---- accumulate vectors into centroid sums ----
__global__ void update_centroids_kernel(const float* __restrict__ vectors,
                                        const int*   __restrict__ assignments,
                                        float* centroids,
                                        int*   cluster_counts,
                                        int n_vectors, int dim, int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    int cluster = assignments[idx];
    atomicAdd(&cluster_counts[cluster], 1);
    for (int j = 0; j < dim; j++)
        atomicAdd(&centroids[cluster * dim + j], vectors[idx * dim + j]);
}

// ---- divide accumulated sums by counts ----
__global__ void normalize_centroids_kernel(float* centroids,
                                           const int* __restrict__ cluster_counts,
                                           int dim, int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_clusters) return;

    int count = cluster_counts[idx];
    if (count > 0) {
        float inv = 1.0f / (float)count;
        for (int j = 0; j < dim; j++)
            centroids[idx * dim + j] *= inv;
    }
}

extern "C" void run_kmeans(const float* d_vectors,
                           float* d_centroids,
                           int* d_assignments,
                           int n_vectors, int dim, int n_clusters, int max_iter) {
    init_kmeanspp(d_vectors, d_centroids, n_vectors, dim, n_clusters);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid_n((n_vectors   + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 grid_k((n_clusters  + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    for (int iter = 0; iter < max_iter; iter++) {
        assign_clusters<<<grid_n, block>>>(
            d_vectors, d_centroids, d_assignments, n_vectors, dim, n_clusters);
        CUDA_CHECK(cudaDeviceSynchronize());

        int* d_cluster_counts;
        CUDA_CHECK(cudaMalloc(&d_cluster_counts, n_clusters * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cluster_counts, 0, n_clusters * sizeof(int)));

        CUDA_CHECK(cudaMemset(d_centroids, 0, (size_t)n_clusters * dim * sizeof(float)));

        update_centroids_kernel<<<grid_n, block>>>(
            d_vectors, d_assignments, d_centroids, d_cluster_counts,
            n_vectors, dim, n_clusters);

        normalize_centroids_kernel<<<grid_k, block>>>(
            d_centroids, d_cluster_counts, dim, n_clusters);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_cluster_counts));
    }
}
