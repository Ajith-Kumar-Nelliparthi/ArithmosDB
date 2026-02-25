// src/kmeans.cu
#include "../include/kmeans.cuh"
#include "../include/distance.cuh"
#include <thrust/device_ptr.h>
#include <thrust/max_element.h>

__global__ void kmenas_min_dist_kernel(const float* vectors,
                                        const float* centeroids,
                                        float* min_dists,
                                        int n, int d, int cur_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float best = FLT_MAX;
    for (int c = 0; c < cur_k; c++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = vectors[idx * d + j] - centeroids[c * d + j];
            sum += diff * diff;
        }
        if (sum < best) best = sum;
    }
    min_dists[idx] = best;
}

extern "C" void init_kmeanspp(const float* d_vectors,
                            float* d_centeroids,
                            int n_vectors, int dim, int n_clusters) {
    if (n_clusters <= 0) return;

    // Randomly select the first centeroid
    int first = rand() % n_vectors;
    cudaMemcpy(d_centeroids, d_centeroids + first * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    float *d_min_dists;
    CUDA_CHECK(cudaMalloc(&d_min_dists, n_vectors * sizeof(float)));

    for (int c = 1; c < n_clusters; c++) {
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((n_vectors + block.x - 1) / block.x);
        kmenas_min_dist_kernel<<<grid, block>>>(d_vectors, d_centeroids, d_min_dists, n_vectors, dim, c);

        thrust::device_ptr<float> ptr(d_min_dists);
        auto max_it = thrust::max_element(ptr, ptr + n_vectors);
        int chosen = max_it - ptr;

        CUDA_CHECK(cudaMemcpy(d_centeroids + c * dim,
                            d_vectors + chosen * dim,
                            dim * sizeof(float),
                            cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaFree(d_min_dists));
}

__global__ void assign_clusters(const float* __restrict__ vectors,
                            const float* __restrict__ centeroids,
                            int* assignments,
                            int n_vectors, int dim, int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    float min_dist = FLT_MAX;
    int best_cluster = 0;
    for (int c = 0; c < n_clusters; c++) {
        float dist = 0.0f;
        for (int j = 0; j < dim; j++) {
            float diff = vectors[idx * dim + j] - centeroids[c * dim + j];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }
    assignments[idx] = best_cluster;
}

__global__ void update_centeroids(const float* __restrict__ vectors,
                                const int* __restrict__ assignments,
                                float* centeroids,
                                int* cluster_counts,
                                int n_vectors, int dim, int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vectors) return;

    int cluster = assignments[idx];
    atomicAdd(&cluster_counts[cluster], 1);

    for (int j = 0; j < dim; j++) {
        atomicAdd(&centeroids[cluster * dim + j], vectors[idx * dim + j]);
    }
}

__global__ void normalize_centroids_kernel(float* centroids,
                                           const int* __restrict__ cluster_counts,
                                           int dim, int n_clusters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_clusters) return;

    int count = cluster_counts[idx];
    if (count > 0) {
        for (int j = 0; j < dim; j++) {
            centroids[idx * dim + j] /= count;
        }
    }
}

extern "C" void run_kmeans(const float* d_vectors,
                    float* d_centeroids,
                    int* d_assignments,
                    int n_vectors, int dim, int n_clusters, int max_iter) {
    init_kmeanspp(d_vectors, d_centeroids, n_vectors, dim, n_clusters);

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n_vectors + block.x - 1) / block.x);

    for (int iter = 0; iter < max_iter; iter++) {
        assign_clusters<<<grid, block>>>(d_vectors, d_centeroids, d_assignments, n_vectors, dim, n_clusters);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Centroid update step
        int *d_cluster_counts;
        CUDA_CHECK(cudaMalloc(&d_cluster_counts, n_clusters * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cluster_counts, 0, n_clusters * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_centeroids, 0, n_clusters * dim * sizeof(float)));

        update_centroids_kernel<<<grid, block>>>(d_vectors, d_assignments, d_centeroids, d_cluster_counts, n_vectors, dim, n_clusters);
        
        dim3 centroid_grid((n_clusters + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
        normalize_centroids_kernel<<<centroid_grid, block>>>(d_centeroids, d_cluster_counts, dim, n_clusters);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_cluster_counts));
    }
}