#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define MAX_ITER 100
#define THREADS_PER_BLOCK 256

__global__ void assign_clusters(const float* __restrict__ vectors,
                                const float* __restrict__ centeroids,
                                int *assignments,
                                float *new_sums,
                                int *counts,
                                int n, int d, int k) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n) return;

    float min_dist = FLT_MAX;
    int best_cluster = 0;

    // find the closest centeroid
    for (int cluster = 0; cluster < k; cluster++) {
        float dist = 0.0f;
        for (int dim = 0; dim < d; dim++) {
            float diff = vectors[vec_idx * d + dim] - centeroids[cluster * d + dim];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = cluster;
        }
    }

    // assign the vector to the closest cluster
    assignments[vec_idx] = best_cluster;

    // atomic update of the centeroid sums and counts
    for (int dim = 0; dim < d; dim++) {
        atomicAdd(&new_sums[best_cluster * d + dim], vectors[vec_idx * d + dim]);
    }
    atomicAdd(&counts[best_cluster], 1);
}

// Update centeroids by dividing the sums by the counts
__global__ void update_centeroids(float* __restrict__ centeroids, 
                                const float *__restrict__ new_sums, 
                                int *counts,
                                int k, int d){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;

    if (counts[idx] > 0) {
        for (int dim = 0; dim < d; dim++) {
            centeroids[idx * d + dim] = new_sums[idx * d + dim] / counts[idx];
        }
    }
}

// host wrapper
extern "C" void kmeans(const float* d_vectors,
                        float* d_centeroids,
                        int *d_assignments,
                        int n, int d, int k,
                        int max_iter) {
    float *d_new_sums;
    int *d_counts;
    cudaMalloc(&d_new_sums, k * d * sizeof(float));
    cudaMalloc(&d_counts, k * sizeof(int));

    dim3 blocksize(THREADS_PER_BLOCK);
    dim3 gridSize((k + blocksize.x - 1) / blocksize.x);
    for (int iter = 0; iter < max_iter; iter++) {
        // reset new_sums and counts
        cudaMemset(d_new_sums, 0, k * d * sizeof(float));
        cudaMemset(d_counts, 0, k * sizeof(int));

        // assign clusters and update sums/counts
        assign_clusters<<<gridSize, blocksize>>>(d_vectors, d_centeroids, d_assignments, d_new_sums, d_counts, n, d, k);
        cudaDeviceSynchronize();

        // update centeroids
        update_centeroids<<<gridSize, blocksize>>>(d_centeroids, d_new_sums, d_counts, k, d);
        cudaDeviceSynchronize();
    }
    cudaFree(d_new_sums);
    cudaFree(d_counts);
}