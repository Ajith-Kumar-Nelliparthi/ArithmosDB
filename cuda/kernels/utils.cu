// kernels/utils.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#include "../include/common.cuh"

// Transpose  [m, n] → [n, m]  with shared-memory tiling
__global__ void transpose_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int m, int n) {
    __shared__ float tile[32][33]; // +1 column padding kills bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < n && y < m)
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    __syncthreads();

    // transposed write
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    if (x < m && y < n)
        output[y * m + x] = tile[threadIdx.x][threadIdx.y];
}

extern "C" void transpose(const float* d_input, float* d_output, int m, int n) {
    dim3 block(32, 32);
    dim3 grid((n + 31) / 32, (m + 31) / 32);
    transpose_kernel<<<grid, block>>>(d_input, d_output, m, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------
// Normalise vectors in-place  (for cosine similarity)
// ----------------------------------------------------------------
__global__ void normalize_kernel(float* __restrict__ vectors, int n, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float norm = 0.0f;
    for (int i = 0; i < d; i++) {
        float v = vectors[idx * d + i];
        norm += v * v;
    }
    norm = sqrtf(norm);

    if (norm > 1e-10f) {
        float inv = 1.0f / norm;
        for (int i = 0; i < d; i++)
            vectors[idx * d + i] *= inv;
    }
}

extern "C" void normalise_vectors(float* d_vectors, int n, int d) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    normalize_kernel<<<grid, block>>>(d_vectors, n, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------
// Cosine similarity  (assumes pre-normalised vectors)
// ----------------------------------------------------------------
__global__ void cosine_similarity_kernel(const float* __restrict__ queries,
                                          const float* __restrict__ vectors,
                                          float* __restrict__ similarities,
                                          int n, int d, int nq) {
    int vec_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;
    if (vec_idx >= n || query_idx >= nq) return;

    extern __shared__ float s_query[];
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        s_query[i] = queries[query_idx * d + i];
    __syncthreads();

    float dot = 0.0f;
    for (int i = 0; i < d; i++)
        dot += vectors[vec_idx * d + i] * s_query[i];

    // store as distance (1 - similarity) so top-k kernels
    similarities[query_idx * n + vec_idx] = 1.0f - dot;
}

extern "C" void cosine_similarity(const float* d_queries,
                                  const float* d_vectors,
                                  float* d_similarities,
                                  int n, int d, int nq) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, nq);
    size_t shared = d * sizeof(float);
    cosine_similarity_kernel<<<grid, block, shared>>>(
        d_queries, d_vectors, d_similarities, n, d, nq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void topk_selection(const float* __restrict__ distances,
                               int* __restrict__ indices,
                               float* __restrict__ top_distances,
                               int n, int k, int nq) {
    int query_idx = blockIdx.x;
    if (query_idx >= nq) return;

    extern __shared__ char shared_mem[];
    float* s_dist = (float*)shared_mem;
    int*   s_idx  = (int*)(shared_mem + n * sizeof(float));

    int tid = threadIdx.x;
    const float* qd = distances + query_idx * n;

    for (int i = tid; i < n; i += blockDim.x) {
        s_dist[i] = qd[i];
        s_idx [i] = i;
    }
    __syncthreads();

    for (int i = 0; i < k; i++) {
        for (int j = tid; j < n - 1 - i; j += blockDim.x) {
            if (s_dist[j] > s_dist[j + 1]) {
                float td = s_dist[j]; s_dist[j] = s_dist[j+1]; s_dist[j+1] = td;
                int   ti = s_idx [j]; s_idx [j] = s_idx [j+1]; s_idx [j+1] = ti;
            }
        }
        __syncthreads();
    }

    if (tid < k) {
        indices      [query_idx * k + tid] = s_idx [tid];
        top_distances[query_idx * k + tid] = s_dist[tid];
    }
}

extern "C" void find_topk(const float* d_distances,
                          int* d_indices,
                          float* d_top_distances,
                          int n, int k, int nq) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(nq);
    size_t shared = (size_t)n * (sizeof(float) + sizeof(int));
    topk_selection<<<grid, block, shared>>>(
        d_distances, d_indices, d_top_distances, n, k, nq);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
