// kernels/distance.cu
#include "../include/distance.cuh"
#include "../include/common.cuh"
#include <float.h>

// ----------------------------------------------------------------
// Brute-force L2 distance kernel
// One thread per vector, one block-row per query.
// Query is cached in shared memory to reduce global memory traffic.
// ----------------------------------------------------------------
__global__ void distance_kernel(const float* __restrict__ queries,
                                const float* __restrict__ vectors,
                                float* __restrict__ distances,
                                int n_vectors, int dim, int n_queries) {
    int vec_idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;
    if (vec_idx >= n_vectors || query_idx >= n_queries) return;

    extern __shared__ float s_query[];
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        s_query[i] = queries[query_idx * dim + i];
    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = s_query[i] - vectors[vec_idx * dim + i];
        sum += diff * diff;
    }
    distances[query_idx * n_vectors + vec_idx] = sum;
}

extern "C" void compute_distances(const float* d_queries,
                                  const float* d_vectors,
                                  float* d_distances,
                                  int n_vectors, int dim, int n_queries) {
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((n_vectors + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, n_queries);
    size_t shared = dim * sizeof(float);
    distance_kernel<<<grid, block, shared>>>(
        d_queries, d_vectors, d_distances, n_vectors, dim, n_queries);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

#define WARP_SIZE       32
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)

__device__ __forceinline__
void heap_sift_down(float* hd, int* hi, int k, int pos) {
    while (true) {
        int l = 2*pos+1, r = 2*pos+2, worst = pos;
        if (l < k && hd[l] > hd[worst]) worst = l;
        if (r < k && hd[r] > hd[worst]) worst = r;
        if (worst == pos) break;
        float tf = hd[pos]; hd[pos] = hd[worst]; hd[worst] = tf;
        int   ti = hi[pos]; hi[pos] = hi[worst]; hi[worst] = ti;
        pos = worst;
    }
}

__device__ __forceinline__
void heap_insert(float* hd, int* hi, int k, float dist, int idx) {
    if (dist >= hd[0]) return;
    hd[0] = dist; hi[0] = idx;
    heap_sift_down(hd, hi, k, 0);
}

__global__ void fused_search_kernel(
    const float* __restrict__ d_queries,
    const float* __restrict__ d_vectors,
    int*   __restrict__ d_out_idx,
    float* __restrict__ d_out_dist,
    int n, int dim, int nq, int k)
{
    extern __shared__ char smem_raw[];
    float* s_query = (float*)smem_raw;
    float* s_dim   = s_query + dim;
    float* s_wdist = s_dim   + THREADS_PER_BLOCK;
    int*   s_widx  = (int*)(s_wdist + WARPS_PER_BLOCK * k);

    int q       = blockIdx.x;
    int tid     = threadIdx.x;
    int lane    = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    if (q >= nq) return;

    for (int i = tid; i < dim; i += blockDim.x)
        s_query[i] = __ldg(&d_queries[(size_t)q * dim + i]);
    __syncthreads();

    float hd[MAX_K]; int hi[MAX_K];
    for (int i = 0; i < k; i++) { hd[i] = FLT_MAX; hi[i] = -1; }

    for (int vec_base = 0; vec_base < n; vec_base += THREADS_PER_BLOCK) {
        int vec_idx = vec_base + tid;
        bool active = (vec_idx < n);
        float dist  = 0.0f;

        for (int db = 0; db < dim; db += THREADS_PER_BLOCK) {
            int dim_idx = db + tid;
            s_dim[tid]  = (active && dim_idx < dim)
                          ? __ldg(&d_vectors[(size_t)vec_idx * dim + dim_idx])
                          : 0.0f;
            __syncthreads();

            if (active) {
                int chunk = min(THREADS_PER_BLOCK, dim - db);
                for (int di = 0; di < chunk; di++) {
                    float diff = s_query[db + di] - s_dim[di];
                    dist += diff * diff;
                }
            }
            __syncthreads();
        }

        if (active) heap_insert(hd, hi, k, dist, vec_idx);
    }

    // Each warp writes its heap to shared
    for (int i = lane; i < k; i += WARP_SIZE) {
        s_wdist[warp_id * k + i] = hd[i];
        s_widx [warp_id * k + i] = hi[i];
    }
    __syncthreads();

    // Thread 0 merges all warp heaps and writes sorted output
    if (tid == 0) {
        float fd[MAX_K]; int fi[MAX_K];
        for (int i = 0; i < k; i++) { fd[i] = FLT_MAX; fi[i] = -1; }

        for (int w = 0; w < WARPS_PER_BLOCK; w++)
            for (int i = 0; i < k; i++)
                heap_insert(fd, fi, k, s_wdist[w*k+i], s_widx[w*k+i]);

        // selection sort → closest first
        for (int i = 0; i < k; i++) {
            int best = i;
            for (int j = i+1; j < k; j++)
                if (fd[j] < fd[best]) best = j;
            float tf = fd[i]; fd[i] = fd[best]; fd[best] = tf;
            int   ti = fi[i]; fi[i] = fi[best]; fi[best] = ti;
            d_out_dist[q * k + i] = fd[i];
            d_out_idx [q * k + i] = fi[i];
        }
    }
}

extern "C" void search(const float* d_queries,
                       const float* d_vectors,
                       int* d_indices,
                       float* d_distances_out,
                       int n, int dim, int nq, int k) {
    if (k > MAX_K) {
        fprintf(stderr, "search: k=%d > MAX_K=%d. Raise MAX_K and recompile.\n",
                k, MAX_K);
        return;
    }
    // shared = query(dim) + tile(TPB) + warp_dist(WARPS*k) + warp_idx(WARPS*k)
    size_t shmem = ((size_t)dim + THREADS_PER_BLOCK + 2 * WARPS_PER_BLOCK * k)
                   * sizeof(float);
    fused_search_kernel<<<nq, THREADS_PER_BLOCK, shmem>>>(
        d_queries, d_vectors, d_indices, d_distances_out, n, dim, nq, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
