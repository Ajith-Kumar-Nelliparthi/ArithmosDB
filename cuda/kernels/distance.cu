// kernels/distance.cu
#include "../include/common.cuh"
#include "../include/distance.cuh"
#include <float.h>

// ----------------------------------------------------------------
// Brute-force distance kernel (unfused, writes full nq*n matrix)
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
    distance_kernel<<<grid, block, dim * sizeof(float)>>>(
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
    // shared: query vector + warp heaps
    extern __shared__ char smem[];
    float* s_query = (float*)smem;
    float* s_wdist = (float*)(smem + dim * sizeof(float));
    int*   s_widx  = (int*)(smem + dim * sizeof(float)
                                 + WARPS_PER_BLOCK * k * sizeof(float));

    int q       = blockIdx.x;
    int tid     = threadIdx.x;
    int lane    = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    if (q >= nq) return;

    // Load query into shared
    for (int i = tid; i < dim; i += blockDim.x)
        s_query[i] = __ldg(&d_queries[(size_t)q * dim + i]);
    __syncthreads();

    // Each thread maintains a max-heap over its assigned vectors
    float hd[MAX_K]; int hi[MAX_K];
    for (int i = 0; i < k; i++) { hd[i] = FLT_MAX; hi[i] = -1; }

    // Thread tid owns vectors: tid, tid+TPB, tid+2*TPB, ...
    // Read dimensions directly — no shared staging needed
    for (int vec_idx = tid; vec_idx < n; vec_idx += blockDim.x) {
        float dist = 0.0f;
        const float* vec = d_vectors + (size_t)vec_idx * dim;
        for (int j = 0; j < dim; j++) {
            float diff = s_query[j] - __ldg(&vec[j]);
            dist += diff * diff;
        }
        heap_insert(hd, hi, k, dist, vec_idx);
    }

    // Write each thread's heap to its warp's shared slot
    // (only lane 0..k-1 matter; we write all k per warp via striding)
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

        // Selection sort → closest first
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
    // shared = s_query(dim) + s_wdist(WARPS*k) + s_widx(WARPS*k)
    size_t shmem = dim * sizeof(float)
                 + WARPS_PER_BLOCK * k * sizeof(float)
                 + WARPS_PER_BLOCK * k * sizeof(int);

    fused_search_kernel<<<nq, THREADS_PER_BLOCK, shmem>>>(
        d_queries, d_vectors, d_indices, d_distances_out, n, dim, nq, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
