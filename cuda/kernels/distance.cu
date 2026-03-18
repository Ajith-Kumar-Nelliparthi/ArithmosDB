// cuda/kernels/distance.cu
#include "../include/common.cuh"
#include "../include/distance.cuh"
#include <float.h>

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
    dim3 grid((n_vectors + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, n_queries);
    distance_kernel<<<grid, block, dim*sizeof(float)>>>(
        d_queries, d_vectors, d_distances, n_vectors, dim, n_queries);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / 32)

__device__ __forceinline__
void warp_insert(float* hd, int* hi, int k, float dist, int idx) {
    if (dist >= hd[k-1]) return;
    hd[k-1] = dist; hi[k-1] = idx;
    for (int j = k-1; j > 0 && hd[j] < hd[j-1]; j--) {
        float tf = hd[j]; hd[j] = hd[j-1]; hd[j-1] = tf;
        int   ti = hi[j]; hi[j] = hi[j-1]; hi[j-1] = ti;
    }
}

__global__ void fused_search_kernel(
    const float* __restrict__ d_queries,
    const float* __restrict__ d_vectors,
    int*   __restrict__ d_out_idx,
    float* __restrict__ d_out_dist,
    int n, int dim, int nq, int k)
{
    extern __shared__ char smem[];
    float* s_query = (float*)smem;
    float* s_dist  = (float*)(smem + dim * sizeof(float));
    int*   s_idx   = (int*)  (smem + dim * sizeof(float)
                                   + WARPS_PER_BLOCK * k * sizeof(float));

    int q       = blockIdx.x;
    int tid     = threadIdx.x;
    int lane    = tid & 31;
    int warp_id = tid >> 5;
    if (q >= nq) return;

    // Load query into shared — all threads participate, coalesced
    for (int i = tid; i < dim; i += blockDim.x)
        s_query[i] = __ldg(&d_queries[(size_t)q * dim + i]);
    __syncthreads();

    // Initialise each warp's heap (lane 0 only)
    if (lane == 0) {
        for (int i = 0; i < k; i++) {
            s_dist[warp_id * k + i] = FLT_MAX;
            s_idx [warp_id * k + i] = -1;
        }
    }
    __syncthreads();

    // Each warp processes its stripe of vectors
    // 32 lanes split the dim dimensions: lane j handles dims j, j+32, j+64...
    for (int vi = warp_id; vi < n; vi += WARPS_PER_BLOCK) {
        const float* vec = d_vectors + (size_t)vi * dim;

        // Each lane accumulates partial squared distance over its dimensions
        float partial = 0.0f;
        for (int j = lane; j < dim; j += 32) {
            float diff = s_query[j] - __ldg(&vec[j]);
            partial += diff * diff;
        }

        // Warp shuffle reduction — sum all 32 lanes into lane 0
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);

        // Lane 0 has the full distance — update this warp's heap
        if (lane == 0)
            warp_insert(s_dist + warp_id * k,
                        s_idx  + warp_id * k,
                        k, partial, vi);
    }
    __syncthreads();

    for (int stride = WARPS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            int partner = tid + stride;
            for (int i = 0; i < k; i++) {
                warp_insert(s_dist + tid * k,
                            s_idx  + tid * k,
                            k,
                            s_dist[partner * k + i],
                            s_idx [partner * k + i]);
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the final top-k (already sorted ascending)
    if (tid == 0) {
        for (int i = 0; i < k; i++) {
            d_out_dist[q * k + i] = s_dist[i];
            d_out_idx [q * k + i] = s_idx [i];
        }
    }
}

extern "C" void search(const float* d_queries,
                       const float* d_vectors,
                       int* d_indices,
                       float* d_distances_out,
                       int n, int dim, int nq, int k) {
    if (k > MAX_K) {
        fprintf(stderr, "search: k=%d > MAX_K=%d. Raise MAX_K in common.cuh and recompile.\n",
                k, MAX_K);
        return;
    }
    // shared = query(dim) + per-warp heaps(WARPS*k) x2
    size_t shmem = dim * sizeof(float)
                 + WARPS_PER_BLOCK * k * sizeof(float)
                 + WARPS_PER_BLOCK * k * sizeof(int);
    fused_search_kernel<<<nq, THREADS_PER_BLOCK, shmem>>>(
        d_queries, d_vectors, d_indices, d_distances_out, n, dim, nq, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
