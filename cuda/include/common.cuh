// include/common.cuh
#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
} while (0)

struct  IVFIndex {
    float* d_centeroids;  // nlist * d
    float* d_inverted_vectors;
    int* d_inverted_ids;
    int* d_list_offsets;
    int* d_list_sizes;
    int nlist;
    int n_vectors;
    int dim;
};
