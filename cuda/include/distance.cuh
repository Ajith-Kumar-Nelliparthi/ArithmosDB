// include/distance.cuh
#pragma once
#include "common.cuh"

// Brute-force distance kernel (unfused, writes nq*n distance matrix)
__global__ void distance_kernel(const float* __restrict__ queries,
                                const float* __restrict__ vectors,
                                float* __restrict__ distances,
                                int n_vectors, int dim, int n_queries);

// Compute full distance matrix; output shape [n_queries, n_vectors]
extern "C" void compute_distances(const float* d_queries,
                                  const float* d_vectors,
                                  float* d_distances,
                                  int n_vectors, int dim, int n_queries);

extern "C" void search(const float* d_queries,
                       const float* d_vectors,
                       int* d_indices,
                       float* d_distances_out,
                       int n, int dim, int nq, int k);
