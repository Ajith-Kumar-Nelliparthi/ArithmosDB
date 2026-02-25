// include/distance.cuh
#pragma once

#include "common.cuh"

__global__ void distance_kernel(const float* __restrict__ queries,
                                const float* __restrict__ vectors,
                                float* __restrict__ distances,
                                int n_vectors, int dim, int n_queries);

extern "C" void compute_distances(const float* d_queries,
                                    const float* d_vectors,
                                    float* d_distances,
                                    int n_vectors, int dim, int n_queries);