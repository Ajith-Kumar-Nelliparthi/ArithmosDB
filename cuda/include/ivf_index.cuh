// include/ivf_index.cuh
#pragma once

#include "common.cuh"

extern "C" void build_ivf(const float* d_vectors,
                          IVFIndex* index,
                          int n_vectors, int dim, int nlist = 0, int max_iter = 25);

extern "C" void search_ivf(const float* d_queries,
                           const IVFIndex* index,
                           int* d_indices,
                           float* d_distances,
                           int n_queries, int k, int nprobe);