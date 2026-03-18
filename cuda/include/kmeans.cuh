// include/kmeans.cuh
#pragma once
#include "common.cuh"

extern "C" void init_kmeanspp(const float* d_vectors,
                              float* d_centroids,
                              int n_vectors, int dim, int n_clusters);

extern "C" void run_kmeans(const float* d_vectors,
                           float* d_centroids,
                           int* d_assignments,
                           int n_vectors, int dim, int n_clusters,
                           int max_iter);
