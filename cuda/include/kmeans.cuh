// include/kmeans.cuh
#pragma once

#include "distance.cuh"

extern "C" void init_kmeans(const float* d_vectors,
                            float* d_centeroids,
                            int n_vectors, int dim, int n_clusters);

extern "C" void run_kmeans(const float* d_vectors,
                            float* d_centeroids,
                            int* d_assignments,
                            int n_vectors, int dim, int n_clusters,
                            int max_iter = 25);