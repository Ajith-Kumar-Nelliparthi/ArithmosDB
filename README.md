# ArithmosDB
![alt text](assets/arithmosdb-logo.png)
GPU-accelerated vector database built from scratch in CUDA. Custom kernels, no dependencies on FAISS or cuVS.

---

## Benchmark results (Tesla T4, k=10, 100% recall)

| Vectors | Dimensions | QPS | Latency |
|---------|------------|-----|---------|
| 1,000 | 128 | 191,422 | 0.52 ms |
| 10,000 | 128 | 41,963 | 2.38 ms |
| 100,000 | 128 | 6,636 | 15.07 ms |
| 10,000 | 256 | 59,137 | 1.69 ms |
| 10,000 | 512 | 34,603 | 2.89 ms |

---

## Project structure

```
ARITHMOSDB/
├── cuda/
│   ├── include/
│   │   ├── common.cuh          CUDA_CHECK, THREADS_PER_BLOCK, MAX_K, IVFIndex struct
│   │   ├── distance.cuh        compute_distances() + search() declarations
│   │   ├── ivf_index.cuh       build_ivf() + search_ivf() declarations
│   │   └── kmeans.cuh          run_kmeans() + init_kmeanspp() declarations
│   └── kernels/
│       ├── distance.cu         fused distance + top-k (warp-per-vector)
│       ├── kmeans.cu           k-means++ init + Lloyd iterations
│       ├── ivf_build.cu        build_ivf: k-means → reorder → prefix sum
│       ├── ivf_search.cu       search_ivf: coarse centroid search + fine top-k
│       ├── top-k selection.cu  standalone topk on precomputed distance matrix
│       └── utils.cu            transpose, normalise_vectors, cosine_similarity
├── python/
│   └── arithmosdb/
│       ├── __init__.py
│       ├── index.py            FlatIndex Python wrapper (ctypes)
│       └── utils.py            library loader + ctypes signatures
├── tests/
│   ├── conftest.py
│   ├── test_flat_index.py      unit + correctness tests vs NumPy
│   └── test_correctness.py     recall@k validation vs FAISS
├── notebooks/
│   └── ArithmosDB_demo.ipynb   Colab demo: build → correctness → benchmark
├── assets/
├── setup.py
└── build.sh
```

---

## Quick start

### 1. Build the shared library

```bash
bash build.sh
```

Or manually:

```bash
mkdir -p build
nvcc -O3 -arch=sm_75 -Xcompiler -fPIC --shared \
    cuda/kernels/distance.cu \
    cuda/kernels/kmeans.cu \
    cuda/kernels/ivf_build.cu \
    cuda/kernels/ivf_search.cu \
    "cuda/kernels/top-k selection.cu" \
    cuda/kernels/utils.cu \
    -I cuda/include \
    -o build/libvectordb.so
```

### 2. Install the Python package

```bash
pip install -e .
```

### 3. Use

```python
import numpy as np
from arithmosdb import FlatIndex

index = FlatIndex(d=128)
index.add(np.random.rand(10_000, 128).astype("float32"))

distances, indices = index.search(
    np.random.rand(100, 128).astype("float32"),
    k=10
)
```

### 4. Run tests

```bash
pytest tests/test_flat_index.py -v

# Requires: pip install faiss-cpu
pytest tests/test_correctness.py -v
```

### 5. Colab demo

Open `notebooks/ArithmosDB_demo.ipynb` with a T4 GPU runtime.

---

## How the search kernel works

The core kernel (`cuda/kernels/distance.cu: fused_search_kernel`) uses a **warp-per-vector** design:

```
One block = one query
One warp  = one vector at a time

For each vector:
  32 lanes each own dim/32 dimensions
  → fully coalesced global memory reads
  → warp shuffle reduction sums 32 partial distances into lane 0
  → lane 0 updates the warp's register heap

After all vectors:
  8 warp heaps in shared memory
  → parallel reduction tree merges them in log2(8) = 3 steps
  → thread 0 writes sorted top-k to output
```

Key properties:
- No intermediate `nq × n` distance matrix — distances stay in registers
- `__ldg()` read-only cache for vector loads
- Shared memory usage: `dim*4 + WARPS*k*8` bytes (1152 bytes for dim=128, k=10)
- 100% recall on all tested configurations

### Performance evolution

| Version | Design | 10K×128 QPS |
|---------|--------|-------------|
| v1 | Separate distance kernel + bubble sort top-k | ~8K |
| v2 | Bug fixes (wrong memcpy, missing sync, name mismatch) | ~9K |
| v3 | Fused kernel, thread-per-vector | ~22K |
| v4 | Warp-per-vector + shuffle reduction | **42K** |

---

## Kernel reference

| Function | File | Description |
|----------|------|-------------|
| `search()` | distance.cu | Fused brute-force search, returns top-k directly |
| `compute_distances()` | distance.cu | Writes full `[nq, n]` distance matrix |
| `build_ivf()` | ivf_build.cu | Builds IVF index: k-means → inverted lists |
| `search_ivf()` | ivf_search.cu | Approximate search with coarse + fine stages |
| `run_kmeans()` | kmeans.cu | k-means++ init + Lloyd iterations, all on GPU |
| `normalise_vectors()` | utils.cu | In-place L2 normalisation for cosine similarity |
| `cosine_similarity()` | utils.cu | Cosine distance (1 − dot product) |
| `transpose()` | utils.cu | Bank-conflict-free matrix transpose |

---

## Roadmap

- [ ] IVFIndex Python bindings
- [ ] `float4` vectorised loads
- [ ] Multi-query block tiling (increase SM occupancy for large nq)
- [ ] HNSW index
- [ ] Multi-GPU support
- [ ] REST API server
