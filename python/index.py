"""
Python wrappers for ArithmosDB GPU indices.

Usage
-----
import numpy as np
from arithmosdb import FlatIndex

# Build
index = FlatIndex(d=128)
index.add(vectors)          # numpy float32 array [n, d]

# Search
distances, indices = index.search(queries, k=10)
"""
from __future__ import annotations

import ctypes
import numpy as np
from .utils import load_library

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = load_library()
    return _lib


def _f32(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous float32."""
    return np.ascontiguousarray(arr, dtype=np.float32)


def _ptr(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.c_void_p)


# ============================================================
# GPU memory helpers (thin ctypes wrappers around cudaMalloc)
# ============================================================
_cuda = ctypes.CDLL("libcudart.so", use_errno=True)
_cuda.cudaMalloc.restype  = ctypes.c_int
_cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cuda.cudaFree.restype    = ctypes.c_int
_cuda.cudaFree.argtypes   = [ctypes.c_void_p]
_cuda.cudaMemcpy.restype  = ctypes.c_int
_cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_size_t, ctypes.c_int]
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2


def _gpu_alloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    err = _cuda.cudaMalloc(ctypes.byref(ptr), nbytes)
    if err:
        raise RuntimeError(f"cudaMalloc failed (error {err})")
    return ptr


def _gpu_free(ptr: ctypes.c_void_p):
    _cuda.cudaFree(ptr)


def _to_gpu(arr: np.ndarray) -> ctypes.c_void_p:
    ptr = _gpu_alloc(arr.nbytes)
    _cuda.cudaMemcpy(ptr, _ptr(arr), arr.nbytes, cudaMemcpyHostToDevice)
    return ptr


def _from_gpu(ptr: ctypes.c_void_p, shape, dtype) -> np.ndarray:
    arr = np.empty(shape, dtype=dtype)
    nbytes = arr.nbytes
    _cuda.cudaMemcpy(_ptr(arr), ptr, nbytes, cudaMemcpyDeviceToHost)
    return arr


# ============================================================
# FlatIndex  —  exact brute-force search
# ============================================================
class FlatIndex:
    """
    Exact nearest-neighbour search using the fused GPU kernel.

    Parameters
    ----------
    d : int
        Vector dimensionality.
    """

    def __init__(self, d: int):
        self.d = d
        self._n = 0
        self._d_vectors = None   # GPU pointer

    def add(self, vectors: np.ndarray):
        """
        Add vectors to the index.  Replaces any existing vectors.

        Parameters
        ----------
        vectors : np.ndarray  shape [n, d], dtype float32
        """
        vectors = _f32(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != self.d:
            raise ValueError(f"Expected shape [n, {self.d}], got {vectors.shape}")

        if self._d_vectors is not None:
            _gpu_free(self._d_vectors)

        self._n = vectors.shape[0]
        self._d_vectors = _to_gpu(vectors)

    def search(self, queries: np.ndarray, k: int = 10):
        """
        Search for the k nearest neighbours of each query.

        Parameters
        ----------
        queries : np.ndarray  shape [nq, d], dtype float32
        k       : int         number of neighbours to return

        Returns
        -------
        distances : np.ndarray  shape [nq, k], float32  (squared L2)
        indices   : np.ndarray  shape [nq, k], int32
        """
        if self._d_vectors is None:
            raise RuntimeError("Index is empty. Call add() first.")

        queries = _f32(queries)
        if queries.ndim == 1:
            queries = queries[None, :]
        nq = queries.shape[0]

        d_queries   = _to_gpu(queries)
        d_indices   = _gpu_alloc(nq * k * 4)
        d_distances = _gpu_alloc(nq * k * 4)

        _get_lib().search(
            d_queries, self._d_vectors,
            d_indices, d_distances,
            self._n, self.d, nq, k,
        )

        indices   = _from_gpu(d_indices,   (nq, k), np.int32)
        distances = _from_gpu(d_distances, (nq, k), np.float32)

        _gpu_free(d_queries)
        _gpu_free(d_indices)
        _gpu_free(d_distances)

        return distances, indices

    def __len__(self):
        return self._n

    def __del__(self):
        if self._d_vectors is not None:
            _gpu_free(self._d_vectors)

    def __repr__(self):
        return f"FlatIndex(d={self.d}, n={self._n})"


# ============================================================
# IVFIndex  —  approximate search (build not yet exposed here;
#              use the C API directly for IVF until Python
#              bindings are extended in a future PR)
# ============================================================
class IVFIndex:
    """
    Placeholder for IVF approximate search.
    Full Python bindings coming in v0.2.
    """

    def __init__(self, d: int, nlist: int = 0):
        self.d = d
        self.nlist = nlist
        raise NotImplementedError(
            "IVFIndex Python bindings are not yet implemented. "
            "Use FlatIndex for now, or call the C API directly."
        )
