"""
Utility: locate and load the compiled libvectordb.so shared library.
"""
import ctypes
import os
from pathlib import Path


def _find_lib() -> Path:
    """Search standard locations for libvectordb.so."""
    candidates = [
        # built in-tree
        Path(__file__).parent.parent.parent / "build" / "libvectordb.so",
        # installed alongside the package
        Path(__file__).parent / "libvectordb.so",
        # system / LD_LIBRARY_PATH
        Path("libvectordb.so"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        "libvectordb.so not found. "
        "Build the project first:\n"
        "  cd ARITHMOSDB && mkdir -p build && cd build\n"
        "  nvcc -O3 -arch=sm_75 -Xcompiler -fPIC --shared "
        "../kernels/*.cu -I../include -o libvectordb.so"
    )


def load_library() -> ctypes.CDLL:
    lib_path = _find_lib()
    lib = ctypes.CDLL(str(lib_path))

    # ---- compute_distances ----
    lib.compute_distances.restype = None
    lib.compute_distances.argtypes = [
        ctypes.c_void_p,  # queries
        ctypes.c_void_p,  # vectors
        ctypes.c_void_p,  # distances
        ctypes.c_int,     # n
        ctypes.c_int,     # d
        ctypes.c_int,     # nq
    ]

    # ---- search (brute-force) ----
    lib.search.restype = None
    lib.search.argtypes = [
        ctypes.c_void_p,  # queries
        ctypes.c_void_p,  # vectors
        ctypes.c_void_p,  # indices  out
        ctypes.c_void_p,  # distances out
        ctypes.c_int,     # n
        ctypes.c_int,     # d
        ctypes.c_int,     # nq
        ctypes.c_int,     # k
    ]

    # ---- run_kmeans ----
    lib.run_kmeans.restype = None
    lib.run_kmeans.argtypes = [
        ctypes.c_void_p,  # vectors
        ctypes.c_void_p,  # centroids
        ctypes.c_void_p,  # assignments
        ctypes.c_int,     # n
        ctypes.c_int,     # d
        ctypes.c_int,     # k
        ctypes.c_int,     # max_iter
    ]

    return lib
