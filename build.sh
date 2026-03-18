#!/usr/bin/env bash
# ============================================================
# build.sh — compile ArithmosDB CUDA shared library
#
# Usage:
#   bash build.sh              # auto-detect GPU arch
#   bash build.sh sm_75        # override arch
# ============================================================
set -e

ARCH=${1:-$(python3 -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader']).decode()
cap = out.strip().split('\n')[0].replace('.','')
print('sm_' + cap)
" 2>/dev/null || echo "sm_75")}

echo "Building for arch: $ARCH"

mkdir -p build

SOURCES=(
    kernels/distance.cu
    kernels/kmeans.cu
    kernels/ivf_build.cu
    kernels/ivf_search.cu
    "kernels/top-k selection.cu"
    kernels/utils.cu
)

nvcc -O3 -arch=$ARCH \
     -Xcompiler -fPIC \
     --shared \
     "${SOURCES[@]}" \
     -I include \
     -o build/libvectordb.so

echo "✅ Built: build/libvectordb.so"
ls -lh build/libvectordb.so
