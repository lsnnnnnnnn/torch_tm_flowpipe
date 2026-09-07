#!/usr/bin/env bash
set -euo pipefail
RUN_DIR=/srv/local/shengenli/xiangru_adoption_20260907T032448Z
export PYTHONPATH="$RUN_DIR/xiangru_upstream/src"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$RUN_DIR/candidate_env/bin:$CUDA_HOME/bin:$PATH"
export TORCH_EXTENSIONS_DIR="$RUN_DIR/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_DIR/torchinductor"
export TRITON_CACHE_DIR="$RUN_DIR/triton_cache"
export HYPOTHESIS_STORAGE_DIRECTORY="$RUN_DIR/hypothesis_cache"
export TORCH_CUDA_ARCH_LIST=7.0
export FLOWSTAR_CUDA_HOST_COMPILER=/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-g++
export CXX="$FLOWSTAR_CUDA_HOST_COMPILER"
export CC=/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-gcc
export MAX_JOBS=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset FLOWSTAR_NO_CUDA_KERNEL
exec "$RUN_DIR/candidate_env/bin/python" "$@"
