#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/srv/local/shengenli/xiangru_adoption_20260907T032448Z/candidate_env/bin:$CUDA_HOME/bin:$PATH
export TORCH_EXTENSIONS_DIR=/srv/local/shengenli/strict_backend_reevaluation_20260907T061907Z/extensions_A
export TORCHINDUCTOR_CACHE_DIR=/srv/local/shengenli/strict_backend_reevaluation_20260907T061907Z/inductor_A
export TRITON_CACHE_DIR=/srv/local/shengenli/strict_backend_reevaluation_20260907T061907Z/triton_A
export HYPOTHESIS_STORAGE_DIRECTORY=/srv/local/shengenli/strict_backend_reevaluation_20260907T061907Z/hypothesis_A
export TORCH_CUDA_ARCH_LIST=7.0
export FLOWSTAR_CUDA_HOST_COMPILER=/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-g++
export CXX=$FLOWSTAR_CUDA_HOST_COMPILER
export CC=/srv/local/shengenli/.huan-audit-gxx13/bin/x86_64-conda-linux-gnu-gcc
export MAX_JOBS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset FLOWSTAR_NO_CUDA_KERNEL
exec /srv/local/shengenli/xiangru_adoption_20260907T032448Z/candidate_env/bin/python "$@"
