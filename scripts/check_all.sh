#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

echo "[check] pytest"
pytest -q

echo "[check] examples and experiments"
python -u scripts/check_runtime.py

echo "[check] Flow* comparison smoke (Flow* optional)"
bash scripts/check_flowstar_comparison.sh

echo "[check] done"
