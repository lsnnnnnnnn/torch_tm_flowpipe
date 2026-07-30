#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ARGUMENT=()
if [[ $# -gt 0 ]]; then
  OUTPUT_ARGUMENT=(--output-dir "$1")
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" experiments/consolidated_study/cli.py \
  smoke "${OUTPUT_ARGUMENT[@]}"
