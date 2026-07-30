#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA="${CONDA_EXE:-$(command -v conda)}"
DEEP_RESULTS="${DEEP_STUDY_RESULTS_DIR:-}"

cd "${REPO_ROOT}"

# Historical experiment directories intentionally contain top-level modules
# with repeated names such as common.py.  Isolated pytest processes are the
# complete-test contract; every test file is still collected.
"${CONDA}" run -n py11 python -m pytest -q tests
"${CONDA}" run -n py11 python -m pytest -q \
  experiments/first_order_followup/tests/test_torch_basis.py
"${CONDA}" run -n diffreach312 python -m pytest -q \
  experiments/first_order_followup/tests/test_diffreach_projection.py \
  experiments/first_order_followup/tests/test_diffreach_parity.py
"${CONDA}" run -n py11 python -m pytest -q \
  experiments/first_order_three_way/tests/test_benchmark.py
"${CONDA}" run -n diffreach312 python -m pytest -q \
  experiments/first_order_three_way/tests/test_diffreach_support.py
"${CONDA}" run -n py11 python -m pytest -q \
  experiments/three_way_common_contract/tests
"${CONDA}" run -n py11 python -m pytest -q \
  experiments/three_way_comparison_repair/tests

if [[ -n "${DEEP_RESULTS}" ]]; then
  "${CONDA}" run -n py11 python -m pytest -q \
    experiments/three_tool_deep_study/tests \
    --torch-segment \
    "${DEEP_RESULTS}/common_segments/torch_tm_flowpipe_complete_total_degree_1_riccati_h0.01.json" \
    --diffreach-segment \
    "${DEEP_RESULTS}/common_segments/diffreach_upstream_affine_flag_riccati_h0.01.json" \
    --flowstar-segment \
    "${DEEP_RESULTS}/common_segments/flowstar_flowstar_root_cause_patch_riccati_h0.01_o2.json"
else
  "${CONDA}" run -n py11 python -m pytest -q \
    experiments/three_tool_deep_study/tests
fi

printf 'Complete isolated pytest matrix passed.\n'
