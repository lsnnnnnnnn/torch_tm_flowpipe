#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY_DIR}/../.." && pwd)"
CONDA="/srv/local/shengenli/miniforge3/condabin/conda"
OUTPUT_DIR="${1:-$(mktemp -d /tmp/three_tool_deep_study_smoke.XXXXXX)}"
mkdir -p "${OUTPUT_DIR}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.matplotlib"

cd "${REPO_ROOT}"

"${CONDA}" run -n py11 python "${STUDY_DIR}/collect_results.py" \
  --mode initialize --output-dir "${OUTPUT_DIR}"

# Run these suites separately because older experiment directories use
# top-level modules with the same historical name `common`.
"${CONDA}" run -n py11 pytest -q \
  experiments/first_order_followup/tests/test_torch_basis.py
"${CONDA}" run -n diffreach312 pytest -q \
  experiments/first_order_followup/tests/test_diffreach_projection.py
"${CONDA}" run -n py11 pytest -q \
  experiments/three_tool_deep_study/tests

"${CONDA}" run -n py11 python "${STUDY_DIR}/flowstar_correctness.py" \
  --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/flowstar_root_cause.py" \
  --output-dir "${OUTPUT_DIR}/flowstar_root_cause"

"${CONDA}" run -n py11 python "${STUDY_DIR}/run_controlled.py" \
  --tool torch --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n diffreach312 python "${STUDY_DIR}/run_controlled.py" \
  --tool diffreach --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_controlled.py" \
  --tool flowstar --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_controlled.py" \
  --tool collect --output-dir "${OUTPUT_DIR}"

"${CONDA}" run -n py11 python "${STUDY_DIR}/run_native.py" \
  --tool torch --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n diffreach312 python "${STUDY_DIR}/run_native.py" \
  --tool diffreach --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_native.py" \
  --tool flowstar --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_native.py" \
  --tool collect --output-dir "${OUTPUT_DIR}"

"${CONDA}" run -n py11 python "${STUDY_DIR}/run_ablation.py" \
  --mode matched --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_ablation.py" \
  --mode flowstar --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_ablation.py" \
  --mode collect --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/defect_diagnostic.py" \
  --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/bern_feasibility.py" \
  --repetitions 2 --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 pytest -q \
  "${STUDY_DIR}/tests/test_external_exports.py" \
  --torch-segment "${OUTPUT_DIR}/common_segments/torch_tm_flowpipe_complete_total_degree_1_riccati_h0.01.json" \
  --diffreach-segment "${OUTPUT_DIR}/common_segments/diffreach_upstream_affine_flag_riccati_h0.01.json" \
  --flowstar-segment "${OUTPUT_DIR}/common_segments/flowstar_flowstar_root_cause_patch_riccati_h0.01_o2.json"

"${CONDA}" run -n py11 python "${STUDY_DIR}/run_pareto.py" \
  --tool torch --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n diffreach312 python "${STUDY_DIR}/run_pareto.py" \
  --tool diffreach --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_pareto.py" \
  --tool flowstar --smoke --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_pareto.py" \
  --tool collect --output-dir "${OUTPUT_DIR}"

"${CONDA}" run -n py11 python "${STUDY_DIR}/collect_results.py" \
  --mode finalize --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/collect_results.py" \
  --mode tables --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/plot_results.py" \
  --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/generate_report.py" \
  --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/collect_results.py" \
  --mode verify --output-dir "${OUTPUT_DIR}"

printf 'Smoke study passed. Output: %s\n' "${OUTPUT_DIR}"
