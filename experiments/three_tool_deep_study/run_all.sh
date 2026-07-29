#!/usr/bin/env bash
set -euo pipefail

STUDY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${STUDY_DIR}/../.." && pwd)"
CONDA="/srv/local/shengenli/miniforge3/condabin/conda"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${1:-${STUDY_DIR}/results/${TIMESTAMP}}"
PROGRESS_LOG="${OUTPUT_DIR}/progress.log"
mkdir -p "${OUTPUT_DIR}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.matplotlib"

progress() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "${PROGRESS_LOG}"
}

cd "${REPO_ROOT}"
progress "phase0 provenance and initial frozen checksums"
"${CONDA}" run -n py11 python "${STUDY_DIR}/collect_results.py" \
  --mode initialize --output-dir "${OUTPUT_DIR}"

progress "acceptance tests"
"${CONDA}" run -n py11 pytest -q \
  experiments/first_order_followup/tests/test_torch_basis.py
"${CONDA}" run -n diffreach312 pytest -q \
  experiments/first_order_followup/tests/test_diffreach_projection.py
"${CONDA}" run -n py11 pytest -q \
  experiments/three_tool_deep_study/tests

progress "Flowstar correctness matrix and original parity"
"${CONDA}" run -n py11 python "${STUDY_DIR}/flowstar_correctness.py" \
  --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/flowstar_root_cause.py" \
  --output-dir "${OUTPUT_DIR}/flowstar_root_cause"

progress "controlled protocols: Torch"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_controlled.py" \
  --tool torch --output-dir "${OUTPUT_DIR}"
progress "controlled protocols: DiffReach"
"${CONDA}" run -n diffreach312 python "${STUDY_DIR}/run_controlled.py" \
  --tool diffreach --output-dir "${OUTPUT_DIR}"
progress "controlled protocols: Flowstar"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_controlled.py" \
  --tool flowstar --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_controlled.py" \
  --tool collect --output-dir "${OUTPUT_DIR}"

progress "native protocols: Torch"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_native.py" \
  --tool torch --output-dir "${OUTPUT_DIR}"
progress "native protocols: DiffReach"
"${CONDA}" run -n diffreach312 python "${STUDY_DIR}/run_native.py" \
  --tool diffreach --output-dir "${OUTPUT_DIR}"
progress "native protocols: Flowstar"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_native.py" \
  --tool flowstar --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_native.py" \
  --tool collect --output-dir "${OUTPUT_DIR}"

progress "matched basis and component ablations"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_ablation.py" \
  --mode matched --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_ablation.py" \
  --mode flowstar --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_ablation.py" \
  --mode collect --output-dir "${OUTPUT_DIR}"

progress "common defect diagnostics"
"${CONDA}" run -n py11 python "${STUDY_DIR}/defect_diagnostic.py" \
  --output-dir "${OUTPUT_DIR}"

progress "ten-repetition native practical timing: Torch"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_pareto.py" \
  --tool torch --output-dir "${OUTPUT_DIR}"
progress "ten-repetition native practical timing: DiffReach"
"${CONDA}" run -n diffreach312 python "${STUDY_DIR}/run_pareto.py" \
  --tool diffreach --output-dir "${OUTPUT_DIR}"
progress "ten-repetition native practical timing: Flowstar"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_pareto.py" \
  --tool flowstar --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/run_pareto.py" \
  --tool collect --output-dir "${OUTPUT_DIR}"

progress "final frozen checksums, collection, plots, report"
"${CONDA}" run -n py11 python "${STUDY_DIR}/collect_results.py" \
  --mode finalize --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/collect_results.py" \
  --mode tables --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/plot_results.py" \
  --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/generate_report.py" \
  --output-dir "${OUTPUT_DIR}"
"${CONDA}" run -n py11 python "${STUDY_DIR}/collect_results.py" \
  --mode verify --require-ten-repetitions --output-dir "${OUTPUT_DIR}"

date -u +%Y-%m-%dT%H:%M:%SZ > "${OUTPUT_DIR}/RUN_COMPLETE"
progress "complete"
printf 'Full study output: %s\n' "${OUTPUT_DIR}"
