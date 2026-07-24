#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${THREE_WAY_OUTPUT_DIR:-$HERE/results/$TIMESTAMP}"

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/plots"
cp "$HERE/benchmark_spec.yaml" "$OUTPUT_DIR/benchmark_spec.yaml"
cd "$REPO_ROOT"

conda run -n py11 python "$HERE/run_torch.py" --output-dir "$OUTPUT_DIR" \
  >"$OUTPUT_DIR/logs/torch.log" 2>&1
conda run -n diffreach312 python "$HERE/run_diffreach.py" --output-dir "$OUTPUT_DIR" \
  >"$OUTPUT_DIR/logs/diffreach.log" 2>&1
conda run -n py11 python "$HERE/run_flowstar.py" --output-dir "$OUTPUT_DIR" \
  >"$OUTPUT_DIR/logs/flowstar.log" 2>&1
conda run -n py11 python "$HERE/collect_results.py" --output-dir "$OUTPUT_DIR" --strict \
  >"$OUTPUT_DIR/logs/collect.log" 2>&1
MPLCONFIGDIR="$OUTPUT_DIR/.matplotlib" conda run -n py11 python "$HERE/plot_results.py" \
  --output-dir "$OUTPUT_DIR" >"$OUTPUT_DIR/logs/plot.log" 2>&1
conda run -n py11 python "$HERE/generate_report.py" --output-dir "$OUTPUT_DIR" \
  >"$OUTPUT_DIR/logs/report.log" 2>&1

echo "$OUTPUT_DIR"
