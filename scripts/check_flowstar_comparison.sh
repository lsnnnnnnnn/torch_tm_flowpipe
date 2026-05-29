#!/usr/bin/env bash
set -euo pipefail

python comparisons/flowstar/compare_against_torch_tm.py \
  --csv outputs/flowstar_comparison_smoke.csv \
  --model-dir outputs/flowstar_models_smoke \
  --skip-flowstar \
  --no-plots

python comparisons/flowstar/summarize_comparison.py \
  outputs/flowstar_comparison_smoke.csv \
  --out outputs/flowstar_comparison_smoke_summary.md
