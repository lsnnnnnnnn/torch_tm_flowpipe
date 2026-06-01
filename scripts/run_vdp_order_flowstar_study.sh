#!/usr/bin/env bash
set -euo pipefail

FLOWSTAR_ROOT=${FLOWSTAR_ROOT:-${1:-/srv/local/shengenli/flowstar}}
CONDA_ENV=${CONDA_ENV:-py11}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

run_py() {
  conda run -n "$CONDA_ENV" python "$@"
}

run_py -m pip install -e ".[test]"
run_py experiments/tm_order_audit.py --system van_der_pol --orders 2 3 4 5 6 7 8 --h 0.0025 0.005 0.01 --steps 1 5 10 --csv outputs/tm_order_audit_vdp_order2_8.csv
run_py experiments/diagnose_van_der_pol.py --orders 2 3 4 5 6 7 8 --h-values 0.0025 0.005 0.01 --steps-values 1 5 10 --csv outputs/van_der_pol_diagnostics_by_order_v2.csv

settings=(
  "1e-10 1e-15 rem1e-10_cut1e-15"
  "1e-8 1e-15 rem1e-8_cut1e-15"
  "1e-6 1e-12 rem1e-6_cut1e-12"
  "1e-4 1e-10 rem1e-4_cut1e-10"
)

for spec in "${settings[@]}"; do
  read -r rem cutoff label <<< "$spec"
  run_py comparisons/flowstar/compare_against_torch_tm.py \
    --systems van_der_pol \
    --orders 2 3 4 5 6 7 8 \
    --h-values 0.0025 0.005 0.01 \
    --steps-values 1 5 10 \
    --flowstar-root "$FLOWSTAR_ROOT" \
    --flowstar-remainder-radius "$rem" \
    --flowstar-cutoff "$cutoff" \
    --flowstar-setting-label "$label" \
    --flowstar-only \
    --csv "outputs/flowstar_vdp_by_order_${label}_v2.csv" \
    --model-dir "outputs/flowstar_models_${label}" \
    --flowstar-timeout-s 120 \
    --no-plots
done

run_py - <<'PY2'
import csv
from pathlib import Path
files = sorted(Path('outputs').glob('flowstar_vdp_by_order_*_v2.csv'))
out = Path('outputs/flowstar_vdp_remainder_cutoff_sweep.csv')
rows = []
fields = None
for path in files:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames if fields is None else fields
        rows.extend(reader)
out.parent.mkdir(exist_ok=True)
with out.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields or [])
    writer.writeheader()
    writer.writerows(rows)
print(f'wrote {out}')
PY2

run_py comparisons/flowstar/compare_against_torch_tm.py \
  --systems van_der_pol \
  --orders 2 3 4 5 6 7 8 \
  --h-values 0.01 \
  --steps-values 10 \
  --skip-flowstar \
  --csv outputs/flowstar_vdp_torch_h0.01_s10_v2.csv \
  --model-dir outputs/flowstar_models_torch_h0.01_s10 \
  --no-plots

run_py - <<'PY3'
import csv
from pathlib import Path
rows = []
fields = None
for path in [Path('outputs/flowstar_vdp_torch_h0.01_s10_v2.csv'), Path('outputs/flowstar_vdp_remainder_cutoff_sweep.csv')]:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames if fields is None else fields
        for row in reader:
            if path.name.startswith('flowstar_vdp_torch') and row.get('tool') != 'torch_tm_flowpipe':
                continue
            rows.append(row)
out = Path('outputs/flowstar_vdp_plot_input_v2.csv')
with out.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields or [])
    writer.writeheader()
    writer.writerows(rows)
print(f'wrote {out}')
PY3

run_py experiments/plot_order_results.py \
  --diagnostics-csv outputs/van_der_pol_diagnostics_by_order_v2.csv \
  --comparison-csv outputs/flowstar_vdp_plot_input_v2.csv \
  --out-dir outputs \
  --h 0.01 \
  --steps 10

run_py -m pytest -q
