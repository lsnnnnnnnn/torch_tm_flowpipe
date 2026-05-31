# Flow* plant-only comparison harness

This directory is an external benchmark harness. It is not imported by the core
`torch_tm_flowpipe` Taylor-model algorithms.

Default backend: `toolbox_cpp`, matching the current `chenxin415/flowstar`
repository.

Workflow:

1. Read `configs/*.yaml`.
2. Export a Flow* C++ case that includes `Continuous.h`.
3. Compile the case against `FLOWSTAR_ROOT/flowstar-toolbox/libflowstar.a`.
4. Run the compiled executable.
5. Parse stdout/stderr/Flow* plotting files for final and tube boxes.
6. Run torch `range_only` and `dependency_preserving` on the same plant case.
7. Write one CSV and plots.

Minimal torch-only smoke:

```bash
python comparisons/flowstar/compare_against_torch_tm.py \
  --csv outputs/flowstar_comparison_smoke.csv \
  --skip-flowstar \
  --no-plots
```

Full grid with Flow* installed:

```bash
export FLOWSTAR_ROOT=/path/to/flowstar
python comparisons/flowstar/compare_against_torch_tm.py \
  --all \
  --csv outputs/flowstar_comparison.csv
```

Summarize usefulness evidence:

```bash
python comparisons/flowstar/summarize_comparison.py \
  outputs/flowstar_comparison.csv \
  --out outputs/flowstar_comparison_summary.md
```
