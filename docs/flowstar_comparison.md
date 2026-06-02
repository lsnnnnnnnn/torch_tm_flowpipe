# Flow* Plant-Only Comparison Suite

This comparison suite is intentionally **plant-only**. It compares the PyTorch Taylor-model flowpipe kernel in this repository against Flow* on isolated polynomial plant ODEs. It does not run the full CROWN-Reach neural-network-controlled-system pipeline, and it does not add CROWN, auto_LiRPA, Jacobian bounds, non-polynomial functions, or Flow* bindings to the core `torch_tm_flowpipe` library.

CROWN-Reach uses Flow* for the continuous plant dynamics with Taylor Models and combines those plant Taylor Models with CROWN-style neural-network bounds. The scripts under `comparisons/flowstar/` isolate only the plant-flowpipe part so we can compare the local PyTorch Taylor-model kernel against the Flow* backend used in that style of NNCS analysis.

## What Is Compared

The baseline uses fixed step size and fixed Taylor order for both tools. Flow* adaptive step/order features are deliberately disabled for this baseline. A future adaptive baseline should be added separately and labeled `Flow*_adaptive` rather than mixed into the fixed-order results.

The comparison is based on box enclosures extracted from each tool:

- `endpoint_width_sum/max`: endpoint box at the final time, only when a true endpoint box is available.
- `last_segment_width_sum/max`: range box over the final flowpipe segment.
- `tube_width_sum/max`: hull over all parsed flowpipe segment boxes.
- runtime and segment counts.
- torch-only diagnostics such as validation attempts, polynomial term counts, actual degree, and remainder radius.

For torch, endpoint uses `final_tm.range_box()`, last segment uses the last segment Taylor model over `tau in [0,h]`, and tube is the hull over all segments.

For the current `chenxin415/flowstar` toolbox C++ target, generated programs emit GNUPLOT interval files via `plot_2D_interval_GNUPLOT`. The parser hulls two-column numeric plot blocks. These GNUPLOT rectangles are flowpipe segment boxes: the last parsed segment is recorded in `last_segment_width_*`, and the hull over all parsed segments is recorded in `tube_width_*`. They are **not** endpoint boxes. When Flow* endpoint extraction is unavailable, Flow* rows leave `endpoint_width_*` blank, set `endpoint_box_available=False`, and reports must not compute or claim endpoint ratios.

We do **not** compare raw Taylor-model polynomial coefficients. If Flow* emits a format that does not include parseable ranges, the Flow* row is marked `unparsed`; the generated `.cpp` or `.model`, `.stdout.txt`, `.stderr.txt`, and plot files are kept in `outputs/flowstar_models/` for manual inspection.

## Benchmarks

Bundled configs are in `comparisons/flowstar/configs/`:

- `scalar_quadratic.yaml`: `dx/dt = 1 + x^2`, `x0 in [0, 0.1]`.
- `harmonic_oscillator.yaml`: `dx/dt = y`, `dy/dt = -x`, `x0 in [0.9, 1.1]`, `y0 in [-0.1, 0.1]`.
- `van_der_pol.yaml`: `dx/dt = y`, `dy/dt = y - x - x^2*y`, `x0 in [1.1, 1.4]`, `y0 in [2.35, 2.45]`.
- `affine_controlled.yaml`: folds `dx/dt = x + u`, `u = -0.5*x + e`, `e in [-0.01, 0.01]` into `dx/dt = 0.5*x + e`, `de/dt = 0`.

For torch, each benchmark is run with both `mode="range_only"` and `mode="dependency_preserving"`.

## Running

From the repository root:

```bash
pytest -q
python experiments/scalar_quadratic_grid.py
python experiments/harmonic_oscillator.py
python experiments/van_der_pol_sampling.py
python comparisons/flowstar/compare_against_torch_tm.py --all --csv outputs/flowstar_comparison.csv
```

### Current `chenxin415/flowstar` Toolbox Backend

The current `chenxin415/flowstar` repository is a C++ toolbox/static-library interface: it builds `flowstar-toolbox/libflowstar.a`, and individual reachability tasks are C++ programs that include `Continuous.h`, declare variables, construct an `ODE`, configure `Computational_Setting`, and call `ode.reach(...)`. Therefore the default comparison target is `toolbox_cpp`, not an older `.model` parser executable. Flow* is not a Python package in this workflow.

Install/build Flow* separately, then point this suite at the repository root:

```bash
git clone https://github.com/chenxin415/flowstar.git /path/to/flowstar
cd /path/to/flowstar/flowstar-toolbox
make

cd /path/to/torch_tm_flowpipe
python comparisons/flowstar/compare_against_torch_tm.py   --all   --flowstar-root /path/to/flowstar   --csv outputs/flowstar_comparison.csv
```

Alternatively set `FLOWSTAR_ROOT=/path/to/flowstar`. If `libflowstar.a` is missing, the runner attempts `make -C $FLOWSTAR_ROOT/flowstar-toolbox` unless `--no-build-flowstar-lib` is passed.

### Legacy `.model` Executable Backend

For older Flow* installations that expose a model-file parser executable, use:

```bash
python comparisons/flowstar/compare_against_torch_tm.py   --all   --flowstar-target legacy_model   --flowstar-bin /path/to/flowstar   --csv outputs/flowstar_comparison.csv
```

Flow* is optional. If neither `FLOWSTAR_ROOT`/`--flowstar-root` nor a requested legacy executable is available, Flow* rows are written with `status=skipped`; torch rows and plots are still generated.

## CSV Schema

Key fields are:

```text
system,tool,mode,h,steps,order,setting_label,status,
endpoint_width_sum,endpoint_width_max,
last_segment_width_sum,last_segment_width_max,
tube_width_sum,tube_width_max,
box_source,endpoint_box_available,last_segment_box_available,tube_box_available,
runtime_s,num_segments,validation_attempts,term_count,actual_degree,
remainder_radius,containment_failures,flowstar_internal_reach_s,
flowstar_wall_compile_s,flowstar_wall_run_s,flowstar_wall_total_s,
flowstar_model_path,flowstar_stdout_path,flowstar_stderr_path,failure_reason
```

Legacy compatibility fields may still appear in CSVs, but narrative reports and torch-vs-Flow* ratios should use the explicit endpoint, last-segment, and tube fields. Endpoint ratios are allowed only when both compared rows have `endpoint_box_available=True` and nonempty endpoint widths. Current Flow* GNUPLOT-derived rows have `endpoint_box_available=False`, so current torch-vs-Flow* ratios are limited to `last_segment` and `tube`.

Unavailable fields are left empty for Flow*. This is expected for fields that are internal to `torch_tm_flowpipe`, such as validation attempts, term counts, actual degree, and remainder radius.

## Runtime Semantics

For Flow*, `runtime_s` defaults to `flowstar_internal_reach_s` when the generated C++ program printed `FLOWSTAR_RUNTIME_S`. This is the Flow* reachability clock time. If the internal time is unavailable, `runtime_s` falls back to total wall time.

Compile/run wall-clock times are reported separately:

- `flowstar_wall_compile_s`: Python wall time spent compiling the generated C++ case.
- `flowstar_wall_run_s`: Python wall time spent running the compiled case, including executable and plotting overhead.
- `flowstar_wall_total_s`: compile plus run wall time.
- `runtime_s` for torch: Python algorithm wall time.

## Plots

The order/Van der Pol report bundle writes semantic torch-vs-Flow* plots only for matching boxes:

- `outputs/torch_over_flowstar_last_segment_width_ratio_by_order.png`
- `outputs/torch_over_flowstar_tube_width_ratio_by_order.png`

The comparison script may also create generic final-width, runtime, last-segment-ratio, tube-ratio, and dependency-vs-range plots. Ambiguous endpoint-vs-GNUPLOT ratio plots are not part of the corrected report bundle.

## Soundness Note

The sampling containment column is a regression sanity check only. It checks a small sample grid against the endpoint box using exact solutions where simple closed forms exist and RK4 samples for Van der Pol. It is not a formal proof of containment and should not be described as one.

## Flow* Statuses

Flow* statuses are interpreted as follows:

- `skipped`: Flow* root/executable was not available, but the case was exported.
- `compile_failed` or `compile_timeout`: the generated C++ case or local Flow* build failed to compile.
- `run_failed` or `timeout`: compilation succeeded but execution failed or timed out.
- `failed`: Flow* declared a reachability failure such as large overestimation.
- `unparsed`: Flow* executed, but the harness could not parse endpoint, last-segment, or tube boxes from stdout/stderr/plot files.
- `completed`: Flow* executed and at least one semantic box was parsed.

## Summary Report

Use:

```bash
python comparisons/flowstar/summarize_comparison.py   outputs/flowstar_comparison.csv   --out outputs/flowstar_comparison_summary.md
```

The report first gives the dependency-preserving/range-only evidence that can be computed without Flow*. It then gives torch/Flow* ratios only for parsed Flow* rows with compatible box semantics. If all Flow* rows are skipped or unparsed, the report explicitly says not to make numeric torch-vs-Flow* claims yet.
