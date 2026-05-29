# Flow* plant-only comparison suite

This comparison suite is intentionally **plant-only**.  It compares the PyTorch Taylor-model flowpipe kernel in this repository against Flow* on isolated polynomial plant ODEs.  It does not run the full CROWN-Reach neural-network-controlled-system pipeline, and it does not add CROWN, auto_LiRPA, Jacobian bounds, non-polynomial functions, or Flow* bindings to the core `torch_tm_flowpipe` library.

CROWN-Reach uses Flow* for the continuous plant dynamics with Taylor Models and combines those plant Taylor Models with CROWN-style neural-network bounds.  The scripts under `comparisons/flowstar/` isolate only the plant-flowpipe part so we can compare the local PyTorch Taylor-model kernel against the Flow* backend used in that style of NNCS analysis.

## What is compared

The first comparison uses fixed step size and fixed Taylor order for both tools.  Flow* adaptive step/order features are deliberately disabled for fairness in this baseline.  A future adaptive baseline should be added separately and labeled `Flow*_adaptive` rather than mixed into the fixed-order results.

The comparison is based on **box enclosures** extracted from each tool:

- final reachable box at the end of the horizon,
- flowpipe tube box over all segments when available,
- runtime and segment counts,
- torch-only diagnostics such as validation attempts, polynomial term counts, and remainder radius.

We do **not** compare raw Taylor-model polynomial coefficients.  The Flow* parser starts with text/range parsing from stdout/stderr and plotting/range files.  For the current toolbox C++ target, generated C++ programs also emit GNUPLOT interval files via `plot_2D_interval_GNUPLOT`, and the parser can hull two-column numeric plot blocks.  If Flow* emits a format that does not include parseable ranges, the Flow* row is marked `unparsed`; the generated `.cpp` or `.model`, `.stdout.txt`, `.stderr.txt`, and plot files are kept in `outputs/flowstar_models/` for manual inspection.

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

### Current `chenxin415/flowstar` toolbox backend

The current `chenxin415/flowstar` repository is a toolbox/static-library interface: it builds `flowstar-toolbox/libflowstar.a`, and individual reachability tasks are C++ programs that include `Continuous.h`, declare variables, construct an `ODE`, configure `Computational_Setting`, and call `ode.reach(...)`.  Therefore the default comparison target is now `toolbox_cpp`, not an older `.model` parser executable.

Install/build Flow* separately, then point this suite at the repository root:

```bash
git clone https://github.com/chenxin415/flowstar.git /path/to/flowstar
cd /path/to/flowstar/flowstar-toolbox
make

cd /path/to/torch_tm_flowpipe
python comparisons/flowstar/compare_against_torch_tm.py \
  --all \
  --flowstar-root /path/to/flowstar \
  --csv outputs/flowstar_comparison.csv
```

Alternatively set `FLOWSTAR_ROOT=/path/to/flowstar`.  If `libflowstar.a` is missing, the runner attempts `make -C $FLOWSTAR_ROOT/flowstar-toolbox` unless `--no-build-flowstar-lib` is passed.

### Legacy `.model` executable backend

For older Flow* installations that expose a model-file parser executable, use:

```bash
python comparisons/flowstar/compare_against_torch_tm.py \
  --all \
  --flowstar-target legacy_model \
  --flowstar-bin /path/to/flowstar \
  --csv outputs/flowstar_comparison.csv
```

Flow* is optional.  If neither `FLOWSTAR_ROOT`/`--flowstar-root` nor a requested legacy executable is available, Flow* rows are written with `status=skipped`; torch rows and plots are still generated.

The CSV fields are:

```text
system,tool,mode,h,steps,order,status,final_width_sum,final_width_max,
flowpipe_width_sum,flowpipe_width_max,runtime_s,num_segments,
validation_attempts,term_count,remainder_radius,containment_failures
```

Unavailable fields are left empty for Flow*.  This is expected for fields that are internal to `torch_tm_flowpipe`, such as validation attempts, term counts, and remainder radius.

## Plots

The comparison script writes:

- `outputs/final_width_vs_steps.png`
- `outputs/runtime_vs_steps.png`
- `outputs/width_ratio_torch_over_flowstar.png`
- `outputs/dependency_preserving_vs_range_only.png`

If Flow* was unavailable or its output was not parseable, `width_ratio_torch_over_flowstar.png` is still created with a placeholder message.

## Soundness note

The sampling containment column is a regression sanity check only.  It checks a small sample grid against the final box using exact solutions where simple closed forms exist and RK4 samples for Van der Pol.  It is not a formal proof of containment and should not be described as one.

## Why a C++ Flow* backend is still comparable

Flow* is C++, but the comparison is still meaningful because the benchmark task
is language-independent.  For each YAML case we fix the same ODE, initial box,
step size, number of steps, and Taylor order.  The harness then compares the
observable enclosures:

```text
final reachable box width
flowpipe tube box width
runtime
status
```

The harness does not require Flow* and PyTorch to share an internal Taylor-model
class.  Flow* is run as an external verifier/backend, just like many verification
benchmark pipelines run tools with different languages and internal data
structures and then compare standardized outputs.

## Generated artifacts for skipped Flow* rows

When Flow* is unavailable, the comparison script still exports the C++ case into
`outputs/flowstar_models/`.  This makes a skipped row actionable: copy the
corresponding `.cpp` file to a server with `chenxin415/flowstar`, or pass
`--flowstar-root`, and rerun the same command to obtain real Flow* rows.

Flow* statuses are interpreted as follows:

- `skipped`: Flow* root/executable was not available, but the case was exported.
- `compile_failed` or `compile_timeout`: the generated C++ case or local Flow*
  build failed to compile.
- `run_failed` or `timeout`: compilation succeeded but execution failed or timed
  out.
- `unparsed`: Flow* executed, but the harness could not parse a final/tube box
  from stdout/stderr/plot files.
- `completed`: Flow* executed and at least the metric final box was parsed.

## Summary report

Use:

```bash
python comparisons/flowstar/summarize_comparison.py \
  outputs/flowstar_comparison.csv \
  --out outputs/flowstar_comparison_summary.md
```

The report first gives the dependency-preserving/range-only evidence that can be
computed without Flow*.  It then gives torch/Flow* ratios only for parsed Flow*
rows.  If all Flow* rows are skipped, the report explicitly says not to make
numeric torch-vs-Flow* claims yet.
