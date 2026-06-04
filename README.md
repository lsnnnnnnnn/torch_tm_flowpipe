# torch-tm-flowpipe

`torch-tm-flowpipe` is a small PyTorch-native research prototype for fixed-step
Taylor-model flowpipes of polynomial plant dynamics, including a
dependency-preserving multi-step propagation mode. It implements only the
plant-side Taylor model kernel used conceptually by NNCS reachability methods:

```python
from torch_tm_flowpipe import Interval, flowpipe_step
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode

segment = flowpipe_step(
    scalar_quadratic_ode,
    [Interval(0.0, 0.5)],
    h=0.1,
    order=4,
)

print(segment.status)
print(segment.tm.range_box())
print(segment.final_tm.range_box())
```

The package supports:

- scalar interval arithmetic with conservative outward rounding using
  `torch.nextafter` where available;
- sparse total-degree multivariate polynomials;
- scalar Taylor models `p(vars) + I`;
- fixed-step Picard iteration for polynomial ODEs;
- one-step and multi-step flowpipe construction;
- dependency-preserving multi-step propagation that carries `final_tm` forward
  instead of compressing every step to a box;
- constant interval controls and affine controls of the form
  `u = A x0 + b + error`, with `error in [-r, r]`.

It intentionally does not integrate CROWN-Reach, CROWN, auto_LiRPA, Flow*,
hybrid automata machinery, guards, jumps, adaptive order or step-size control,
symbolic remainders, branch-and-bound, Jacobian/sensitivity bounds, or
transcendental functions.

## Development

From an activated Python environment, install the package editable and run the
baseline checks:

```bash
python -m pip install -e ".[test]"
pytest -q
python examples/scalar_quadratic.py
python examples/van_der_pol_short.py
python examples/affine_controlled.py
```

Development setup and checks are split so repeated checks do not reinstall:

```bash
bash scripts/setup_dev.sh
bash scripts/check_all.sh
```

For the requested `py11` environment, run the same commands through `conda`:

```bash
conda run -n py11 python -m pip install -e ".[test]"
conda run -n py11 pytest -q
conda run -n py11 python examples/scalar_quadratic.py
conda run -n py11 python examples/van_der_pol_short.py
conda run -n py11 python examples/affine_controlled.py
```

Experiment scripts keep their stdout output unchanged and can also write CSV
results:

```bash
python experiments/scalar_quadratic_grid.py --csv outputs/scalar_quadratic_grid.csv
python experiments/harmonic_oscillator.py --csv outputs/harmonic_oscillator.csv
python experiments/van_der_pol_sampling.py --csv outputs/van_der_pol_sampling.csv
```

## Diagnostics artifacts

The committed Flow* Van der Pol diagnostics artifacts live under
`outputs/flowstar_benchmark_diagnostics/`,
`outputs/flowstar_benchmark_diagnostics_stage2/`, and
`outputs/flowstar_benchmark_diagnostics_stage3/`. Stage 3 contains an
experimental diagnostic-only symbolic remainder prototype/result. It is not
part of the supported default API and did not improve the benchmark objective.
See `docs/flowstar_vanderpol_pytorch_diagnostics_conclusion.md` for the final
decision record.


## Dependency-preserving multi-step example

```python
from torch_tm_flowpipe import Interval, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode

result = flowpipe_multi_step(
    scalar_quadratic_ode,
    [Interval(0.0, 0.1)],
    h=0.01,
    steps=5,
    order=4,
    mode="dependency_preserving",
)

print(result.status)
print(result.final_tm.range_box())
```

The experiment scripts write CSV files with these columns: `system`, `h`,
`order`, `status`, `final_width`, `flowpipe_width`, `runtime_s`,
`validation_attempts`, `containment_failures`, `device`, and `dtype`.

Optional plots can be generated from the CSV files:

```bash
python experiments/plot_results.py outputs/scalar_quadratic_grid.csv \
  outputs/harmonic_oscillator.csv outputs/van_der_pol_sampling.csv \
  --out-dir outputs
```

## Flow* plant-only comparison suite

The plant-only comparison utilities live under `comparisons/flowstar/`.  They
export fixed-step/fixed-order Flow* models, optionally run a Flow* executable,
parse box-style range output when available, and compare against both
`torch_tm_flowpipe` multi-step modes.

```bash
python comparisons/flowstar/compare_against_torch_tm.py \
  --all \
  --csv outputs/flowstar_comparison.csv
```

If Flow* is not available on `PATH` or via `FLOWSTAR_BIN`, the script still runs
the torch baselines and writes Flow* rows with `status=skipped`.  See
`docs/flowstar_comparison.md` for details, limitations, and the CSV schema.


## Flow* comparison backend note

The plant-only comparison suite under `comparisons/flowstar/` defaults to the current `chenxin415/flowstar` toolbox interface. It generates C++ benchmark programs that include `Continuous.h` and link against `flowstar-toolbox/libflowstar.a`. Older `.model` file export remains available with `--flowstar-target legacy_model`, but it is not the default.

For the local server setup used by the generated reports:

```bash
cd /srv/local/shengenli/torch_tm_flowpipe
export FLOWSTAR_ROOT=/srv/local/shengenli/flowstar
conda run -n py11 python -m pip install -e ".[test]"
conda run -n py11 pytest -q
```


## Evidence and comparison reports

After running the Flow* comparison script, summarize whether the new
`dependency_preserving` mode is useful with:

```bash
python comparisons/flowstar/summarize_comparison.py   outputs/flowstar_comparison.csv   --out outputs/flowstar_comparison_summary.md
```

The summary reports dependency-preserving/range-only width ratios and, when
Flow* is installed and parseable, torch/Flow* width ratios.  See
`docs/technical_validation.md` for what these results do and do not prove.
