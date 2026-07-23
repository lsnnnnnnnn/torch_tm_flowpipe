# First-order three-way benchmark

This directory contains a plant-only, fixed-step comparison of
`torch_tm_flowpipe`, Flow*, and DiffReach on scalar Riccati, a harmonic
oscillator, and Van der Pol dynamics. `benchmark_spec.yaml` is the single
machine-readable source for equations, state order, initial boxes, grids, and
numerical settings. All three adapters construct their right-hand side from
that file.

Read [ORDER_SEMANTICS.md](ORDER_SEMANTICS.md) before comparing widths. The
installed Flow* toolbox rejects fixed order 1, and DiffReach's affine flag
does not retain the same intermediate basis as Torch order 1.

## Environments

- `py11`: Python 3.11, Torch 2.5.1+cu121, SciPy, pandas, Matplotlib, and
  PyYAML. CUDA is unavailable on the benchmark host, so Torch uses float64 CPU.
- `diffreach312`: Python 3.12 with CPU `jax==0.10.2`, SciPy, PyYAML, and
  pytest. JAX x64 is explicitly enabled by the adapter.
- Flow*: system GCC/G++ 15.2 and the existing
  `/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a`.

The declared editable DiffReach install was attempted but its dependency
resolver found incompatible Equinox requirements between `jax2onnx` and
`immrax[cuda]`. The adapter imports only the repository's analytic
plant-reachability source from the unchanged checkout.

The Flow* adapter follows the existing `comparisons/flowstar` toolbox
include/link/compile pattern. It uses a benchmark-specific renderer because
the existing renderer does not check the boolean return of
`setFixedStepsize`, and its plot-file parser supplies final/tube hulls rather
than both endpoint and whole-segment intervals at every step. Direct C++
extraction is required for this benchmark's long-form schema.

## Reproduce

From the benchmark worktree:

```bash
cd /srv/local/shengenli/torch_tm_flowpipe_first_order_bench
experiments/first_order_three_way/run_smoke.sh
experiments/first_order_three_way/run_all.sh
```

To run the smoke gate interactively and then launch the full sweep under tmux:

```bash
experiments/first_order_three_way/launch_background.sh
tmux attach -t tm_first_order_three_way
```

Override the full output location with `FIRST_ORDER_OUTPUT_DIR=/absolute/path`.
`run_all.sh` otherwise creates a UTC timestamp under `results/`. It writes the
environment audit, resolved spec, references, adapter outputs, long-form CSV,
summary CSV, correctness JSON, per-run metadata/logs, PNG/PDF plots, and the
Markdown report.

Individual adapters can be run as:

```bash
conda run -n py11 python experiments/first_order_three_way/run_torch.py --output-dir "$OUT"
conda run -n py11 python experiments/first_order_three_way/run_flowstar.py --output-dir "$OUT"
conda run -n diffreach312 python experiments/first_order_three_way/run_diffreach.py --output-dir "$OUT"
```

Each configuration records the first validation/contraction/non-finite
failure. The fixed-order sweeps do not raise order, shrink the requested step,
partition the box, or invoke rescue logic. Flow* compilation and each process
execution have the specification's 600-second timeout; a failure is serialized
and the sweep continues.

## Result interpretation

`raw_results.csv` has one row per tool, protocol, system, configuration, time,
interval kind, and state. Endpoint rows fix local time at the right endpoint;
tube rows bound the whole stored segment. `run_summary.csv` contains final and
maximum widths, volume, exact inflation where available, successful horizon,
and separated build/warmup/steady timing.

Analytic endpoint hulls validate Riccati and harmonic results. Deterministic
high-accuracy trajectory samples check all tubes, including Van der Pol, but
those samples are not a formal proof of soundness.
