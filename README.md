# torch-tm-flowpipe

`torch-tm-flowpipe` is a PyTorch-native, plant-only Taylor-model flowpipe
prototype for polynomial ODEs. The active evidence branch is
`codex/vdp-terminal-range-closure-20260805`, descended from the generic dense
backend tip `82c54a2`; the canonical package is
`src/torch_tm_flowpipe`.

The project implements interval arithmetic, sparse total-degree polynomials,
Taylor models `p + R`, one-step Picard/Taylor propagation, and multi-step
`range_only` and `dependency_preserving` modes. It is not CROWN-Reach, a
Flowstar rewrite, or a complete NNCS tool.

## Supported quick start

```bash
conda run -n py11 python -m pip install -e ".[test]"
conda run -n py11 pytest -q
conda run -n py11 python examples/scalar_quadratic.py
```

The public package example is:

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
print(result.status, result.final_tm.range_box())
```

`range_only` evaluates the carried set to a box at every step and loses
cross-step dependency. `dependency_preserving` carries polynomial structure,
but is not guaranteed to produce a tighter range.

## Runners and results

The current plant-only dense result includes an exact, safe JSON terminal-state
replay and a validated batched subdivision polynomial-range path. It closes the
original Van der Pol terminal step without changing the numerical contract and
reaches R4 at a fresh horizon of `6.397083942944808`. It does not complete
T=7.5 or T=10. See the
[terminal-range closure](docs/VDP_TERMINAL_RANGE_CLOSURE.md) and the hashed
evidence under
`evidence/vdp_terminal_range_closure/20260805T055556Z`.

The current native-reproduction registry is
[`benchmarks/native_reproduction_registry.json`](benchmarks/native_reproduction_registry.json),
with raw command/artifact evidence under
`outputs/native_reproduction_no_adapters/20260804T081205Z`.  It directly runs the
author/stock entrypoints for Xiangru, stock Flow*, upstream DiffReach and this
project.  Reproduction, horizon completion, property/certificate, soundness and
comparison eligibility are reported separately; adapters, generated harnesses and
endpoint repair are excluded from native rows.

Registry rows label reference evidence as `portable_committed`,
`server_local_private_reference`, or `not_applicable`; an absolute private path is
never presented as portable evidence.

See the [native matrix](docs/NATIVE_REPRODUCTION_MATRIX.md),
[standard](docs/NATIVE_REPRODUCTION_STANDARD.md), and
[current results status](docs/RESULTS_STATUS.md). The clean-stock Flow* scalar
diagnosis is in the
[correctness closure](docs/FLOWSTAR_SCALAR_AFFINE_CORRECTNESS_CLOSURE.md). No cross-tool speedup, Pareto
frontier or winner is currently citable.

The canonical comparison runner is
`experiments/consolidated_study/cli.py`. Formal comparison is fail-closed
until every gate in `benchmarks/cross_tool_gates.yaml` is independently
verified. Do not use historical experiment scripts as alternate headline
runners.

The sole supported Flowstar order-2 diagnostic entrypoint is:

```bash
export FLOWSTAR_ROOT="$(dirname "$(git rev-parse --show-toplevel)")/flowstar"
conda run -n py11 python experiments/flowstar_step_trace_compare.py \
  --flowstar-root "$FLOWSTAR_ROOT" \
  --out-dir /tmp/flowstar-order2-trace-new \
  --horizon 0.1 --max-segments 1 --order 2 \
  --compare-mode attempt_aligned
```

It writes only to a new output directory. Its known outcome is a Picard
remainder self-map validation rejection at the configured step-size floor,
not a crash and not evidence that Flowstar does not support order 2.

Historical comparison artifacts retain their earlier status. See
[reproducibility](docs/REPRODUCIBILITY.md) and the single
[history index](docs/history/BRANCH_AUDIT.md).
