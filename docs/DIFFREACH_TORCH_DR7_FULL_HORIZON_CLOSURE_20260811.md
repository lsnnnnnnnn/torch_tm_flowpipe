# DiffReach / Torch DR7 full-horizon closure

Date: 2026-08-11

## Outcome

`DIFFREACH_TORCH_DR7_FULL_HORIZON_DIVERGED`.

## Eligibility

The full B64, 1,000-step explicit-float64 traces are eligible for a diagnostic
cross-tool semantics comparison. Performance comparison is ineligible because
the preregistered endpoint/tube/J-Phi equality gate diverged. No GPU timing was
run after the CPU semantics gate failed.

## Contract

Pinned DiffReach `dd628eb443b517d6415de93e7035b4baef73963e`, frozen DR7
support, B64 partition hash `e66e54...`, explicit float64 at every builder,
two Picard constructions, ten DR-RP rounds, `h=0.01`, 1,000 steps, and
read-only observers. The two-ULP companion bound was preregistered.

## What was actually run

A minimal patch exposed native upstream step operators without changing them;
observer inertness was bit-exact. The upstream JAX and Torch implementations
each completed 1,000 CPU steps and captured masks, retained models,
normalization, symbolic J/Phi, endpoint, segment tube, and prefix tube. A
sequential upstream repeat reproduced the trace SHA exactly. The stock mixed
builder-dtype lane was run separately as native capability only.

## Exact results

- Both explicit-f64 lanes completed T10; every initial mask was true and all
  initial/later mask decisions matched.
- First divergence: step 1, `poly1_L[4,1,0]`, one ULP:
  DiffReach `-0x1.b962f5c28f5c7p+0`, Torch
  `-0x1.b962f5c28f5c6p+0`.
- First endpoint and tube divergence: step 2.
- J/Phi equality: false; queue counts and clear events remain equal.
- Maximum endpoint absolute delta: `1.7763568394002505e-14`; maximum endpoint
  ULP delta: 471,040.
- Maximum tube absolute delta: `1.7763568394002505e-14`; maximum tube ULP
  delta: 1,325,056. The two-ULP envelope fails.
- The difference is attributed to JAX/XLA versus Torch floating expression
  evaluation order; forcing the CPU platform changes the selected XLA value.

## What is comparable

All 1,000 explicit-f64 steps, masks, discrete queue events, operator fields,
J/Phi fields, endpoints, segment tubes, and prefix tubes under the frozen DR7
contract.

## What remains unavailable

Matched CPU/GPU performance, a formal directed-rounding claim, and a
bit-exact or preregistered-ULP-bounded full-horizon closure are unavailable.

## Negative results

The earlier one-step operator closure does not close the full horizon. Exact
mask equality does not imply model, J/Phi, endpoint, or tube equality. The
stock mixed-dtype endpoint-only driver cannot substitute for explicit-f64
full-state evidence.

## Limitations

The traces are ordinary-float64 empirical results. Large ULP counts occur on
small accumulated differences even though absolute errors remain small; the
preregistered gate is still failed, not retroactively widened.

## Evidence paths

Under `outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z/`:

- `05_diffreach_torch_full_horizon/cross_tool_comparison/comparison.json`;
- `05_diffreach_torch_full_horizon/cross_tool_comparison/endpoint_tube_delta_by_step.csv`;
- per-tool summaries and step-1/step-2 captures;
- `12_pairwise_tables/table_md_diffreach_torch_explicit_f64.csv`;
- `13_figures/diffreach_torch_full_horizon_deltas.svg` and source CSV.

## Reproduction commands

```bash
python experiments/run_diffreach_explicit_f64_full_trace.py --help
python experiments/run_torch_fixed_dr7_full_trace.py --help
python experiments/compare_diffreach_torch_full_horizon.py --help
```

## Next authorized action

If bitwise equivalence is still required, specify one shared floating
evaluation contract and test it as a new study. Do not publish a timing ratio
from the diverged semantics lanes.
