# Architecture

The repository has one active implementation per numerical abstraction and
keeps comparison protocol code outside the numerical core.

```text
src/torch_tm_flowpipe/
  interval.py                 canonical interval arithmetic
  polynomial.py               canonical sparse total-degree polynomial
  taylor_model.py, tm_vector.py
  flowpipe.py                 one- and multi-step propagation
  symbolic_remainder.py       diagnostic symbolic-remainder support
  protocol/
    schema.py, config.py, eligibility.py, runtime.py, pareto.py
    provenance.py, backend_identity.py
benchmarks/
  canonical.yaml              unique benchmark-system source
  smoke.yaml, formal.yaml     versioned selection profiles
  cross_tool_gates.yaml       fail-closed comparison gate state
experiments/consolidated_study/cli.py
                               canonical comparison orchestration
experiments/flowstar_step_trace_compare.py
                               supported order-2 diagnostic
analysis/independent_audit.py  artifact acceptance checks
tests/                         unit, protocol, regression, integration
```

## Boundaries

- Experiments import `src/torch_tm_flowpipe`; output directories are never
  import sources.
- Benchmark parameters have one active source. Historical result-local YAML
  files are frozen provenance, not alternate definitions.
- Reports consume an explicit run directory and never discover a “latest”
  output implicitly.
- Supported runners reject non-empty outputs.
- Flowstar identity is resolved before an output directory is created.
  Primary execution rejects audit-named roots, enabled audit behavior
  variables, and unknown tracked modifications.
- `official-program`, `generated-stock`, and `patched-audit` are distinct
  backend objects. Patched audit is always diagnostic-only and
  `primary_eligible=false`.
- Endpoint, accepted-segment, and full-tube bounds are separate from the
  raw-versus-tightened refinement dimension.
- Requested order, effective retained degree, basis, remainder policy, step
  policy, requested horizon, and successful horizon are explicit fields.

Historical experiment directories remain recoverable scientific lineage.
They are not supported orchestration entrypoints and may not publish a current
winner/Pareto claim.

The evidence chain is:

```text
branch → commit → code → config → backend identity → raw artifact
       → summary → figure → bounded claim
```
