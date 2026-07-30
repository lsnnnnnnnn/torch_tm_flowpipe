# torch-tm-flowpipe

`torch-tm-flowpipe` is a PyTorch research prototype for Taylor-model
flowpipes of polynomial plant dynamics. The canonical development line is
`codex/repository-consolidation-v1`.

The supported package API is under `src/torch_tm_flowpipe`. The only supported
three-tool study entrypoint is:

```bash
python experiments/consolidated_study/cli.py smoke
python experiments/consolidated_study/cli.py formal
```

The runner uses `benchmarks/canonical.yaml` plus the versioned
`benchmarks/smoke.yaml` or `benchmarks/formal.yaml` profile. It refuses a
non-empty output directory. A formal run also refuses a dirty worktree.

## Install and test

Use a local environment compatible with the lock/provenance recorded for the
run:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m pytest -q -m unit
python -m pytest -q -m integration
```

Flowstar and DiffReach are read-only sibling dependencies. The runner resolves
`FLOWSTAR_ROOT`, `FLOWSTAR_AUDIT_ROOT`, `DIFFREACH_ROOT`, and
`DIFFREACH_PYTHON`; otherwise it checks documented sibling defaults. Missing
dependencies are errors, not passes.

## Package example

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

`range_only` and `dependency_preserving` are distinct contracts.
`range_only` collapses the carried set to a box; it must not be interpreted as
dependency preservation.

## Evidence

Repository and branch archaeology is under
`audits/repository_consolidation/20260730T083258Z`. Current architecture,
protocol, limitations, results, and reproduction instructions are in `docs/`.
Historical experiment directories remain supporting evidence only and are not
alternative recommended entrypoints.

The sole authoritative consolidated result is formal run
`artifacts/runs/20260730T153654Z`, generated from frozen source
`0dfdf587ee0fb9cff374dbc41ecdf17dfa2bf781` and accepted by the independent
auditor. `docs/RESULTS.md` states the bounded citable claims; older formal and
smoke runs are explicitly non-citable.

The implementation uses float64 interval operations with conservative
`nextafter` expansion where implemented. It is not a general proof of directed
rounding across every backend. Sampling-based trajectory checks are regression
evidence, not formal proofs.
