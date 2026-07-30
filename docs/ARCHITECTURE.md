# Architecture

The source of truth is arranged by responsibility:

```text
src/torch_tm_flowpipe/
  interval.py, polynomial.py, taylor_model.py, tm_vector.py
  flowpipe.py, symbolic_remainder.py
  protocol/
    schema.py, config.py, eligibility.py, runtime.py, pareto.py, provenance.py
benchmarks/
  canonical.yaml, smoke.yaml, formal.yaml
experiments/consolidated_study/
  cli.py
analysis/
  independent_audit.py
tests/
  package, invariant, regression, integration, and protocol tests
```

The existing flat mathematical modules remain the single canonical core; a
cosmetic directory split would add churn without eliminating a duplicate
engine. Protocol-only imports are lazy and do not import PyTorch, which lets
the JAX adapter consume schemas in its isolated environment.

Historical experiment directories provide adapter and diagnostic support.
They are not supported orchestration entrypoints. Benchmark parameters,
configuration profiles, eligibility, and Pareto logic each have one active
source.

The evidence chain is:

```text
branch → commit → profile/config → raw observations → summary/eligibility
       → independently recomputed Pareto → figure manifest → report
```
