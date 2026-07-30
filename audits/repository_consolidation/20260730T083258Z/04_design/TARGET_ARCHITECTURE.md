# Target architecture

The refactor preserves the stable package imports where renaming would create
noise, but makes each logical boundary explicit and gives the experiment
protocol a single implementation.

```text
src/torch_tm_flowpipe/
  interval.py               core interval arithmetic
  polynomial.py             core sparse polynomial arithmetic
  taylor_model.py           canonical TaylorModel
  tm_vector.py              canonical vector wrapper
  picard.py                 propagation/remainder/validation mechanics
  flowpipe.py               one-step and multi-step public orchestration
  protocol/
    schema.py               versioned row contract and failure taxonomy
    eligibility.py          fail-closed validation and primary eligibility
    pareto.py               pure post-eligibility Pareto recomputation
    provenance.py           config identity and immutable run manifests

benchmarks/
  canonical.yaml            systems, initial sets, steps, horizons, bases

experiments/
  consolidated_study/
    cli.py                   one supported smoke/formal/report/audit CLI
    adapters/
      torch_tm.py
      diffreach.py
      flowstar.py
    runner.py               configuration orchestration only
    report.py               consumes validated summaries/manifests only

tests/
  unit/                     arithmetic and schema pure functions
  regression/               every known protocol defect
  integration/              Torch and optional external adapters
  protocol/                 collector -> eligibility -> Pareto -> audit

analysis/
  independent_audit.py      independent CSV/manifest/Pareto acceptance
```

## Compatibility decision

The existing flat core modules remain the canonical arithmetic engine. They
are not copied into a visually different `core/` directory because that would
duplicate implementations and invalidate many imports without changing
semantics. The new `protocol/` package is deliberately separate from
arithmetic.

Historical experiment generations remain reachable from archive tags. The
active branch keeps only one supported experiment entrypoint and a compact
history index. Tests must import package/protocol APIs, not committed result
bundles.

## Data flow

```text
canonical config
  -> adapter raw observations
  -> schema validation
  -> explicit completion/validation eligibility
  -> raw/tightened and formal/exploratory partition
  -> Pareto recomputation on eligible rows
  -> figures/reports with source manifest
  -> independent audit and SHA256 verification
```

No plotter or report generator may reinterpret missing fields, choose the last
row implicitly, or mutate eligibility/Pareto flags.

## Runtime boundary

`runtime_boundary_version = total_configuration_v2` covers, for every step:

1. solver propagation and validation;
2. raw endpoint extraction and range evaluation;
3. projection/discard accounting;
4. affine/box reset and carry construction; and
5. device synchronization required to complete those operations.

Compile/JIT, cold total, steady repeated total, engine-internal, validation,
and plot/report times are separate fields.

## External adapters

Flow* and DiffReach remain sibling, provenance-pinned, read-only repositories.
Optional integration tests use explicit `flowstar` and `diffreach` markers.
Missing dependencies produce `missing_dependency` capability/failure records
and a blocked gate, never a passing result.
