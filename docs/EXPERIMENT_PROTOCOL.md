# Experiment protocol

`benchmarks/canonical.yaml` defines systems and numerical settings.
`benchmarks/smoke.yaml` selects one steady repetition and is never
authoritative. `benchmarks/formal.yaml` enumerates all expected identities and
requires ten steady repetitions after a separately recorded cold run.

The total configuration boundary includes solver execution, endpoint
materialization, range evaluation, projection/reset, and next-carry
construction. JIT/compile, engine-internal, post-hoc validation, and
plot/report time are separate fields. Configuration-level memory is
`unavailable` because the current runner does not isolate every configuration
in a memory-measurement process.

Eligibility is evaluated before Pareto dominance. A primary row requires:

- all expected steady repetitions;
- finite positive total runtime and the current boundary version;
- completed requested horizon;
- explicit `True` for every applicable required validation;
- raw endpoint semantics and primary comparability;
- canonical order, basis, remainder, and step-policy identity;
- canonical `completed` failure category.

`not_applicable` is a separate enum, never a truthy Boolean substitute.
Single sweeps and smoke rows stay exploratory/excluded. The auditor recomputes
the frontier from eligible rows and compares exact config IDs.

The shared failure taxonomy is `completed`, `validation_rejected`,
`nonfinite`, `timeout`, `process_error`, `compile_error`,
`missing_dependency`, `trajectory_sanity_failed`,
`analytic_containment_failed`, `schema_invalid`, and `incomplete_unknown`.
