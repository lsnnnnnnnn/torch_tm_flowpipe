# Experiment protocol

`benchmarks/canonical.yaml` is the single active source for systems and
numerical settings. `benchmarks/smoke.yaml` selects one steady repetition and
is never authoritative. `benchmarks/formal.yaml` enumerates every expected
identity and requires ten steady repetitions after a separately recorded cold
run.

The total-configuration runtime boundary includes solver execution, endpoint
materialization, range evaluation, projection/reset, and next-carry
construction. JIT/compile, engine-internal, post-hoc validation, and
plot/report time are separate fields. Configuration-level memory remains
`unavailable` unless each configuration is isolated in a suitable
measurement process.

Eligibility is evaluated before Pareto dominance. A primary row requires:

- every expected steady repetition;
- finite positive total runtime and the current boundary version;
- completed requested horizon and canonical `completed` failure category;
- explicit `True` for every applicable required validation;
- raw endpoint, endpoint location, raw refinement, and explicit exporter
  semantics;
- canonical requested order, effective degree, basis, remainder, and step
  policy;
- a recorded backend class/SHA/dirty state and a primary-eligible execution
  route;
- explicit primary comparability.

`not_applicable` is a separate enum, never a truthy Boolean substitute.
Single sweeps and smoke rows remain exploratory/excluded. For an eligible
formal bundle, the independent auditor recomputes the cross-tool frontier
within each system and requested evaluation horizon. Tool identity is not a
grouping boundary.

Formal comparison is currently blocked. Every gate in
`benchmarks/cross_tool_gates.yaml` must be independently verified before the
formal runner creates output: stock backend identity; official-parser versus
generated-stock field parity; endpoint/segment/tube exporter semantics;
raw/tightened separation; order/basis contract; runtime-boundary parity;
fail-closed completion/validation; and exclusion of patched rows from primary
tables.

Flowstar routes are distinct:

- `official-program`: an official parser/program execution;
- `generated-stock`: generated harness linked to an unmodified stock or
  exactly identified GCC15-compatibility checkout;
- `patched-audit`: diagnostic-only, with patch manifest and
  `primary_eligible=false`.

The primary startup check canonicalizes the root, records SHA/status/library
hash, rejects audit-named roots and unknown tracked changes, and rejects
enabled `FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION` or
`FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT`.

The shared failure taxonomy is `completed`, `validation_rejected`,
`nonfinite`, `timeout`, `process_error`, `compile_error`,
`missing_dependency`, `trajectory_sanity_failed`,
`analytic_containment_failed`, `schema_invalid`, and `incomplete_unknown`.
An order-2 remainder self-map miss is `validation_rejected` with reason
`remainder_self_map_failed`, never `crash`, `unsupported_order`, or
`completed`.
