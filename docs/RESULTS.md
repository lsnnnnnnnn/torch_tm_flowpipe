# Results

## Authoritative formal run

Run `20260730T153654Z`, produced from frozen source
`0dfdf587ee0fb9cff374dbc41ecdf17dfa2bf781`, is
`accepted_authoritative`. It contains exactly 24 expected and eligible
configurations, 264 raw observations (one cold plus ten steady observations
per configuration), no missing or duplicate identities, no excluded or failed
configurations, and 12 configurations on the independently recomputed
cross-tool width/runtime frontier.

The table below is the complete frontier from `PRIMARY_PARETO.csv`. Runtime is
the median steady full-configuration time in seconds under boundary
`total_configuration_v2`; width is the raw endpoint width at the requested
evaluation horizon.

| System | Tool | h | Order | Median runtime (s) | Width |
|---|---|---:|---:|---:|---:|
| coupled quadratic | DiffReach | 0.005 | N/A | 0.00219469 | 0.0429775920 |
| coupled quadratic | DiffReach | 0.01 | N/A | 0.00124510 | 0.0429889432 |
| coupled quadratic | Flowstar | 0.005 | 4 | 0.29706135 | 0.0429677009 |
| coupled quadratic | Flowstar | 0.01 | 4 | 0.14948171 | 0.0429677281 |
| coupled quadratic | Torch TM | 0.01 | 4 | 13.98076456 | 0.0429662678 |
| harmonic | DiffReach | 0.01 | N/A | 0.01108746 | 0.3031861786 |
| harmonic | Flowstar | 0.01 | 4 | 0.17094165 | 0.2820913730 |
| Riccati | DiffReach | 0.01 | N/A | 0.00144950 | 0.1138945823 |
| Van der Pol | DiffReach | 0.005 | N/A | 0.00853865 | 0.1406466366 |
| Van der Pol | DiffReach | 0.01 | N/A | 0.00452902 | 0.1513324179 |
| Van der Pol | Flowstar | 0.005 | 4 | 1.29182688 | 0.1128217672 |
| Van der Pol | Flowstar | 0.01 | 4 | 0.64969883 | 0.1132399974 |

These are trade-off statements, not a universal tool ranking. In this
specific CPU environment DiffReach supplies the lower-runtime frontier
points; Flowstar trades more runtime for lower width on harmonic and Van der
Pol; and the Torch TM order-4 `h=0.01` coupled-quadratic configuration has the
smallest width among that system's delivered points but a substantially
higher runtime. Native order labels do not imply matched bases.

The run was executed on macOS arm64 with 8 CPU cores, Python 3.11.11,
PyTorch 2.5.1, no NVIDIA CUDA device, and DiffReach's JAX 0.8.1 CPU backend.
Pinned dependency SHAs and dirty/generated-build state are in
`ENVIRONMENT.json` and `PROVENANCE.json`. Configuration-level memory is
unavailable because configurations were not launched in isolated
peak-memory subprocesses.

## Non-citable evidence

- `20260730T015245Z` remains
  `provisional_due_to_known_protocol_defects`; its runtime ranking, Pareto,
  memory, requested-final-time, failure, and order-specific conclusions are
  not citable.
- `20260730T124958Z` failed the repository-hygiene gate.
- `20260730T141302Z` is rejected because its then-current Pareto computation
  partitioned dominance by tool rather than comparing tool families.
- All consolidation smoke runs are pipeline evidence only and are
  non-authoritative.
