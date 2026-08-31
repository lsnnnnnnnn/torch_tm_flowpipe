# C4 production performance profile

The numerical reference is commit `f34b5fa4155f5475a681411b627d68345ed401ea`.
Every timed production row excludes snapshot construction, JSON/CSV
serialization, and checkpoint export. The authoritative optimized numerical
root is recorded separately in `PROVENANCE.json`; evidence/package commits are
not used as the scientific denominator.

## Observer separation

The solver exposes three explicit observer modes:

- `production_no_observer`: no trace rows or range trace are constructed;
- `lightweight_counters`: scalar execution counters only;
- `full_evidence`: the complete raw-remainder and range evidence hooks.

On the frozen one-step observer comparison, all three lanes have identical
accepted/rejected decisions, endpoint and tube hashes, final remainder, queue
hash, checkpoint hash, replay count, and stop reason. The measured medians were
1.113105 s, 1.115630 s, and 1.223368 s respectively. Full-evidence trace
construction cost 0.110262 s and serialization cost 0.001909 s; the production
lane constructed zero trace rows. The machine-readable record, including peak
RSS and positive Python allocation counts, is
`production_vs_audit_overhead.csv`.

## Profile method and attribution

`experiments/profile_c4_reference_solver.py` uses `cProfile` for inclusive and
exclusive attribution, `tracemalloc` for positive Python allocation counts and
temporary bytes, explicit solver counters, and process high-water RSS. Every
function receives exactly one exclusive high-level bucket, so bucket totals do
not double count nested calls. The required windows are Brusselator steps
1–20, 1–100, 901–1000, and a representative VDP prefix. `hotspot_profile.csv`,
`call_count_matrix.csv`, `allocation_profile.csv`, and `flamegraph.txt` are the
authoritative outputs.

The initial 1–20 reference profile spent 63.412 s under profiler overhead. The
inclusive accepted-boundary reset path accounted for 52.8% and queue
preparation for 45.4%. At steps 996–1000, the reset path accounted for 80.5%,
queue preparation for 53.6%, and queue propagation/validation for about 38%.
The repeated work was construction and scalar dispatch over independent 2×2
interval owner payloads. The same accepted-boundary SR mechanism is used by
the C3 VDP and generic C4 Brusselator lanes.

## Single authorized optimization

The only numerical optimization is packed CPU-float64 accepted-boundary SR
owner propagation. Owner matrices and columns are packed into tensors and the
independent owner dimension is evaluated together. The inner `k` loop and the
outer owner accumulation loop remain in their original order, and every
multiplication and addition keeps its original outward `torch.nextafter`.
The scalar-object implementation remains as a test oracle. Queue validation is
also performed over packed immutable payloads, while every public entry point
continues to validate and fail closed.

This is not a cache and stores no range, proposal, remainder, normalization,
or owner-generation dependent result. No replay, range policy, queue policy,
cutoff, term order, or subset decision changed. Deterministic tests cover owner
counts 0/1/7/99, dimensions 1/2/3, subnormals, mutable tensor tamper,
nonfinite payloads, checkpoint/resume, rejection rollback, legacy/C4-off
behavior, and the existing C3/C4 regression suites. The frozen step-996
endpoint, tube, reset, queue, remainders, and replay counters are bitwise
identical. Its three-run median improved from 6.357847 s to 2.398168 s
(2.65×), satisfying the profile authorization threshold.

The production gates deliberately judge whole-prefix and full-solver latency,
not that tail microbenchmark. `optimization_result.json`,
`prefix_runtime_matrix.csv`, and `full_runtime_matrix.csv` are authoritative
for the final pass/fail classification. Per the one-candidate rule, a correct
optimization that misses those gates is retained and reported without stacking
a second numerical optimization.

