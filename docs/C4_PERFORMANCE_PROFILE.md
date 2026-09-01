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
exclusive attribution, `tracemalloc` for positive Python allocation counts,
lightweight wrappers around Python-visible tensor-producing APIs for temporary
tensor result counts and logical bytes, explicit solver counters, and process
high-water RSS. Every
function receives exactly one exclusive high-level bucket, so bucket totals do
not double count nested calls. The required windows are Brusselator steps
1–20, 1–100, 901–1000, and a representative VDP prefix. `hotspot_profile.csv`,
`call_count_matrix.csv`, `allocation_profile.csv`, and `flamegraph.txt` are the
authoritative outputs.

The formal 1–20 and 1–100 reference profiles spent 42.372 s and 255.912 s under
profiler overhead; the VDP-20 window took 14.697 s. In the 1–100 window the
inclusive accepted-boundary reset path accounted for 56.10% and SR preparation
for 45.81%. At steps 901–1000, which took 662.214 s under profiling, the reset
path accounted for 81.61%, SR preparation for 51.63%, queue validation for
38.43%, and queue propagation for 37.21%. These inclusive figures overlap as
call paths. The mutually exclusive tail buckets were 55.16% Python
orchestration/allocation, 29.86% outward interval accounting, and 6.64% SR
history propagation. The repeated work was construction and scalar dispatch
over independent 2×2 interval owner payloads. The same accepted-boundary SR
mechanism is used by the C3 VDP and generic C4 Brusselator lanes.

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

## Formal production gates

All formal rows were pinned to CPU 0 and generated from clean detached
scientific roots. The 100-step Brusselator reference and optimized medians were
205.223872 s and 184.237304 s (1.113911×). The 300-step medians were
667.551832 s and 521.273988 s (1.280616×). The single full-T20 runs were
4060.693636 s and 1971.634787 s (2.059557×). Full-run peak RSS was 474,890,240
bytes and 477,151,232 bytes, so the 1.5× memory gate passed. The prefix gates
did not reach 2×, so the overall CPU speed gate failed even though the full-run
gate passed. VDP and Brusselator outputs remained exact, and no second
optimization was attempted.
