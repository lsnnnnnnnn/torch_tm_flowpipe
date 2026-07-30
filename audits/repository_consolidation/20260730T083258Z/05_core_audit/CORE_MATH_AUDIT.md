# Core mathematics audit

Audited canonical modules:

| Layer | Implementation | Executable evidence | Consolidation finding |
|---|---|---|---|
| Interval | `interval.py` | `tests/test_interval.py` | invalid order and NaN rejected; zero-crossing reciprocal rejected; arithmetic expands with `nextafter`; infinity remains representable |
| Polynomial | `polynomial.py` | `tests/test_polynomial.py` | exponent arity/nonnegativity checked; zero terms merged; monomials now stored in deterministic exponent order; truncation returns kept and dropped parts |
| Taylor model | `taylor_model.py`, `tm_vector.py` | `tests/test_taylor_model.py`, `tests/test_tm_vector.py` | polynomial/remainder/domain arity checked; multiplication moves dropped range to remainder; clone now deep-copies mutable tensors |
| One step | `flowpipe.py` | `tests/test_flowpipe.py`, analytic Riccati cases | local time, validation, raw endpoint, and failure status are explicit; failed status is not treated as completion |
| Multi step | `flowpipe.py` | mode and failure-carry regressions | `range_only`, `dependency_preserving`, and Flowstar-style reset are distinct; a failed segment now stops propagation and is never carried |
| Protocol | `protocol/` | `tests/test_protocol_contracts.py` | versioned identity, timing, horizon, failure, raw-bound, eligibility, repetition, and Pareto contracts fail closed |

Soundness boundary:

- Torch/JAX float64 results are not universal real-arithmetic proofs.
- `torch.nextafter` expansion is used where implemented, but no claim is made
  that every backend operation is directed-rounding sound.
- analytic interval solutions are correctness gates where available;
  deterministic nonlinear sampling is only regression sanity evidence.
- configuration memory is intentionally unavailable because no isolated
  per-configuration peak measurement is implemented.

Historical defect tests added during this audit cover timer completion
boundaries, repeated-only primary rows, eligibility-before-Pareto, requested
versus successful horizon, missing failure status, Flowstar order/basis
identity, raw/tightened separation, final-point versus trajectory validation,
non-empty output reuse, stale resume manifests, and config identity collision.
