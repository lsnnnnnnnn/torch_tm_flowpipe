# What can be shared between weighted rounds

Read-only source analysis after the fixed-batch candidate was prepared. This
does not add a numerical candidate or use the historical phase timings as a
profile of the current paper-contract run.

The frozen `weighted_validation._map` inserts `candidate / h` into the time
coefficient of its **interval** polynomial before calling
`sparse_exec.exec_valid_s(..., interval_coefficients=True)`. Both the interval
polynomial output `field` and the strict `tail` can therefore change when the
accepted first-round image becomes the second-round candidate. Reusing the
whole first-round polynomial field or trigonometric series would not preserve
the current validation computation. Reusing only a point nominal field would
require a new validated error decomposition, not a scheduling-only change.

The following terms are independent of the candidate within one call:

| Term | Present treatment | Safe reuse scope |
|---|---|---|
| `_plan`, specialized tape, integration weights, support positions, factors | Cached by original plan | Already reused |
| Point coefficients expanded into the full support | New fused body builds once before both maps | Already reused by this candidate |
| Initial-polynomial equality and fixed input finite checks | New fused body checks once | Already reused when committing both successful rounds |
| `iv.from_point(point[:, :, shifted_positions]) * h_iv` subtraction | Recomputed inside each unchanged `_map` | Could compute once per chunk and supply to a separately qualified map body |
| Point-to-interval base coefficient tensor before time-slope injection | Recomputed inside each map | Could share an owned immutable base, but each round still needs a writable copy |
| Direct-variable trig constant `c` and `sin_iv(c)`, `cos_iv(c)` | Recomputed by trig bodies | Only if the input is demonstrably the same direct variable/plan and constant interval in both calls |

The subtraction/base conversion work is small relative to an interval ODE
execution. Its actual share is not measured here. The candidate deliberately
keeps `_map` unchanged, so it does not claim that saving these operations will
close a large remaining timing gap.

For QUAD, the existing `trig_direct_reuse._bind` already reuses an identical
direct-variable **same-op** trig series within one `exec_valid_s` call. It
resets its memo on every call and keys on `s.op`, so it neither shares sin and
cos power chains nor reuses a series across the two different candidates.

A potentially larger next code change is **within-map** sin/cos power-chain
sharing for the same direct variable and identical chain/accumulator bindings.
In `_trig_series_valid_g`, the sequence of `(pow_c, pow_r)` and the associated
three-field multiplication cache records does not depend on `cos_cycle`;
the derivative wheel, accumulated sine/cosine coefficients, and Lagrange tail
do. A separate candidate could reuse only owned snapshots of those powers and
their original cache records, while retaining every sine/cosine accumulation
and each strict Lagrange tail. It must preserve the advancing `ctx` support
state, all original cache/tail offsets, and rounding order. This is more
invasive than the delivered fusion and needs its own direct tensor/cache/tail
qualification; it is not implemented or approved for execution here.

Source locations inspected:

- `research/gpu_verified_20260930/source/engine/src/flowstar_gpu/weighted_validation.py`: `_plan`, `_map`, `validate_failed`, `refine_accepted`.
- `research/gpu_verified_20260930/source/engine/src/flowstar_gpu/sparse_exec.py`: `exec_valid_s`, the SR-capacity `_weighted_graphs` reclamation, and the accepted/failed weighted call order.
- `research/gpu_verified_20260930/source/engine/src/flowstar_gpu/elementary.py`: `_trig_series_valid_g` and `rec_series_valid_g`.
- `research/gpu_verified_20260930/source/adapters/quad_fullbatch_p3_20260928/trig_direct_reuse.py`: metadata plan and call-local `_bind`; its digest-based `install` was read only and never invoked.
- `tools/archcomp26_quad_paper_p3_nohash.py`: current paper prepare uses `trig._bind(se)` and the original strict map.

No source file above was modified, no existing checker was run, and no remote
experiment or content digest operation was performed for this analysis.
