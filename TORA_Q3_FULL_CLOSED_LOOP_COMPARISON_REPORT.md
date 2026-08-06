# Native Torch TORA-Q3 full closed-loop comparison

Status: **T1 PASS; T5 gate FAILS at segment 44; T10/T20 N/A.**

This is the native lane: Torch projects its own state enclosure, computes its
own controller bounds from the externally supplied, hash-verified original
ONNX, refreshes every ten segments, and holds `u1` between refreshes.

The nominal controller gate passes with max error `5.1034085e-7` against the
ONNX float32 reference (tolerance `1e-6`).  Initial B48 CROWN bounds match the
Xiangru observation within `3.5527137e-15` (2 ULP at the worst scalar).

T1 completes.  At the first refresh, the controller is no longer receiving
the same box: the method-native plant/state projection differs from Xiangru by
up to `0.0142110`.  The corresponding control interval differs by up to
`0.171190`; Xiangru's interval contains all 48 Torch intervals at this refresh.
The divergence grows and reverses containment later because each lane closes
the loop around its own enclosure.

Torch completes 43 segments and certifies through T=4.3.  At segment 44
(physical time 4.4), leaf 0's tube property margins for `[x1,x2,x3,x4]` are
`[1.462986, 1.114402, -0.0420120, 0.0626519]`; `x3` crosses the frozen ±2
property.  The run fails closed.  T5, T10, and T20 widths and runtime are
therefore `N/A`; the T4.3 endpoint must not be compared with Xiangru T5/T20.

This is not evidence of a different controller file or nominal-network bug.
It is classified as method-native plant/state-projection enclosure growth
feeding a different, sound controller input domain and then accumulating
through feedback.  The correct next technical direction would be a separately
audited tighter range/reconditioning method, not a larger seed, a weaker
property, or a substituted controller.

Exact per-refresh abs/ULP differences and containment counts, the T1 common-
horizon width table, failure detail, source hashes, and pre-fix actual-bug
hashes are in `outputs/tora_q3_native_matched_20260806/full_closed_loop/`.
