# Legacy narrow-RPC QUAD interval gate: corrected Decimal recheck

This is a **new saved-evidence recheck**, separate from the historical
[interval audit](../native_quad_independent_interval_gate_20261002_001/README.md).
It reads the original-library `_005` and VAR-tail copied-library `_007`
RPC and observer CSV files. It runs no Flow*, CROWN, neural network, or old
reachability job, and changes none of those receipts. The reproducible
read-only calculation is [`reaudit_saved.py`](reaudit_saved.py), using the
corrected [`tools/check_native_quad_fullset_interval_gate_nohash.py`](../../../../../tools/check_native_quad_fullset_interval_gate_nohash.py).

The historical interval checker used Decimal unary minus and `abs` on
high-precision endpoints outside its directed 60-digit contexts. Python's
ambient 28-digit context could round these operations inward. The
[method check](METHOD_SELF_CHECK.json) reproduces all three defects for an
exact binary64 decimal and a Taylor remainder, then checks the corrected
exact sign, absolute-value, float-constructor, remainder, and time-step
operations. The corrected checker also rejects non-exact binary64
center/radius reconstruction of the float32 residual endpoints. Each of
the three saved residual channels in both arms passes that check.

Each arm again passes 1,000 strict Picard self-inclusions over the first
`h=0.005` paper-equation ODE step. The [original fixed audit](ORIGINAL_FIXED_AUDIT.json)
contains **23/32** saved numerical comparisons; the
[repaired fixed audit](REPAIRED_FIXED_AUDIT.json) contains **27/32**. The
32 observer keys and saved numerical bounds agree exactly with the
historical receipts, and no Boolean comparison changes. These are now
results of the corrected outward interval calculation; the old calculation
alone was not a valid directed-rounding certificate. The recheck also
inspects 64 finite CSV bounds per arm against the observer's 17-digit
binary64 output plus one outward ULP.

For the repaired arm, its five broad-interval gaps are composed `x6`,
composed and pre-column `x10`, and composed and pre-column `x11`.
[New exact-rational algebraic calculations](RESULT.json) on the same saved
RPC and observer bounds place all five physical-state bounds inside their
saved numerical columns. Combining the corrected 27 with those exact five
establishes **20/20 composed physical-state numerical comparisons** for
this saved RPC domain and this one step. A physical-state bound falling in
a pre-column is **not** a proof that the pre symbolic set is enclosed.
The original arm's nine broad-interval gaps remain gaps of this method;
no algebraic promotion of that arm is claimed.

The saved RPC lower limits for `x1`–`x3` are
`-0.39999999999999997`, while the frozen C++ source first box begins at
binary64 `-0.4`. The exact missing lower strip is
`1/18014398509481984`, one binary64 ULP. Thus even the repaired 20/20
statement **does not cover the complete source-defined first box**.
It also does not validate the CROWN/NN calculation, Flow* parsing or
internal floating-point soundness, other boxes, later small steps or
control calls, or the `T=5` reach-and-remain property. The native octagon
production gate remains **closed**.
