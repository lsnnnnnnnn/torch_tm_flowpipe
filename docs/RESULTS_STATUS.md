# Results status

No cross-tool speedup, Pareto frontier, winner or runtime/tightness ranking is
citable.  Workloads differ in plant/controller, partitions, effective support,
device and timing boundary, and every row is currently
`primary_comparison_eligible=false`.

## Citable reproduction facts

- Xiangru CROWN-Reach/Flow* TORA B12: exact selected-field reproduction, T=20,
  `VERIFIED`, no Flow* termination.
- Xiangru complete-Q3 DR-RP TORA B48: T=20 and full-tube property reproduced across
  2,850 non-timing numbers within the author's `1e-6` tolerance (maximum absolute
  error `1.421e-13`).
- Xiangru DiffReach CPU U0: the failed/conservative NPZ is byte-identical; only
  66.22% of returned initial shrink flags are true.  This is a reproduced failure,
  not verification.
- Stock Flow* official VDP order 4: clean source, official program, 290 segments,
  T=10 and native safe verdict.  Upstream supplies no raw reference.
- Upstream DiffReach official VDP: official README command, 64 partitions, 1,000
  steps and T=10.  Upstream supplies no raw reference, and its returned flag does
  not cover every roundoff-inclusive refinement.
- Our Torch order-4 H10 command: no lane reaches T=10; best fresh adaptive horizon
  is T=6.049038 before the declared wall cap.  It is a partial failed result.

## Open correctness gates

The existing Flow* scalar-affine generated diagnostic still misses both analytic
endpoint corners by about `3.5e-10` at worst and misses the upper final path sample.
The official VDP native completion does not close that backend-wide gate.

DiffReach `src/picard.py` returns the initial contraction flag while later
roundoff-inclusive refinement failures are not combined into that returned flag.
All-true initial flags therefore do not establish a formal full self-map.

Xiangru Q3 uses outward host rounding for controller composition, but its dynamics
interval add/mul/sin/cos path uses ordinary float64 operations.  Its fresh result
is recorded as empirical, not formal.

See [native matrix](NATIVE_REPRODUCTION_MATRIX.md), [Xiangru reproduction](XIANGRU_NATIVE_REPRODUCTION.md),
and [code audit](XIANGRU_VS_OUR_TORCH_TM_CODE_AUDIT.md).
