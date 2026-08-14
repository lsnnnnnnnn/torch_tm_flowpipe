# Flow*–Torch actual-path stage ledger — 2026-08-13

Both ledgers come from production arithmetic with read-only observation:

- Flow*: pinned `b85a3211748cb77b736fe4ad42ee02d8d2b81148`, public
  `ODE<Real>::reach`, fixed symbolic-remainder `Flowpipe::advance`, 791 step-1
  records plus actual segment/endpoint polynomial and final ranges;
- Torch: `dense_picard_validate_step` used by the legacy dense flowpipe lane,
  four polynomial Picard images, the 18-node raw-remainder expression DAG,
  validator image/margin/decision, returned segment, endpoint, reset, and
  step-2 prestate.

Every row records source file/function/line, source SHA and observer-patch hash,
stage and iteration, canonical basis/support, binary64 bits and hexfloat,
the exact dyadic rational decoded from that encoding, interval precision and
rounding provenance, remainder ownership, and chained input/output hashes.
The full JSON ledger carries coefficients and intervals; CSV is only an index.

Flow* observation points include the fixed-path calls around
`Continuous.cpp:2373`, `2413`, `2453`, `2481`, and `2483`. Degree truncation
and cutoff intervals are emitted at the actual operations in
`TaylorModel.h:479`, `865`, `869`, and `1111`. Torch's optional observer is
forwarded through `dense_picard_validate_step`; its default is `None` and an
observer-on/off replay verifies identical status, trace, coefficients,
remainders, endpoint, and subset margins.

The first cross-tool binary difference is already in the normalized initial
TM radii. It is classified `UNDER_ENCLOSURE_WITNESS`, not merely a coefficient
ULP report, because the exact-rational set required by the contract is absent
from both point-coefficient TMs. Later findings are:

- Picard iterations 1–3 have different exact construction schedules: Flow*
  uses staged RHS degree `i-1`, Torch uses complete O4. This is a real
  intermediate semantic difference.
- The independent exact oracle proves their fourth mathematical Picard images
  equal: 13 x terms and 18 y terms.
- Both segment polynomial ranges contain the exact natural-range oracle.
- Torch's endpoint is narrower because it substitutes `tau=h` and merges
  monomials before ranging; both final endpoint boxes contain the independently
  certified true solution.

Only the approved classification vocabulary appears in the machine audit.
