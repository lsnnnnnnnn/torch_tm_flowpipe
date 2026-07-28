# Torch endpoint semantics audit

`FlowpipeSegment` now exposes three distinct objects:

- `tube_tm`: the validated Taylor model on `tau ∈ [0,h]`;
- `endpoint_raw_tm`: direct substitution `tau=h` in `tube_tm`;
- `endpoint_tightened_tm`: the existing supplemental fixed-time residual
  recomputation.

For compatibility, `final_tm` remains the tightened object. Direct
`FlowpipeSegment` construction defaults both new endpoint fields to `final_tm`;
the step builder populates them explicitly and records
`endpoint_semantics`, `endpoint_tightening_applied`, and the validation method.

For a degree-`p` Picard polynomial \(P\), fixed-time tightening evaluates

\[
R_h = X_0 + \int_0^h f(P(s)+R(s))\,ds - P(h),
\]

with polynomial terms above the configured degree moved into intervals and all
pre-existing interval remainders retained through interval arithmetic. The time
variable is then removed on the singleton domain `tau=h`; state-generator
domains remain unchanged. This is a different enclosure operation from direct
substitution and therefore is supplemental, never a primary comparison row.

Regression tests cover exact Riccati and harmonic endpoints plus seeded random
scalar polynomial vector fields and starting boxes. They verify finiteness,
raw-endpoint-in-tube containment, exact/sampled endpoint containment, distinct
raw/tightened objects, and preservation of prior interval remainder
contributions.

At Riccati `h=0.01`:

| Endpoint | Lower | Upper | Width | Remainder width |
|---|---:|---:|---:|---:|
| raw | -1.2500001252500047e-05 | 0.10011250000125255 | 0.10012500000250504 | 0.00012500000250500041 |
| tightened | -2.5015065006565793e-08 | 0.10010022512756506 | 0.10010025014263006 | 0.00010025014263001466 |

The exact upper endpoint is `0.1001001001001001`. Tightening reduces excess
width by roughly 165×, confirming H2: it materially caused the apparent
Riccati tightness when compared with other tools' raw endpoints. Protocols A–C
use only `endpoint_raw_tm`; native raw and legacy-tightened carry are separate
variants.
