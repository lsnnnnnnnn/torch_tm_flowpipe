# TORA-Q3 Algorithm-Aligned Native Implementation

## Result

`algorithm_aligned_q3` is a new PyTorch-native complete-Q3 lane.  It retains
the frozen K2 polynomial Picard schedule and ten remainder Picard rounds, but
replaces the generic sine composition remainder route with an independently
derived, outward, fixed-shape route.  It does not replace or modify
`baseline_native_k2` or `k3_picard`.

All G0--G4 one-step gates pass.  The formal common-control B48 T20 replay also
passes all 200 segments, with the same accepted status as the frozen Xiangru
plant records and a minimum physical property margin of
`0.287069725086306`.

## Independent soundness argument

Write the sine input Taylor model as

```text
x = c + p + r,
```

where `c` is the stored constant coefficient, `p` is the nonconstant Q3
polynomial, and `r` is the interval remainder.  The retained polynomial is

```text
sin(c) + cos(c) p - sin(c) Q3(p^2) / 2.
```

The implementation encloses every discarded contribution as follows:

1. `sin(c)` and `cos(c)` use the existing outward rational Maclaurin point
   enclosure, never device `sin` or `cos` in the formal path.
2. The sound Q3 product routes `p^2 - Q3(p^2)`, retained-route binary64 error,
   and coefficient uncertainty into the interval remainder.
3. The signed input-remainder correction is evaluated as
   `cos(c) r - sin(c) (2 p r + r^2) / 2` with outward interval add and
   multiply.  In particular, `2 p r` is not incorrectly forced nonnegative.
4. The third-order Lagrange remainder is enclosed by
   `-cos(c + theta delta) delta^3 / 6`, where `delta = p + r` and the cosine
   interval covers the complete line-segment hull from `c` to `c + delta`.
   Endpoint evaluation and every possible multiple-of-pi extremum in the
   proved argument domain are included outwardly.
5. Non-finite values, invalid remainders, unsupported dtype, wrong Q3 shape,
   changed K2/K10 schedule, and composition radius outside the proved domain
   fail closed.

This identity proves containment independently of randomized samples.
Randomized and high-precision samples are auxiliary regression checks only.
Stage evidence had suggested a one-sided square-difference route, but direct
evaluation of the signed `2 p r` term shows that a nonzero input remainder can
make that difference negative.  The new lane therefore aligns the retained
quadratic semantics while preserving the signed soundness compensation.

## Soundness and regression tests

`tests/test_tora_algorithm_aligned_q3.py` covers:

- scalar zero and affine retained cases;
- positive and negative sine-extremum crossings;
- nonzero input remainder;
- explicit complete-Q3 degree overflow and retained-route roundoff;
- a high-precision `mpmath` oracle;
- randomized point containment as non-proof sanity evidence;
- exact held-control derivative and final remainder invariants;
- CPU float64 and CUDA float64 containment of the same oracle;
- eager/compiled bitwise equality or compiled-outward containment;
- frozen baseline source hashes;
- invalid-input fail-closed behavior;
- formal artifact G0--G4 and T20 protocol invariants.

The focused result is `13 passed`.  The two frozen baseline files remain
byte-identical:

```text
batched_dense_tm.py  7198489a4adcce07ad741a021da96e7e3ca4a033ba7d947d02edb88e141f1980
tora_q3.py           e343b2a60c7bc3861f655f952b4333744e6ecd9929543a469a2f4a542f59b63b
```

On a representative affine input with nonzero remainder, the new signed route
has strictly smaller interval-remainder width than the frozen generic order-2
sine route.

## One-step parity gates

All comparisons use the identity complete-Q3 coordinate map.  No frozen
output is supplied to the native computation.

| Gate | Input | Candidate/reference accepted | Max coefficient difference | Max local-remainder center difference | Max endpoint center difference |
| --- | --- | ---: | ---: | ---: | ---: |
| G0 | one leaf, initial | 1/1, equal | 5.110e-10 | 1.952e-05 | 1.738e-05 |
| G1 | B48, initial | 48/48, equal | 5.110e-10 | 1.966e-05 | 1.752e-05 |
| G2 | B48, segment 2 | 48/48, equal | 4.335e-10 | 1.892e-05 | 1.711e-05 |
| G3 | B48, R1 at T=1 | 48/48, equal | 4.664e-10 | 1.780e-05 | 1.713e-05 |
| G4 | B48, R2 at T=4 | 48/48, equal | 1.322e-09 | 2.572e-05 | 2.512e-05 |

The formal JSON additionally records polynomial range, lower/upper remainder,
tube, all subset margins, accepted leaves, and property-margin differences.

## Common-control B48 T20

Each of the 20 periods restarts from the same frozen Xiangru state box and
control interval.  Ten native segments are then propagated inside that period.
The previous-period native enclosure is never substituted for the next
period's frozen input, and reference outputs are never used as native inputs.

| Quantity | Width-ratio median | Width-ratio maximum | Maximum absolute center difference | Maximum absolute width difference |
| --- | ---: | ---: | ---: | ---: |
| Endpoint | 1.81738 | 12.1162 | 0.0236115 | 1.22604 |
| Tube | 1.63951 | 5.50192 | 0.0236115 | 1.22535 |
| Physical interval remainder | 5.851e-10 | 1.08790 | 0.0236115 | 0.753860 |

Exactly zero reference widths are excluded from ratios rather than assigned a
fabricated value.  The endpoint and tube statistics remain separate.  These
numbers are a plant comparison under common control, not an independent native
closed-loop T20 certificate.

## Public evidence

- `outputs/tora_q3_stage_parity_fused_20260809/algorithm_aligned/one_step_gates.json`
- `outputs/tora_q3_stage_parity_fused_20260809/algorithm_aligned/common_control_t20.json`
- `outputs/tora_q3_stage_parity_fused_20260809/algorithm_aligned/summary.json`

The public records contain only source/config/input hashes and aggregate
metrics.  They contain no raw stage tensors, per-leaf traces, server paths,
controller bytes, or private source.
