# TORA-Q3 Stage-Parity Root-Cause Report

## Decision

The first numerical difference is A2 (the outward point sine enclosure), but its maximum bound error is only about `4.22e-15`. The first material and causally dominant difference is A3: sine composition remainder routing and analytic-tail semantics. The retained sine polynomial remains equal to roundoff scale; the interval remainder does not.

Replacing only K2 has no material effect. Replacing K2 and the observed Xiangru sine aggregate reduces the same-input local-remainder error by `98.174362%` or more at every selected replay point. The residual is traced to integration degree-overflow routing and, for full tubes, a secondary same-polynomial range semantic difference.

## Required stage table

| stage | input equal | coordinate map | max lower diff | max upper diff | max ULP | center diff | width diff | containment | first segment/leaf | classification |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| A0 | True | IDENTICAL_NORMALIZED_INPUT_TENSORS | 0 | 0 | 0 | 0 | 0 | bitwise_equal | -/- | numerically_negligible |
| A1 | True | IDENTICAL_NORMALIZED_INPUT_TENSORS | 0 | 0 | 0 | 0 | 0 | bitwise_equal | -/- | numerically_negligible |
| A2 | True | IDENTICAL_NORMALIZED_INPUT_TENSORS | 4.21885e-15 | 4.10783e-15 | 42 | 3.33067e-16 | 7.88258e-15 | torch_contains_xiangru | 1/0 | expected_outward_roundoff |
| A3 | True | IDENTICAL_NORMALIZED_INPUT_TENSORS | 0.00987269 | 0.014848 | 4546076069747733864 | 0.00806357 | 0.0145973 | mixed_overlapping | 1/0 | dominant_candidate |
| A4 | True | IDENTICAL_NORMALIZED_INPUT_TENSORS | 6.25976e-07 | 6.25976e-07 | 4514015402828782331 | 5.54595e-19 | 1.25195e-06 | torch_contains_xiangru | 1/0 | numerically_negligible |
| A5 | False | IDENTICAL_NORMALIZED_INPUT_TENSORS | 4.2884e-06 | 4.36662e-06 | 203712425705493207 | 5.45288e-07 | 8.65501e-06 | mixed_overlapping | 1/0 | numerically_negligible |
| A6 | False | IDENTICAL_NORMALIZED_INPUT_TENSORS | 9.35168e-14 | 4.00663e-13 | 4349892936118112771 | 2.00333e-13 | 4.0066e-13 | mixed_overlapping | 1/0 | numerically_negligible |
| A7 | False | IDENTICAL_NORMALIZED_INPUT_TENSORS | 3.72658e-05 | 0.000143033 | 4503599627370501 | 7.8777e-05 | 0.000128512 | mixed_overlapping | 1/0 | algorithm_semantics_difference |
| A8 | False | IDENTICAL_NORMALIZED_INPUT_TENSORS | 3.72658e-05 | 0.000143033 | 11406498417143699 | 7.8777e-05 | 0.00014149 | mixed_overlapping | 1/0 | algorithm_semantics_difference |
| A9 | False | IDENTICAL_NORMALIZED_INPUT_TENSORS | 0.00964209 | 0.00810347 | 9457559217478046 | 0.00482104 | 0.00964209 | mixed_overlapping | 1/0 | algorithm_semantics_difference |
| A10 | True | IDENTICAL_NORMALIZED_INPUT_TENSORS | 0 | 0 | 0 | 0 | 0 | bitwise_equal | -/- | numerically_negligible |
| A11 | True | IDENTICAL_NORMALIZED_INPUT_TENSORS | 1.11022e-16 | 1.11022e-16 | 1 | 5.55112e-17 | 2.22045e-16 | mixed_overlapping | 11/0 | expected_outward_roundoff |
| A12 | True | IDENTICAL_NORMALIZED_INPUT_TENSORS | 0 | 0 | 0 | 0 | 0 | bitwise_equal | -/- | numerically_negligible |

## T=1 attribution

The frozen common-control direct endpoint difference is `0.014211021942602`. `99.924210%` is already present before projection. A2 is far too small to explain it. A3 recurs on S0, S1, R1, R2, and F0; at R1 the sine substitution reduces the local-remainder error by `99.716333%`. The `0.014211` value is the ten-step accumulated consequence, not a one-step A2 rounding artifact.

## Segment 40 remainder attribution

The maximum pre-projection interval-remainder width is `1.21861858820087`. The broad `composition_overflow` ledger category accounts for `1.21861858820087`, while the current local `picard_residual` maximum is only `0.00126978318226989`. Affine composition labels carried prior remainder under this broad category; projection inflation remains roundoff-scale. At the exact segment-40 replay input, sine substitution removes `98.174362%` of the local-remainder error. Thus the accumulated category is `composition_overflow`, and its earliest material generator is A3 sine remainder semantics; A7/A8 integration overflow is secondary.

## Counterfactual scope

All substitutions are diagnostic only. Xiangru outputs are not used by the formal native runner. The reverse `Torch sine -> Xiangru integration` check changes the integration remainder width by up to `0.000449354`, confirming the A3 effect persists when the downstream implementation is swapped. Same-input A12 CROWN bounds are bitwise equal.
