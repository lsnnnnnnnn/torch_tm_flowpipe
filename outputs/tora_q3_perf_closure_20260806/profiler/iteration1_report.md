# TORA-Q3 GPU bottleneck report

This report describes profiler event counts, not formal runtime. The raw Chrome trace remains private.

## Baseline totals

- `aten::item`: 6585
- `aten::_local_scalar_dense`: 6585
- paired host-scalar synchronization estimate: 6585
- `aten::to`: 12137

## Largest mathematical stages

| stage | host-sync estimate | share | aten::to |
|---|---:|---:|---:|
| sin_tm | 3293 | 50.008% | 10308 |
| tora_rhs | 1440 | 21.868% | 181 |
| affine_composition | 160 | 2.430% | 669 |
| remainder_picard_round_01 | 124 | 1.883% | 44 |
| remainder_picard_round_02 | 124 | 1.883% | 44 |
| remainder_picard_round_03 | 124 | 1.883% | 44 |
| remainder_picard_round_04 | 124 | 1.883% | 44 |
| remainder_picard_round_05 | 124 | 1.883% | 44 |
| remainder_picard_round_06 | 124 | 1.883% | 44 |
| remainder_picard_round_07 | 124 | 1.883% | 44 |
| remainder_picard_round_08 | 124 | 1.883% | 44 |
| remainder_picard_round_09 | 124 | 1.883% | 44 |
| remainder_picard_round_10 | 124 | 1.883% | 44 |
| initial_remainder_picard | 119 | 1.807% | 44 |
| polynomial_picard_k1 | 100 | 1.519% | 44 |
| polynomial_picard_k2 | 100 | 1.519% | 44 |
| full_q3_step | 25 | 0.380% | 3 |
| composed_exact_endpoint_substitution | 21 | 0.319% | 78 |
| endpoint_projection | 20 | 0.304% | 78 |
| exact_endpoint_substitution | 20 | 0.304% | 78 |

## Largest host-sync callsites

| stage | source | host-sync estimate | share |
|---|---|---:|---:|
| sin_tm | `<stack-unavailable>:0:unknown` | 3293 | 50.008% |
| tora_rhs | `<stack-unavailable>:0:unknown` | 1440 | 21.868% |
| affine_composition | `<stack-unavailable>:0:unknown` | 160 | 2.430% |
| remainder_picard_round_01 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_02 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_03 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_04 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_05 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_06 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_07 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_08 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_09 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| remainder_picard_round_10 | `<stack-unavailable>:0:unknown` | 124 | 1.883% |
| initial_remainder_picard | `<stack-unavailable>:0:unknown` | 119 | 1.807% |
| polynomial_picard_k1 | `<stack-unavailable>:0:unknown` | 100 | 1.519% |
| polynomial_picard_k2 | `<stack-unavailable>:0:unknown` | 100 | 1.519% |
| full_q3_step | `<stack-unavailable>:0:unknown` | 25 | 0.380% |
| composed_exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 21 | 0.319% |
| exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 20 | 0.304% |
| endpoint_projection | `<stack-unavailable>:0:unknown` | 20 | 0.304% |

## Largest device/dtype conversion callsites

| stage | source | aten::to | share |
|---|---|---:|---:|
| sin_tm | `<stack-unavailable>:0:unknown` | 6552 | 53.984% |
| sin_tm | `<stack-unavailable>:0:unknown` | 2535 | 20.887% |
| sin_tm | `<stack-unavailable>:0:unknown` | 949 | 7.819% |
| affine_composition | `<stack-unavailable>:0:unknown` | 576 | 4.746% |
| tora_rhs | `<stack-unavailable>:0:unknown` | 181 | 1.491% |
| sin_tm | `<stack-unavailable>:0:unknown` | 130 | 1.071% |
| sin_tm | `<stack-unavailable>:0:unknown` | 129 | 1.063% |
| affine_composition | `<stack-unavailable>:0:unknown` | 74 | 0.610% |
| exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 60 | 0.494% |
| composed_exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 60 | 0.494% |
| endpoint_projection | `<stack-unavailable>:0:unknown` | 60 | 0.494% |
| polynomial_picard_k1 | `<stack-unavailable>:0:unknown` | 38 | 0.313% |
| polynomial_picard_k2 | `<stack-unavailable>:0:unknown` | 38 | 0.313% |
| initial_remainder_picard | `<stack-unavailable>:0:unknown` | 38 | 0.313% |
| remainder_picard_round_01 | `<stack-unavailable>:0:unknown` | 38 | 0.313% |
| remainder_picard_round_02 | `<stack-unavailable>:0:unknown` | 38 | 0.313% |
| remainder_picard_round_03 | `<stack-unavailable>:0:unknown` | 38 | 0.313% |
| remainder_picard_round_04 | `<stack-unavailable>:0:unknown` | 38 | 0.313% |
| remainder_picard_round_05 | `<stack-unavailable>:0:unknown` | 38 | 0.313% |
| remainder_picard_round_06 | `<stack-unavailable>:0:unknown` | 38 | 0.313% |

The paired synchronization estimate is `max(aten::item, aten::_local_scalar_dense)` so one scalar extraction is not counted twice. Stage and callsite CSV files retain the complete sanitized aggregation.
