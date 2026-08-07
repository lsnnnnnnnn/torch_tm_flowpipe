# TORA-Q3 GPU bottleneck report

This report describes profiler event counts, not formal runtime. The raw Chrome trace remains private.

## Baseline totals

- `aten::item`: 77
- `aten::_local_scalar_dense`: 77
- paired host-scalar synchronization estimate: 77
- `aten::to`: 80

## Largest mathematical stages

| stage | host-sync estimate | share | aten::to |
|---|---:|---:|---:|
| sin_tm | 26 | 33.766% | 26 |
| affine_composition | 4 | 5.195% | 15 |
| remainder_picard_round_01 | 4 | 5.195% | 0 |
| remainder_picard_round_02 | 4 | 5.195% | 0 |
| remainder_picard_round_03 | 4 | 5.195% | 0 |
| remainder_picard_round_04 | 4 | 5.195% | 0 |
| remainder_picard_round_05 | 4 | 5.195% | 0 |
| remainder_picard_round_06 | 4 | 5.195% | 0 |
| remainder_picard_round_07 | 4 | 5.195% | 0 |
| remainder_picard_round_08 | 4 | 5.195% | 0 |
| remainder_picard_round_09 | 4 | 5.195% | 0 |
| remainder_picard_round_10 | 4 | 5.195% | 0 |
| full_q3_step | 2 | 2.597% | 3 |
| initial_remainder_picard | 2 | 2.597% | 0 |
| final_acceptance_property_decision | 1 | 1.299% | 2 |
| unscoped | 1 | 1.299% | 2 |
| composed_acceptance_property_decision | 1 | 1.299% | 0 |
| tora_rhs | 0 | 0.000% | 26 |
| composed_exact_endpoint_substitution | 0 | 0.000% | 2 |
| endpoint_projection | 0 | 0.000% | 2 |

## Largest host-sync callsites

| stage | source | host-sync estimate | share |
|---|---|---:|---:|
| sin_tm | `<stack-unavailable>:0:unknown` | 26 | 33.766% |
| remainder_picard_round_01 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_02 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_03 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_04 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_05 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_06 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_07 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_08 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_09 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| remainder_picard_round_10 | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| affine_composition | `<stack-unavailable>:0:unknown` | 4 | 5.195% |
| initial_remainder_picard | `<stack-unavailable>:0:unknown` | 2 | 2.597% |
| full_q3_step | `<stack-unavailable>:0:unknown` | 2 | 2.597% |
| final_acceptance_property_decision | `<stack-unavailable>:0:unknown` | 1 | 1.299% |
| composed_acceptance_property_decision | `<stack-unavailable>:0:unknown` | 1 | 1.299% |
| unscoped | `<stack-unavailable>:0:unknown` | 1 | 1.299% |

## Largest device/dtype conversion callsites

| stage | source | aten::to | share |
|---|---|---:|---:|
| tora_rhs | `<stack-unavailable>:0:unknown` | 26 | 32.500% |
| sin_tm | `<stack-unavailable>:0:unknown` | 13 | 16.250% |
| sin_tm | `<stack-unavailable>:0:unknown` | 13 | 16.250% |
| affine_composition | `<stack-unavailable>:0:unknown` | 5 | 6.250% |
| affine_composition | `<stack-unavailable>:0:unknown` | 3 | 3.750% |
| affine_composition | `<stack-unavailable>:0:unknown` | 3 | 3.750% |
| full_q3_step | `<stack-unavailable>:0:unknown` | 2 | 2.500% |
| affine_composition | `<stack-unavailable>:0:unknown` | 2 | 2.500% |
| affine_composition | `<stack-unavailable>:0:unknown` | 2 | 2.500% |
| full_q3_step | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| final_acceptance_property_decision | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| final_acceptance_property_decision | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| composed_exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| composed_exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| unscoped | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| unscoped | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| endpoint_projection | `<stack-unavailable>:0:unknown` | 1 | 1.250% |
| endpoint_projection | `<stack-unavailable>:0:unknown` | 1 | 1.250% |

The paired synchronization estimate is `max(aten::item, aten::_local_scalar_dense)` so one scalar extraction is not counted twice. Stage and callsite CSV files retain the complete sanitized aggregation.
