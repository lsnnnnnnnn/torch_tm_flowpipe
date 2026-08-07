# TORA-Q3 GPU bottleneck report

This report describes profiler event counts, not formal runtime. The raw Chrome trace remains private.

## Baseline totals

- `aten::item`: 253
- `aten::_local_scalar_dense`: 253
- paired host-scalar synchronization estimate: 253
- `aten::to`: 80

## Largest mathematical stages

| stage | host-sync estimate | share | aten::to |
|---|---:|---:|---:|
| sin_tm | 26 | 10.277% | 26 |
| remainder_picard_round_01 | 20 | 7.905% | 0 |
| remainder_picard_round_02 | 20 | 7.905% | 0 |
| remainder_picard_round_03 | 20 | 7.905% | 0 |
| remainder_picard_round_04 | 20 | 7.905% | 0 |
| remainder_picard_round_05 | 20 | 7.905% | 0 |
| remainder_picard_round_06 | 20 | 7.905% | 0 |
| remainder_picard_round_07 | 20 | 7.905% | 0 |
| remainder_picard_round_08 | 20 | 7.905% | 0 |
| remainder_picard_round_09 | 20 | 7.905% | 0 |
| remainder_picard_round_10 | 20 | 7.905% | 0 |
| initial_remainder_picard | 18 | 7.115% | 0 |
| affine_composition | 4 | 1.581% | 15 |
| full_q3_step | 2 | 0.791% | 3 |
| final_acceptance_property_decision | 1 | 0.395% | 2 |
| unscoped | 1 | 0.395% | 2 |
| composed_acceptance_property_decision | 1 | 0.395% | 0 |
| tora_rhs | 0 | 0.000% | 26 |
| composed_exact_endpoint_substitution | 0 | 0.000% | 2 |
| endpoint_projection | 0 | 0.000% | 2 |

## Largest host-sync callsites

| stage | source | host-sync estimate | share |
|---|---|---:|---:|
| sin_tm | `<stack-unavailable>:0:unknown` | 26 | 10.277% |
| remainder_picard_round_01 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_02 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_03 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_04 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_05 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_06 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_07 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_08 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_09 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| remainder_picard_round_10 | `<stack-unavailable>:0:unknown` | 20 | 7.905% |
| initial_remainder_picard | `<stack-unavailable>:0:unknown` | 18 | 7.115% |
| affine_composition | `<stack-unavailable>:0:unknown` | 4 | 1.581% |
| full_q3_step | `<stack-unavailable>:0:unknown` | 2 | 0.791% |
| final_acceptance_property_decision | `<stack-unavailable>:0:unknown` | 1 | 0.395% |
| composed_acceptance_property_decision | `<stack-unavailable>:0:unknown` | 1 | 0.395% |
| unscoped | `<stack-unavailable>:0:unknown` | 1 | 0.395% |

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
