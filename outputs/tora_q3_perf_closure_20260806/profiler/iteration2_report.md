# TORA-Q3 GPU bottleneck report

This report describes profiler event counts, not formal runtime. The raw Chrome trace remains private.

## Baseline totals

- `aten::item`: 269
- `aten::_local_scalar_dense`: 269
- paired host-scalar synchronization estimate: 269
- `aten::to`: 9500

## Largest mathematical stages

| stage | host-sync estimate | share | aten::to |
|---|---:|---:|---:|
| sin_tm | 26 | 9.665% | 7670 |
| remainder_picard_round_01 | 20 | 7.435% | 54 |
| remainder_picard_round_02 | 20 | 7.435% | 54 |
| remainder_picard_round_03 | 20 | 7.435% | 54 |
| remainder_picard_round_04 | 20 | 7.435% | 54 |
| remainder_picard_round_05 | 20 | 7.435% | 54 |
| remainder_picard_round_06 | 20 | 7.435% | 54 |
| remainder_picard_round_07 | 20 | 7.435% | 54 |
| remainder_picard_round_08 | 20 | 7.435% | 54 |
| remainder_picard_round_09 | 20 | 7.435% | 54 |
| remainder_picard_round_10 | 20 | 7.435% | 54 |
| initial_remainder_picard | 18 | 6.691% | 54 |
| affine_composition | 5 | 1.859% | 669 |
| final_acceptance_property_decision | 4 | 1.487% | 43 |
| unscoped | 4 | 1.487% | 43 |
| composed_exact_endpoint_substitution | 3 | 1.115% | 78 |
| endpoint_projection | 3 | 1.115% | 78 |
| exact_endpoint_substitution | 3 | 1.115% | 78 |
| full_q3_step | 2 | 0.743% | 3 |
| composed_acceptance_property_decision | 1 | 0.372% | 0 |

## Largest host-sync callsites

| stage | source | host-sync estimate | share |
|---|---|---:|---:|
| sin_tm | `<stack-unavailable>:0:unknown` | 26 | 9.665% |
| remainder_picard_round_01 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_02 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_03 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_04 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_05 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_06 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_07 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_08 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_09 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| remainder_picard_round_10 | `<stack-unavailable>:0:unknown` | 20 | 7.435% |
| initial_remainder_picard | `<stack-unavailable>:0:unknown` | 18 | 6.691% |
| affine_composition | `<stack-unavailable>:0:unknown` | 5 | 1.859% |
| final_acceptance_property_decision | `<stack-unavailable>:0:unknown` | 4 | 1.487% |
| unscoped | `<stack-unavailable>:0:unknown` | 4 | 1.487% |
| exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 3 | 1.115% |
| composed_exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 3 | 1.115% |
| endpoint_projection | `<stack-unavailable>:0:unknown` | 3 | 1.115% |
| full_q3_step | `<stack-unavailable>:0:unknown` | 2 | 0.743% |
| composed_acceptance_property_decision | `<stack-unavailable>:0:unknown` | 1 | 0.372% |

## Largest device/dtype conversion callsites

| stage | source | aten::to | share |
|---|---|---:|---:|
| sin_tm | `<stack-unavailable>:0:unknown` | 6552 | 68.968% |
| sin_tm | `<stack-unavailable>:0:unknown` | 780 | 8.211% |
| affine_composition | `<stack-unavailable>:0:unknown` | 576 | 6.063% |
| sin_tm | `<stack-unavailable>:0:unknown` | 130 | 1.368% |
| sin_tm | `<stack-unavailable>:0:unknown` | 78 | 0.821% |
| sin_tm | `<stack-unavailable>:0:unknown` | 78 | 0.821% |
| exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 60 | 0.632% |
| composed_exact_endpoint_substitution | `<stack-unavailable>:0:unknown` | 60 | 0.632% |
| endpoint_projection | `<stack-unavailable>:0:unknown` | 60 | 0.632% |
| tora_rhs | `<stack-unavailable>:0:unknown` | 52 | 0.547% |
| polynomial_picard_k1 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| polynomial_picard_k2 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| initial_remainder_picard | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| remainder_picard_round_01 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| remainder_picard_round_02 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| remainder_picard_round_03 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| remainder_picard_round_04 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| remainder_picard_round_05 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| remainder_picard_round_06 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |
| remainder_picard_round_07 | `<stack-unavailable>:0:unknown` | 48 | 0.505% |

The paired synchronization estimate is `max(aten::item, aten::_local_scalar_dense)` so one scalar extraction is not counted twice. Stage and callsite CSV files retain the complete sanitized aggregation.
