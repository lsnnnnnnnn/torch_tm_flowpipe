# Flowstar Raw Remainder Compat h5 Width Attribution

This is an attribution audit only. It does not run h10, work on NNCS/GPU, add symbolic queue variants, change default solver behavior, or claim Flowstar parity.

## Formatting Checks

The requested physical line and csv.reader row-count checks passed locally; CSV physical line counts match csv.reader row counts for all checked CSVs.

| path | physical lines | csv.reader rows | status |
| --- | --- | --- | --- |
| outputs/flowstar_raw_remainder_compat_h5/h5_report.md | 50 |  | ok |
| outputs/flowstar_raw_remainder_compat_h5/h5_sample_containment.csv | 6 | 6 | ok |
| outputs/flowstar_raw_remainder_compat_h5/h5_schedule_compare.csv | 6 | 6 | ok |
| outputs/flowstar_raw_remainder_compat_h5/h5_segments.csv | 702 | 702 | ok |
| outputs/flowstar_raw_remainder_compat_h5/h5_summary.csv | 6 | 6 | ok |
| outputs/flowstar_raw_remainder_compat_h5/h5_width_vs_flowstar.csv | 6 | 6 | ok |
| outputs/flowstar_raw_remainder_compat_h5_divergence/h5_divergence_ledger.csv | 4 | 4 | ok |
| outputs/flowstar_raw_remainder_compat_h5_divergence/h5_divergence_report.md | 37 |  | ok |
| outputs/flowstar_raw_remainder_compat_h5_divergence/h5_schedule_divergence.csv | 182 | 182 | ok |
| outputs/flowstar_raw_remainder_compat_h5_divergence/h5_width_growth.csv | 182 | 182 | ok |

## Answers

- Component first correlated with width ratio >1.1: `full_step_tube_relative_ratio` at t `0.90651094330482829`.
- Component driving >1.5: `polynomial_range` at t `3.0235741944970576`.
- Component driving >2.0: `right_map_range` at t `3.6851867809205943`.
- Raw residual or right_map dominates at the first crossing: `neither_growth`.
- Flowstar component fields: `unknown_missing_h5_reference_component_fields`; missing component fields are reported as unknown, not zero.
- Compat residual_y_hi stays below target in audited windows: `yes`; target margin range `8.0048412987621006e-06` to `9.064614039757807e-05`.
- Does right_map_range begin to diverge before raw residual? `no at the first crossing`; neither raw residual nor right_map grows there, and right_map becomes the dominant raw-vs-right-map signal by the 2.0 crossing while raw residual remains target-bounded.
- Is final 2.6x last-segment width early accumulation or late local blowup? `gradual accumulation`; the ratio crosses 1.1, 1.5, and 2.0 progressively before the final segment.
- Does tube ratio stay close because earlier extrema dominate? `yes`; last ratio `2.6079115564373536`, tube ratio `1.0110368009992328`.
- Next mechanism before h10: inspect normalized-insertion right-map/reset scaling and full-step tube range source attribution before changing raw remainder mechanics.
- Does this justify h10 now? `no`; recommendation `review_width_before_h10`.

## Event Windows

| event | source | step | t | h | width ratio | tube prefix ratio | raw residual y_hi | right_map width | reset width | target margin | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_schedule_divergence | flowstar | 5 | 0.096445130000000004 | 0.020131380000000004 | 1 | 1 |  |  |  |  | Flowstar h5 reference exposes segment boxes only; component attribution fields are unknown, not zero |
| first_schedule_divergence | compat | 5 | 0.09644512500000002 | 0.020131375000000007 | 1.0503889157948785 | 1.0193049152630218 | 9.1995157701237895e-05 | 0.59100732857041216 | 0.59379572892644672 | 8.0048412987621006e-06 | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |
| width_ratio_gt_1_1 | flowstar | 48 | 0.92419030000000002 | 0.037897300000000023 | 1 | 1 |  |  |  |  | Flowstar h5 reference exposes segment boxes only; component attribution fields are unknown, not zero |
| width_ratio_gt_1_1 | compat | 48 | 0.90651094330482829 | 0.037897356947215827 | 1.1035025400929366 | 1.0014635639699925 | 6.7024187976355714e-05 | 0.22791902612175929 | 0.2417025584055284 | 3.2975811023644281e-05 | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |
| width_ratio_gt_1_5 | flowstar | 92 | 3.078875 | 0.019619000000000053 | 1 | 1 |  |  |  |  | Flowstar h5 reference exposes segment boxes only; component attribution fields are unknown, not zero |
| width_ratio_gt_1_5 | compat | 92 | 3.0235741944970576 | 0.039238020998147327 | 1.5513085996398646 | 0.98201974793950786 | 6.7236867425506496e-05 | 0.29025843208347601 | 0.34511977063874788 | 3.2763131574493499e-05 | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |
| width_ratio_gt_2_0 | flowstar | 127 | 3.8966409999999998 | 0.034459000000000017 | 1 | 1 |  |  |  |  | Flowstar h5 reference exposes segment boxes only; component attribution fields are unknown, not zero |
| width_ratio_gt_2_0 | compat | 127 | 3.6851867809205943 | 0.017229437611616515 | 2.1358225056343265 | 1.0004825055699327 | 4.1137389810808309e-05 | 0.53423976786364791 | 0.60421449265912097 | 5.8862609189191686e-05 | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |
| final_segment_near_t5 | flowstar | 148 | 5.0335789999999996 | 0.063750999999999891 | 1 | 1 |  |  |  |  | Flowstar h5 reference exposes segment boxes only; component attribution fields are unknown, not zero |
| final_segment_near_t5 | compat | 180 | 5 | 0.0083398093842310672 | 2.6079115564373536 | 1.0110368009992328 | 9.3538586024219299e-06 | 0.45279941074348473 | 0.47034070823292951 | 9.064614039757807e-05 | compat mode checks replayed raw remainder plus poly_diff_range; target remainder is only the containment set |

## Outputs

- `outputs/flowstar_raw_remainder_compat_h5_width_attribution/h5_width_attribution_ledger.csv`
- `outputs/flowstar_raw_remainder_compat_h5_width_attribution/h5_component_growth.csv`
- `outputs/flowstar_raw_remainder_compat_h5_width_attribution/h5_crossing_windows.csv`
- `outputs/flowstar_raw_remainder_compat_h5_width_attribution/h5_width_attribution_report.md`
