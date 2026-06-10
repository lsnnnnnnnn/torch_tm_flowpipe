# Flowstar Raw Remainder Compat h5 Audit

This h5-only audit does not run h10, does not add NNCS/GPU work, does not add symbolic queue variants, does not change default solver behavior, and does not claim Flowstar parity.

## Scope

- Requested horizon: `5`.
- Compat remains opt-in through `validation_mode="flowstar_raw_remainder_compat"` and `step_policy_mode="flowstar_compat"`.
- Flowstar GNUPLOT rectangles are treated as segment boxes, not endpoint boxes.
- Endpoint-only legacy artifacts are not used for Flowstar width ratios.

## Answers

- Did raw compat + Flowstar step policy reach h5? `yes`; reached_t `5`.
- Did it remain sample-contained? `yes`; violations `0`.
- Did it use any non-final h below 0.002? `no`; count `0`.
- Is it width-close to Flowstar h5 segment/tube boxes? last-segment ratio `2.6079115564373536`, tube ratio `1.0110368009992328`.
- Compared with previous normalized-insertion h5, it is `wider last-segment width than previous normalized-insertion h5 endpoint artifact`; previous runtime_s `344.34351133462042`, current runtime_s `125.44745034910738`.
- Did Flowstar step policy improve schedule match over raw compat default? `yes`; default `2.4104983230011037`, Flowstar step `1.9229736906179671`.
- Does h5 justify h10 next? `no`; recommendation `review_width_before_h10`.
- Did compat become too conservative? `no`.

## Summary

| mode | status | reached_t | completed_h5 | accepted | rejected | min_h | below_0.002 | final_width_sum | last_ratio | tube_ratio | schedule_distance | samples | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_flowstar_h5_reference | completed | 5 | true | 149 |  | 0.0098075999999999997 | 0 | 0.18597819999999987 | 1 | 1 | 0 | not_applicable | reference_only |
| flowstar_style_o6_candidate8_output6_cutoff_insert | max_horizon_reached | 5 | true | 56 | 7 | 0.025000000000001243 | 0 | 0.19394259329355046 |  |  | 7.5254740000000009 | not_run_existing_artifact | existing_normalized_insertion_h5_baseline |
| current_no_queue_default_policy | completed | 5 | true | 135 | 75 | 0.012670540809631349 | 0 | 0.46282913690268868 | 2.4886203700363216 | 1.0129858149627102 | 2.8156815097998269 | passed | baseline_only |
| raw_remainder_compat_default_policy | completed | 5 | true | 180 | 105 | 0.01001129150390625 | 0 | 0.52870916239798293 | 2.8428555733843175 | 1.0109866551717765 | 2.4104983230011037 | passed | compare_against_flowstar_step_policy |
| raw_remainder_compat_flowstar_step_policy | completed | 5 | true | 181 | 27 | 0.0083398093842310672 | 0 | 0.48501469702541711 | 2.6079115564373536 | 1.0110368009992328 | 1.9229736906179671 | passed | review_width_before_h10 |

## Width Semantics

| mode | enabled | semantics | disabled_reason |
| --- | --- | --- | --- |
| generated_flowstar_h5_reference | true | reference segment/tube |  |
| flowstar_style_o6_candidate8_output6_cutoff_insert | false | disabled endpoint-vs-segment comparison | endpoint-only or legacy width artifact; Flowstar GNUPLOT segment ratio disabled |
| current_no_queue_default_policy | true | torch segment TM boxes vs Flowstar GNUPLOT segment boxes |  |
| raw_remainder_compat_default_policy | true | torch segment TM boxes vs Flowstar GNUPLOT segment boxes |  |
| raw_remainder_compat_flowstar_step_policy | true | torch segment TM boxes vs Flowstar GNUPLOT segment boxes |  |

## Outputs

- `outputs/flowstar_raw_remainder_compat_h5/h5_summary.csv`
- `outputs/flowstar_raw_remainder_compat_h5/h5_segments.csv`
- `outputs/flowstar_raw_remainder_compat_h5/h5_width_vs_flowstar.csv`
- `outputs/flowstar_raw_remainder_compat_h5/h5_schedule_compare.csv`
- `outputs/flowstar_raw_remainder_compat_h5/h5_sample_containment.csv`
- `outputs/flowstar_raw_remainder_compat_h5/h5_report.md`
