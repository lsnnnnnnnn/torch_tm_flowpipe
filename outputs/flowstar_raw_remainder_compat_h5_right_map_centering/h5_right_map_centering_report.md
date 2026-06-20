# Flowstar Raw Remainder Compat h5 Right-Map Centering Audit

This is an h5-only opt-in mechanism audit. No h10 was run. The default solver behavior remains `right_map_center_mode="constant"`, and sample containment is reported only as a sanity check.

## Decision

- Decision: `promote_range_midpoint_to_h10_candidate`.
- Minimum target margin across range-midpoint h5 runs: `2.3488222932062895e-08`.
- Frozen final segment width improvement: `0.13290459146677422`.
- Frozen tube width change: `-0.0021337137100844172`.
- Reset width reduction at >1.5: `0.16744982971241973`.
- Reset width reduction at >2.0: `0.13749927577701038`.
- Reconstruction pass: `true`; max polynomial diff `4.4408920985006262e-16`, max remainder endpoint diff `0`.
- Raw residual target violations: `0`.
- Sample-containment sanity violations: `0`.

## Reasons

- Promote criteria and rejection criteria evaluated from frozen/adaptive h5 rows.

## Key Comparisons

- Baseline reproduction from existing h5 artifact: `true`; existing_status=completed; reproduced_status=completed; existing_steps=181; reproduced_steps=181; existing_final_width_sum=0.48501469702541711; reproduced_final_width_sum=0.4850146970254171; h_sequence_match=True.
- Frozen schedule complete: `true`; first failure step ``, reason ``.
- Adaptive range_midpoint h5 complete: `true`; accepted steps `170`, rejected attempts `25`.
- Baseline frozen final width sum `0.48501469702541711` vs range_midpoint frozen `0.42055401686187277`.
- Baseline frozen tube width sum `9.3638214692448987` vs range_midpoint frozen `9.3438417549971877`.
- Flowstar last-width ratio: baseline frozen `2.6079115564373536`, range_midpoint frozen `2.261308136447568`, range_midpoint adaptive `2.4175608568320763`.
- Flowstar tube-width ratio: baseline frozen `1.0110368009992328`, range_midpoint frozen `1.0088795379155409`, range_midpoint adaptive `1.009477893601594`.

## Required Window Answers

1. Does range centering affect the first >1.1 crossing? Baseline frozen reset width `0.22171701388694534` vs range_midpoint frozen `0.20802138169837744` at checkpoint `0.92419030000000002`.
2. Does it lower polynomial-range-dominated >1.5 accumulation? Polynomial range width baseline `0.4594273808550805` vs range_midpoint `0.39892573582124763`; reset reduction `0.16744982971241973`.
3. Does it lower right-map-dominated >2.0 accumulation? Right-map range width baseline `0.40834025347847613` vs range_midpoint `0.38244562430793272`; reset reduction `0.13749927577701038`.
4. Improvement propagation: final frozen width improvement `0.13290459146677422` and tube change `-0.0021337137100844172` distinguish reset-scale-only improvement from downstream width propagation.
5. Frozen-schedule improvement exists: `true`.
6. Adaptive schedule-only explanation: schedule effects are separated because frozen replay uses identical prescribed h values; adaptive range has its own row and is not used alone for the decision.

## Summary

| mode | status | reached_h5 | accepted | rejected | final_width_sum | tube_width_sum | Flowstar last ratio | Flowstar tube ratio | min target margin | raw target violations | samples | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_flowstar_h5_reference | completed | true | 149 |  | 0.18597819999999987 | 9.2616030000000009 | 1 | 1 |  |  | not_applicable | reference_only |
| baseline_adaptive_constant | completed | true | 181 | 27 | 0.48501469702541711 | 9.3638214692448987 | 2.6079115564373536 | 1.0110368009992328 | 2.0019740902233417e-06 | 0 | passed | baseline_reproduction |
| baseline_frozen_constant | completed | true | 181 | 0 | 0.48501469702541711 | 9.3638214692448987 | 2.6079115564373536 | 1.0110368009992328 | 2.0019740902233417e-06 | 0 | passed | baseline_comparator |
| range_midpoint_frozen | completed | true | 181 | 0 | 0.42055401686187277 | 9.3438417549971877 | 2.261308136447568 | 1.0088795379155409 | 4.0866916207331111e-06 | 0 | passed | promote_range_midpoint_to_h10_candidate |
| range_midpoint_adaptive | completed | true | 170 | 25 | 0.44961361654408694 | 9.3493834878142046 | 2.4175608568320763 | 1.009477893601594 | 2.3488222932062895e-08 | 0 | passed | promote_range_midpoint_to_h10_candidate |

## Time Alignment Semantics

- Physical-time-aligned rows use the segment that contains the checkpoint time; no box-bound interpolation is performed.
- Frozen-schedule rows are strict step-paired comparisons using the baseline accepted h sequence with `h_min=h_max=prescribed_h`.
- Flowstar component fields remain `unknown_missing_h5_reference_component_fields`, not zero.

## Formatting Checks

| path | physical lines | csv.reader rows | status |
| --- | --- | --- | --- |
| outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_crossings.csv | 41 | 41 | ok |
| outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_frozen_schedule.csv | 182 | 182 | ok |
| outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_segments.csv | 863 | 863 | ok |
| outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_summary.csv | 6 | 6 | ok |
| outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_decision.txt | 1 |  | ok |

## Outputs

- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_summary.csv`
- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_segments.csv`
- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_frozen_schedule.csv`
- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_crossings.csv`
- `outputs/flowstar_raw_remainder_compat_h5_right_map_centering/h5_right_map_centering_decision.txt`
