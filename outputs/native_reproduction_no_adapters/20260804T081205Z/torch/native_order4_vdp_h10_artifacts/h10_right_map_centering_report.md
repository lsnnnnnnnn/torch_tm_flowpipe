# h10 Right-Map Range-Midpoint Centering Audit

This h10 audit keeps `right_map_center_mode="constant"` as the default. h10 was run only by this opt-in experiment.
Terminal validation rejection is a safe failure-to-progress, not an accepted unsound enclosure.

## Decision

- Decision: `h10_not_reached_no_material_improvement`.
- Reasons: `criteria evaluated from h10 artifacts`.
- Accepted soundness checks passed.
- Centering produced material common-prefix width improvement, but validated-horizon extension did not meet the predefined 0.5 threshold.
- Minimum target margin: `2.348822293214421e-08` at step `53`, t `1.1910872506835912`, h `0.061034072337060589`.
- Accepted raw-target violations: `0`.
- Rejected raw-target attempts: `35`.
- Terminal raw-target rejection: `false`.
- Baseline counterfactual immediate saving max: `0.17003372367652919`.
- Post-centering remaining asymmetry max: `0.0022487356326687294`.
- Frozen cumulative final width saving: `0.17108069510880494`.
- Common-time width worsening count: `26`.

## Run Summary

| mode | replay status | reached_t | reached_h10 | accepted | rejected | accepted raw | rejected raw attempts | terminal target rejection | final width | time-aligned Flowstar width | time-aligned ratio | time-aligned tube ratio | samples | min margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_flowstar_h10_reference |  | 10 | true | 290 |  |  |  |  | 0.7047129999999997 | 0.7047129999999997 | 1 | 1 |  |  |
| constant_adaptive_h10 |  | 5.904686584519073 | false | 236 | 36 | 0 | 36 | false | 1.7385948451599837 | 0.38628250000000014 | 4.5008377163345044 | 1.0110368009992325 | 0 | 9.0837879836650609e-07 |
| range_midpoint_adaptive_h10 |  | 6.0490383280541096 | false | 230 | 35 | 0 | 35 | false | 1.867462569555199 | 0.52398506 | 3.5639614792742353 | 1.0136152029934893 | 0 | 2.348822293214421e-08 |
| range_midpoint_on_constant_schedule | completed_source_schedule | 5.904686584519073 | false | 236 | 0 | 0 | 0 | false | 1.4411548305374287 | 0.38628250000000014 | 3.7308312712520713 | 1.0088795379155402 | 0 | 4.0866916207333822e-06 |
| constant_on_range_midpoint_schedule | validation_failed | 0.43677646366378969 | false | 29 | 1 | 0 | 1 | true | 0.82403352058588819 | 0.77074450000000017 | 1.0691396702615303 | 1.0113150202927621 | 0 | 3.5043588820240189e-06 |

## Margin Watch

| rank | mode | step | t_hi | h | margin | width | had rejection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | range_midpoint_adaptive_h10 | 53 | 1.1910872506835912 | 0.061034072337060589 | 2.348822293214421e-08 | 0.24564733518684384 | false |
| 2 | range_midpoint_adaptive_h10 | 135 | 4.0491397437914918 | 0.036932829666854308 | 1.0405469784489745e-07 | 0.46302300766076504 | false |
| 3 | range_midpoint_adaptive_h10 | 63 | 1.7260865099637273 | 0.079153332466724513 | 7.3679614692312835e-07 | 0.22709179786648426 | false |
| 4 | constant_adaptive_h10 | 203 | 5.4771453511216812 | 0.023540474750928464 | 9.0837879836650609e-07 | 0.89113122033855374 | false |
| 5 | range_midpoint_adaptive_h10 | 126 | 3.8151729935722596 | 0.031326250202939117 | 1.1069335502507068e-06 | 0.65273756762651025 | false |
| 6 | range_midpoint_adaptive_h10 | 29 | 0.46156254643209432 | 0.024786082768304608 | 1.3061032199940389e-06 | 0.79804990285581878 | false |
| 7 | constant_adaptive_h10 | 123 | 3.6251102820809904 | 0.023535875434214207 | 2.0019740902242633e-06 | 0.84260669879163641 | false |
| 8 | range_midpoint_adaptive_h10 | 40 | 0.71418581874543208 | 0.035358793412637569 | 2.0611459770947439e-06 | 0.4989055541572805 | false |
| 9 | constant_adaptive_h10 | 77 | 2.3776437705154296 | 0.075146162079494025 | 2.8533243139041499e-06 | 0.41412888728250552 | false |
| 10 | range_midpoint_adaptive_h10 | 97 | 3.2352233178827374 | 0.031596612598863136 | 2.9880716870832774e-06 | 0.47131409244269074 | false |

## Formatting

| path | physical lines | csv.reader rows | status |
| --- | --- | --- | --- |
| outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_attempts.csv | 804 | 804 | ok |
| outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_checkpoints.csv | 36 | 36 | ok |
| outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_cross_schedule.csv | 467 | 467 | ok |
| outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_margin_watch.csv | 11 | 11 | ok |
| outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_segments.csv | 1023 | 1023 | ok |
| outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_summary.csv | 6 | 6 | ok |
| outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_report.md | 56 |  | ok |
| outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_decision.txt | 1 |  | ok |
