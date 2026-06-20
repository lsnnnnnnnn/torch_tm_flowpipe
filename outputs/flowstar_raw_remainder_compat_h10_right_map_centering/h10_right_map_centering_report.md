# h10 Right-Map Range-Midpoint Centering Audit

This h10 audit keeps `right_map_center_mode="constant"` as the default. h10 was run only by this opt-in experiment.

## Decision

- Decision: `reject_due_to_soundness_or_reconstruction_failure`.
- Reasons: `constant_adaptive_h10 raw target violation; range_midpoint_adaptive_h10 raw target violation; constant_on_range_midpoint_schedule raw target violation`.
- Minimum target margin: `2.3488222932062895e-08` at step `53`, t `1.1910872506835912`, h `0.061034072337060589`.
- Immediate same-state saving max: `0.0022487356326686362`.
- Cumulative downstream saving max: `0.22509633248238251`.

## Run Summary

| mode | reached_t | reached_h10 | accepted | rejected | final width | tube width | Flowstar final ratio | Flowstar tube ratio | samples | min margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generated_flowstar_h10_reference | 10 | true | 290 |  | 0.7047129999999997 | 9.5663119999999999 | 1 | 1 |  |  |
| constant_adaptive_h10 | 6.3172908799330765 | false | 308 | 48 | 4.18646949336906 | 10.823889674070674 | 5.940673002156994 | 1.1314589858736235 | 0 | 9.0837879836220994e-07 |
| range_midpoint_adaptive_h10 | 6.3909311096815973 | false | 293 | 46 | 4.2364622601250339 | 10.951974122131782 | 6.0116136074189575 | 1.1448481005147837 | 0 | 2.3488222932062895e-08 |
| range_midpoint_on_constant_schedule | 6.3172908799330765 | false | 308 | 0 | 3.2481433699106885 | 10.449734049790425 | 4.6091719180867816 | 1.0923471918739871 | 0 | 4.0866916207331111e-06 |
| constant_on_range_midpoint_schedule | 0.43677646366378969 | false | 29 | 1 | 0.82403352058588952 | 3.0256663349906692 | 1.1693178933635251 | 0.31628346796452689 | 0 | 3.5043588820237207e-06 |

## Margin Watch

| rank | mode | step | t_hi | h | margin | width | had rejection |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | range_midpoint_adaptive_h10 | 53 | 1.1910872506835912 | 0.061034072337060589 | 2.3488222932062895e-08 | 0.24564733518684356 | false |
| 2 | range_midpoint_adaptive_h10 | 135 | 4.0491397437914918 | 0.036932829666854308 | 1.0405469784378614e-07 | 0.46302300766076659 | false |
| 3 | range_midpoint_adaptive_h10 | 63 | 1.7260865099637273 | 0.079153332466724513 | 7.3679614692306059e-07 | 0.2270917978664837 | false |
| 4 | range_midpoint_adaptive_h10 | 239 | 6.1482698989146005 | 0.011370299618243158 | 7.6009076952537262e-07 | 2.242032790295911 | false |
| 5 | constant_adaptive_h10 | 203 | 5.4771453511216812 | 0.023540474750928464 | 9.0837879836220994e-07 | 0.89113122033856706 | false |
| 6 | range_midpoint_adaptive_h10 | 126 | 3.8151729935722596 | 0.031326250202939117 | 1.1069335502497581e-06 | 0.65273756762651436 | false |
| 7 | range_midpoint_adaptive_h10 | 29 | 0.46156254643209432 | 0.024786082768304608 | 1.3061032199937136e-06 | 0.79804990285582145 | false |
| 8 | range_midpoint_adaptive_h10 | 280 | 6.3555406561555614 | 0.0044224408276537681 | 1.6213185542658686e-06 | 3.764678646270248 | false |
| 9 | range_midpoint_adaptive_h10 | 268 | 6.3104346995709957 | 0.0056365037487639451 | 1.7899743070947397e-06 | 3.2821380049134161 | false |
| 10 | constant_adaptive_h10 | 123 | 3.6251102820809904 | 0.023535875434214207 | 2.0019740902233417e-06 | 0.84260669879163907 | false |

## Formatting

| path | physical lines | csv.reader rows | status |
| --- | --- | --- | --- |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering/h10_right_map_centering_attempts.csv | 1034 | 1034 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering/h10_right_map_centering_checkpoints.csv | 41 | 41 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering/h10_right_map_centering_cross_schedule.csv | 604 | 604 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering/h10_right_map_centering_margin_watch.csv | 11 | 11 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering/h10_right_map_centering_segments.csv | 1232 | 1232 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering/h10_right_map_centering_summary.csv | 6 | 6 | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering/h10_right_map_centering_report.md | 49 |  | ok |
| outputs/flowstar_raw_remainder_compat_h10_right_map_centering/h10_right_map_centering_decision.txt | 1 |  | ok |
