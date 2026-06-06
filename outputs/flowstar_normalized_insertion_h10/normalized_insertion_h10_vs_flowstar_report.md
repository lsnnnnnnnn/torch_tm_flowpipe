# Normalized Insertion H10 Vs Original Flow* Comparison

Requested horizon: `10`.
Original Flow* boxes are GNUPLOT segment boxes; adaptive PyTorch grids are not expected to match segment counts.
This report is a width and overlap comparison, not an exact Flow* parity claim.

## Metrics

| run_id | status | runtime_s | segments | last_validated_t | min_h_used | min_regular_h_used | h_below_flowstar_min_count | max_h_used | step_rejections | final_width_sum | py_tube_width_sum | flowstar_last_width_sum | flowstar_tube_width_sum | last_width_ratio | tube_width_ratio | max_overlap_width_ratio | median_overlap_width_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_candidate8_output6_cutoff_insert | failed | 1130.5990600492805 | 150 | 7.496039258138734 | 0.0021450425182496136 | 0.0021450425182496136 | 0 | 0.1 | 60 | 21.89967533358815 | 23.822502404511717 | 0.2162567999999998 | 9.532820000000001 | 101.26699060370896 | 2.498998450040147 | 101.26699060370896 | 3.039299910852322 |
| flowstar_style_o6_candidate8_output6_insert | failed | 1198.2035875059664 | 150 | 7.496039258138734 | 0.0021450425182496136 | 0.0021450425182496136 | 0 | 0.1 | 60 | 21.899038480793845 | 23.821894197747625 | 0.2162567999999998 | 9.532820000000001 | 101.26404571229142 | 2.4989346486923725 | 101.26404571229142 | 3.0392892340204725 |
| flowstar_style_o4_target_cutoff_insert | failed | 165.8405120112002 | 239 | 6.47300880580919 | 0.0020171156691309 | 0.0020171156691309 | 0 | 0.1 | 140 | 4.822393638097562 | 11.255485273800556 | 0.34324940000000004 | 9.532820000000001 | 14.049241274995854 | 1.1807088850728908 | 14.049241274995854 | 1.3359322478410458 |
| flowstar_style_o4_target_insert | failed | 163.3395317820832 | 239 | 6.47300880580919 | 0.0020171156691309 | 0.0020171156691309 | 0 | 0.1 | 140 | 4.8223900700104 | 11.255483854449576 | 0.34324940000000004 | 9.532820000000001 | 14.049230879967741 | 1.1807087361819038 | 14.049230879967741 | 1.3359318424177438 |
