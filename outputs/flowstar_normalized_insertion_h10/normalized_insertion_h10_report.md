# Normalized Insertion H10 Report

Did any normalized insertion config reach horizon 10? no.
Which config is best? `flowstar_style_o6_candidate8_output6_insert`.
Did order4 reach horizon 10? no.
Did order6/candidate8 reach horizon 10? no.
Are widths comparable to Flow*? no; width ratios exceed the 10% comparison threshold.
Did any non-final step go below Flow* min step 0.002? no.
Did sample containment sanity pass? passed.
Branch decision: NEEDS_MORE_WORK.
Recommended next step: investigate any h10 failure point.

## Best Config Metrics

| run_id | status | runtime_s | segments | last_validated_t | min_regular_h_used | h_below_flowstar_min_count | final_width_sum | last_width_ratio | tube_width_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_candidate8_output6_insert | failed | 1198.2035875059664 | 150 | 7.4960392581387341 | 0.0021450425182496136 | 0 | 21.899038480793845 | 101.26404571229142 | 2.4989346486923725 |

## Config Status

| run_id | status | runtime_s | segments | last_validated_t | min_regular_h_used | h_below_flowstar_min_count | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_cutoff_insert | failed | 1130.5990600492805 | 150 | 7.4960392581387341 | 0.0021450425182496136 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_insert | failed | 1198.2035875059664 | 150 | 7.4960392581387341 | 0.0021450425182496136 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o4_target_cutoff_insert | failed | 165.84051201120019 | 239 | 6.4730088058091901 | 0.0020171156691308999 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o4_target_insert | failed | 163.33953178208321 | 239 | 6.4730088058091901 | 0.0020171156691308999 | 0 | Picard residual not subset of target remainder |

## Interpretation

Do not claim exact Flow* parity from this report; adaptive grids differ and segment boxes are not expected to match exactly.
If only order6/candidate8 reaches h10, this is a higher-order PyTorch rescue result, not original order-4 parity.
If an order4 insert config reaches h10, treat it as the closer result to original Flow* settings and highlight it separately.

## Sample Containment

| run_id | samples | checked_pairs | violations | max_outside_distance | status |
| --- | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_cutoff_insert | 500 | 75000 | 0 | 0 | passed |
