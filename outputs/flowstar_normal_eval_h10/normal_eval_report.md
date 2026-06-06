# Normal Eval H10 Report

Baseline o4 no-normal-eval: t=`6.4730088058091901`, final width=`4.8223900700103997`.
Baseline o6 no-normal-eval: t=`7.4960392581387341`, final width=`21.899038480793845`.
Best normal_eval config: `flowstar_style_o6_candidate8_output6_insert_normaleval` at t=`7.496039258138734`.
Did normal_eval beat the o4 baseline t~=6.4730088058091901 or o6 baseline t~=7.4960392581387341? no.
Did any config reach h10? no.
Did width ratios improve? last=`101.26404571229271`, tube=`2.4989346486924005`.
Did right_map_scaling shrink? no; old max=`21.88451275984548`, normal max=`21.88451275984548`, inserted max=`21.88451275984548`.
Did sample containment pass? passed.
Did normal_eval remain conservative in tests? See pytest result for `evaluate_interval_normal` sample containment tests.
Branch decision: NEEDS_MORE_WORK.

## Config Status

| run_id | status | last_validated_t | right_map_range_mode | final_width_sum | last_width_ratio | tube_width_ratio | failure_reason |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| flowstar_style_o4_target_insert_normaleval | failed | 6.4730088058091901 | normal_eval | 4.8223900700104023 | 14.049230879967748 | 1.1807087361819038 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_insert_normaleval | failed | 7.4960392581387341 | normal_eval | 21.899038480794122 | 101.26404571229271 | 2.4989346486924005 | Picard residual not subset of target remainder |
| flowstar_style_o4_target_cutoff_insert_normaleval | failed | 6.4730088058091901 | normal_eval | 4.8223936380975623 | 14.049241274995854 | 1.1807088850728908 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_insert_normaleval | failed | 7.4960392581387341 | normal_eval | 21.899675333588121 | 101.26699060370883 | 2.4989984500401441 | Picard residual not subset of target remainder |

## Sample Containment

| run_id | samples | checked_pairs | violations | max_outside_distance | status |
| --- | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_insert_normaleval | 500 | 75000 | 0 | 0 | passed |
