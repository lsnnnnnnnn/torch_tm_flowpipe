# Normalized Insertion Plus Symbolic Queue Split H10 Report

Did split semantics beat old symqueue t~=3.3500000000000014? yes; best split t=`7.496039258138734`.
Did split beat no-queue o4 t~=6.4730088058091901? no; best order4 t=`6.4730088058091901`.
Did split beat no-queue o6 t~=7.4960392581387341? no; best order6 t=`7.4960392581387341`.
Did any config reach horizon 10? no.
Did order4 reach horizon 10? no.
Did order6/candidate8 reach horizon 10? no.
Did range boxes remain conservative and sample containment pass? passed.
Did symbolic contribution remain bounded? no; max symbolic=`5.01787076889624`, max output materialized=`5.01787076889624`.
Did ordinary target remainder stay at 1e-4? yes.
Flow* width ratios for best config: last=`124.46569353503585`, tube=`2.9696033909707853`.
Runtime cost for best config: `1530.6576520055532` seconds.
If it still fails, exact reported failure component: `Picard residual not subset of target remainder`.
Queue split note: propagated symbolic width is added to reported output/range boxes, while ordinary target containment checks the local Picard target remainder channel.

## Best Config Metrics

| run_id | status | last_validated_t | runtime_s | queue_size | ordinary_only_range | symbolic_contribution | output_materialized | total_with_symbolic | last_width_ratio | tube_width_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_candidate8_output6_insert_symqueue_split | failed | 7.4960392581387341 | 1530.6576520055532 | 99 | 21.902999702465415 | 5.0175141128736733 | 5.0175141128736733 | 26.920513815339088 | 124.46569353503585 | 2.9696033909707853 |

## Config Status

| run_id | status | last_validated_t | accepted | rejected | max_queue | max_symbolic | max_output_materialized | target_checked_width | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o4_target_insert_symqueue_split | failed | 6.4730088058091901 | 239 | 140 | 99 | 0.30285839887550597 | 0.30285839887550597 | 9.8813129168249309e-324 | Picard residual not subset of target remainder |
| flowstar_style_o4_target_cutoff_insert_symqueue_split | failed | 6.4730088058091901 | 239 | 140 | 99 | 0.30285889450645781 | 0.30285889450645781 | 9.8813129168249309e-324 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_insert_symqueue_split | failed | 7.4960392581387341 | 150 | 60 | 99 | 5.0175141128736733 | 5.0175141128736733 | 9.8813129168249309e-324 | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue_split | failed | 7.4960392581387341 | 150 | 60 | 99 | 5.0178707688962403 | 5.0178707688962403 | 9.8813129168249309e-324 | Picard residual not subset of target remainder |

## Sample Containment

| run_id | samples | checked_pairs | violations | max_outside_distance | status |
| --- | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_insert_symqueue_split | 500 | 75000 | 0 | 0 | passed |
