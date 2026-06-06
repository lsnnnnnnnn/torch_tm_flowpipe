# Normalized Insertion Plus Symbolic Queue H10 Report

Did any symqueue config reach horizon 10? no.
Did order4 reach horizon 10? no.
Did order6/candidate8 reach horizon 10? no.
Did symqueue improve last_validated_t over o4_insert baseline t~=6.4730088058091901? no; best order4 t=`3.217705703618956`.
Did symqueue improve last_validated_t over o6_insert baseline t~=7.4960392581387341? no; best order6 t=`3.3500000000000014`.
Did symqueue reduce width ratios? yes.
Did queue remain stable or materialize too much width? stable; max propagated=`0.0004888079255194722`, max materialized=`0.0004888079255194722`.
Runtime cost for best config: `285.78735389374197` seconds.
Did sample containment still pass? passed.
Which config is best? `flowstar_style_o6_candidate8_output6_insert_symqueue`.
Queue implementation note: this is a conservative limited queue. It propagates older interval columns through the inserted endpoint linear part and materializes propagated width on the next normalized reset; current insertion uncertainty is queued for future propagation.
Branch status remains NEEDS_MORE_WORK unless h10 is reached and Flow* comparison/sample checks are acceptable.

## Best Config Metrics

| run_id | status | last_validated_t | runtime_s | queue_size | propagated_width | materialized_width | last_width_ratio | tube_width_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_candidate8_output6_insert_symqueue | failed | 3.3500000000000014 | 285.78735389374197 | 38 | 0.00048880602852173364 | 0.00048880602852173364 | 0.70399519325185822 | 0.99518834361763953 |

## Config Status

| run_id | status | last_validated_t | accepted | rejected | max_queue | max_propagated | max_new_symbolic | max_materialized | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o4_target_insert_symqueue | failed | 3.217705703618956 | 77 | 45 | 77 | 0.00031163711170903571 | 0.10039724438641102 | 0.00031163711170903576 | initial or cutoff remainder exceeds target remainder |
| flowstar_style_o4_target_cutoff_insert_symqueue | failed | 3.217705703618956 | 77 | 45 | 77 | 0.00031163737157711714 | 0.1003973017526974 | 0.00031163737157711719 | initial or cutoff remainder exceeds target remainder |
| flowstar_style_o6_candidate8_output6_insert_symqueue | failed | 3.3500000000000014 | 38 | 12 | 38 | 0.00048880602852173364 | 0.053489211638176284 | 0.00048880602852173364 | initial or cutoff remainder exceeds target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue | failed | 3.3500000000000014 | 38 | 12 | 38 | 0.00048880792551947222 | 0.05348939675662296 | 0.00048880792551947222 | initial or cutoff remainder exceeds target remainder |

## Sample Containment

| run_id | samples | checked_pairs | violations | max_outside_distance | status |
| --- | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_insert_symqueue | 500 | 19000 | 0 | 0 | passed |
