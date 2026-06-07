# Flowstar Linear Symbolic Queue V2 H10 Report

Did v2 beat split queue t~=7.4960392581387341? no; best v2 t=`7.496039258138734`.
Did v2 beat no-queue o4 t~=6.4730088058091901? no; best order4 t=`6.4730088058091901`.
Did v2 beat no-queue o6 t~=7.4960392581387341? no; best order6 t=`7.4960392581387341`.
Did any config reach h10? no.
Did o4 reach h10? no.
Did o6 reach h10? no.
How large are J/Phi_L/scalars? max J=`99.0`, max Phi_L=`99.0`, max |scalar|=`44.27126943263349`, max current L norm=`0.35552562779993413`.
Did v2 reduce reset width or only add output width? target-clean reset; propagated queue contribution is output-only in v2; max reset=`21.903636698476824`, max output-only symbolic=`0.24507727045990219`.
Did sample containment pass? passed.
Is v2 conservative? yes; output range includes symbolic contribution for validated rows=yes.
Flow* width ratios for best config: last=`101.53378043476893`, tube=`2.5046094332157787`.
Failure reason if still failed: `Picard residual not subset of target remainder`.
This is experimental clean-room queue propagation, not Flow* parity.

## Best Config Metrics

| run_id | status | last_validated_t | runtime_s | max_j | max_phi_l | max_target_check | max_output_only_symbolic | last_width_ratio | tube_width_ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_candidate8_output6_insert_symqueue_v2 | failed | 7.4960392581387341 | 1181.2034425213933 | 99.0 | 99.0 | 1e-323 | 0.24507717887847935 | 101.53378043476893 | 2.5046094332157787 |

## Config Status

| run_id | status | last_validated_t | accepted | rejected | max_queue | max_output_only_symbolic | max_reset_width | conservative | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| flowstar_style_o4_target_insert_symqueue_v2 | failed | 6.4730088058091901 | 239 | 140 | 99 | 0.17422689579314798 | 4.820920894732832 | yes | Picard residual not subset of target remainder |
| flowstar_style_o4_target_cutoff_insert_symqueue_v2 | failed | 6.4730088058091901 | 239 | 140 | 99 | 0.17422690132680085 | 4.820924466161561 | yes | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_insert_symqueue_v2 | failed | 7.4960392581387341 | 150 | 60 | 99 | 0.24507717887847935 | 21.902999702465415 | yes | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue_v2 | failed | 7.4960392581387341 | 150 | 60 | 99 | 0.24507727045990219 | 21.903636698476824 | yes | Picard residual not subset of target remainder |

## Sample Containment

| run_id | samples | checked_pairs | violations | max_outside_distance | status |
| --- | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_insert_symqueue_v2 | 500 | 75000 | 0 | 0 | passed |
