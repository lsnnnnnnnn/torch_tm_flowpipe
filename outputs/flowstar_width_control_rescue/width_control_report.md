# Flowstar Width-Control Rescue Report

Chosen mechanism: Flow*-style symbolic remainder queue skeleton (`J`, `Phi_L`, `scalars`) because the original Van der Pol benchmark calls `ode.reach(..., sr)` with a symbolic queue of size 100.
Previous best `flowstar_style_o6_candidate8_output6_cutoff` reached t=`2.4007376673997931`.
New width-control `flowstar_style_o6_candidate8_output6_cutoff_symqueue` reached t=`0.094531250000000011`.
Did the new width-control beat t~=2.400737? no.
Did it reach horizon 5? no.
Runtime cost: previous=`452.03751178923994`, new=`76.29850434884429` seconds.
Width ratio vs Flow*: previous tube=`4.835528765199405`, new tube=`0.7743007976781516`.
Did width ratio improve over the validated same-run horizon? yes (not comparable as a success if the new run stops much earlier).
Did reset box width shrink vs previous best? yes; previous max reset width sum=`26.97665552472704`, new=`0.59193589324218188`.
Queue peak size after accepted steps: `3.0`; propagated remainder peak width sum: `6.697320704630282e-06`.
Did the local one-step oracle become easier? Flow* validates the same local box/h_try after width control; see `outputs/flowstar_one_step_oracle_after_width_control/oracle_after_width_control_report.md`.
Failure mode if still failing: `Picard residual not subset of target remainder`.
Branch decision: NEEDS_MORE_WORK.
Next recommendation: The queue is tighter over its short horizon but fails much earlier; implement normalized insertion/composition next.

## Rows

| run_id | reset_mode | status | last_validated_t | runtime_s | max_queue_after | max_propagated_width | failure_reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_candidate8_output6_cutoff | normalized_endpoint_box | failed | 2.400737667399793 | 452.03751178923994 |  |  | Picard residual not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_symqueue | flowstar_symbolic_remainder_queue | failed | 0.09453125000000001 | 76.29850434884429 | 3.0 | 6.697320704630282e-06 | Picard residual not subset of target remainder |
