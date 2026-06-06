# Flowstar-Style Rescue Report

Requested max horizon: `10`.
Best old baseline in this run: `` at t=`0`.
Best flowstar_style run: `flowstar_style_o6_candidate8_output6_insert_symqueue` at t=`3.3500000000000014`.

Did flowstar_style beat the old best t~=0.7661635? yes.
Did flowstar_style_o6_target reach the requested horizon? no.
Did cutoff help? tied.
Did target remainder stay bounded at width sum 0.0004? yes; max target-mode remainder width sum was `0.0004000000000000001`.
Did recenter/rescale help compared to range_only and dependency_preserving? yes; best flowstar_style t=`3.3500000000000014` vs best baseline t=`0`.
Best rescue candidate: `flowstar_style_o6_candidate8_output6_insert_symqueue`.
Accepted/rejected steps for best rescue: `38` accepted, `12` rejected.
min_regular_h_used for best rescue: `0.025`.
Did any non-final step go below Flow* min step 0.002? no.
How do widths compare to original Flow* over the same horizon? last width ratio=`0.7039951932518582`, tube width ratio=`0.9951883436176395`.
Is this a reachability success, a tightness success, or both? neither yet.
Failure mode for the best rescue candidate: `initial or cutoff remainder exceeds target remainder`.
Do not treat this as Flow* parity unless horizon 10 is reached and boxes are compared separately.

## Summary Rows

| run_id | status | last_validated_t | accepted | rejected | min_h_used | min_regular_h_used | non_final_h_below_0.002 | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o4_target_insert_symqueue | failed | 3.217705703618956 | 77 | 45 | 0.012670540809631349 | 0.012670540809631349 | 0 | initial or cutoff remainder exceeds target remainder |
| flowstar_style_o4_target_cutoff_insert_symqueue | failed | 3.217705703618956 | 77 | 45 | 0.012670540809631349 | 0.012670540809631349 | 0 | initial or cutoff remainder exceeds target remainder |
| flowstar_style_o6_candidate8_output6_insert_symqueue | failed | 3.3500000000000014 | 38 | 12 | 0.025 | 0.025 | 0 | initial or cutoff remainder exceeds target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_insert_symqueue | failed | 3.3500000000000014 | 38 | 12 | 0.025 | 0.025 | 0 | initial or cutoff remainder exceeds target remainder |
