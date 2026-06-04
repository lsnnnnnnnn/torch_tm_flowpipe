# Flowstar-Style Rescue Report

Requested max horizon: `5`.
Best old baseline in this run: `` at t=`0`.
Best flowstar_style run: `flowstar_style_o6_target_r5e-4` at t=`2.2567245519061094`.

Did flowstar_style beat the old best t~=0.7661635? yes.
Did flowstar_style_o6_target reach the requested horizon? no.
Did cutoff help? inconclusive.
Did target remainder stay bounded at width sum 0.0004? no; max target-mode remainder width sum was `0.0020000000000000005`.
Did recenter/rescale help compared to range_only and dependency_preserving? yes; best flowstar_style t=`2.2567245519061094` vs best baseline t=`0`.
Best rescue candidate: `flowstar_style_o6_target_r5e-4`.
Accepted/rejected steps for best rescue: `105` accepted, `67` rejected.
min_regular_h_used for best rescue: `0.0022039725596597255`.
Did any non-final step go below Flow* min step 0.002? no.
How do widths compare to original Flow* over the same horizon? last width ratio=`52.322206004362734`, tube width ratio=`2.9715313585372285`.
Is this a reachability success, a tightness success, or both? neither yet.
Failure mode for the best rescue candidate: `Picard residual not subset of target remainder`.
Do not treat this as Flow* parity unless horizon 10 is reached and boxes are compared separately.

## Summary Rows

| run_id | status | last_validated_t | accepted | rejected | min_h_used | min_regular_h_used | non_final_h_below_0.002 | failure_reason |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| flowstar_style_o6_target_r2e-4 | failed | 2.177556584500918 | 120 | 76 | 0.002120594498529912 | 0.002120594498529912 | 0 | Picard residual not subset of target remainder |
| flowstar_style_o6_target_r5e-4 | failed | 2.2567245519061094 | 105 | 67 | 0.0022039725596597255 | 0.0022039725596597255 | 0 | Picard residual not subset of target remainder |
