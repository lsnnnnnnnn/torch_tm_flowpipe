# Flowstar Ctrunc Validation Report

This opt-in mode uses a clean-room Flow*-style Picard ctrunc validation decision. It does not replace the default target-remainder validator.
Requested horizon: `5`.
Best ctrunc variant: `flowstar_style_o6_candidate8_output6_flowstar_ctrunc` at t=`2.400737667399793`.
Did flowstar_ctrunc validation beat t~=2.400737? no.
Did it reach horizon 5? no.
Which dimension still fails? `y`.
Is the failure still shift or width? `width`.
Does normal eval reduce the residual shift? yes; ordinary max center=`3.363649456839351e-05`, normal max center=`0.0`.
Runtime impact: best runtime_s=`442.6456627137959`.
Width ratio vs Flow*: last=`68.36824483254583`, tube=`4.835529788768181`.

## Rows

| run_id | status | last_validated_t | runtime_s | tmp_subset_fail_dim | last_width_ratio | tube_width_ratio | failure_reason |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| flowstar_style_o6_target_flowstar_ctrunc | failed | 2.1095541733932355 | 215.47526893299073 | y | 30.05005498910863 | 1.962772284043342 | Flowstar ctrunc tmp remainder not subset of target remainder |
| flowstar_style_o6_candidate8_output6_flowstar_ctrunc | failed | 2.400737667399793 | 442.6456627137959 | y | 68.36824483254583 | 4.835529788768181 | Flowstar ctrunc tmp remainder not subset of target remainder |
| flowstar_style_o6_candidate8_output6_cutoff_flowstar_ctrunc | failed | 2.400737667399793 | 448.6030978066847 | y | 68.36824483254506 | 4.835529788768126 | Flowstar ctrunc tmp remainder not subset of target remainder |
