# Adaptive Order Rescue Report

Requested horizon: `5`.
Best adaptive-order variant: `flowstar_style_o6_target_adaptive_order_8` at t=`2.2771582567640953`.
Did adaptive order fallback beat t~=2.10955? yes.
Did it reach horizon 5? no.
Across all configs, accepted order-8 steps in this artifact: `88`; best-run order-8 steps=`44`.
If both cutoff and no-cutoff adaptive configs are present, the aggregate count is the total across those configs, not a single-run step count.
Runtime impact: best runtime_s=`472.1290770089254` vs h5 baseline runtime_s=`221.48196132015437`.
Width vs Flow* ratio: last=`80.93089904899215`, tube=`4.915137700918099`.
Did cutoff help? tied.

## Rows

| run_id | status | last_validated_t | order8_steps | runtime_s | failure_reason |
| --- | --- | ---: | ---: | ---: | --- |
| flowstar_style_o6_target_adaptive_order_8 | failed | 2.2771582567640953 | 44 | 472.1290770089254 | Picard residual not subset of target remainder |
| flowstar_style_o6_target_cutoff_adaptive_order_8 | failed | 2.2771582567640953 | 44 | 466.7356306249276 | Picard residual not subset of target remainder |
