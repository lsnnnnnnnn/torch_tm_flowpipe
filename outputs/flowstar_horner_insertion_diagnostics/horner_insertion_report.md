# Horner Insertion Diagnostic Report

Requested diagnostic horizon: `0.0040000000000000001`; runs stop earlier if validation fails.
Does Horner diagnostic reduce inserted endpoint/right-map range compared to direct substitution? no.
Does Horner diagnostic change the inserted range? no.
Best range delta row: `flowstar_style_o6_candidate8_output6_insert` segment `0` at t=`0.004` with delta=`-7.771561172376096e-16`.
Peak direct range row: `flowstar_style_o4_target_insert` at t=`0.004` width=`0.408931185144492`.
Peak Horner range row: `flowstar_style_o4_target_insert` at t=`0.004` width=`0.40893118514449184`.
Which stage dominates width? `x` / `add_outer_remainder` at stage `7` with width=`0.3003842899825029`.
Largest uncertainty component: `result_range_width` in `x` stage `18` width=`0.29998394388370875`.
Does time branch matter? no; accumulated time-branch stage range width=`0`.
Does state y branch dominate? no; x-stage width sum=`1.8023077921979609`, y-stage width sum=`0.65393440369333433`.
Is the current direct substitution over-conservative or under-accounting? the Horner diagnostic is numerically equal to direct substitution for the recorded reset ranges.
Is Horner diagnostic conservative under sampling? yes for the helper-level sampling tests added with this task; this report is diagnostic-only and does not claim full Flow* parity.

## Peak Rows

| run_id | segment | t_hi | direct_width | horner_width | delta | reduced | dominant_stage |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| flowstar_style_o4_target_insert | 0 | 0.004 | 0.408931185144492 | 0.40893118514449184 | -1.6653345369377348e-16 | False | x:add_outer_remainder |
| flowstar_style_o6_candidate8_output6_insert | 0 | 0.004 | 0.4089307320883276 | 0.4089307320883268 | -7.771561172376096e-16 | False | x:add_outer_remainder |

This report is diagnostic-only and does not claim exact Flow* parity.

## Failure-Neighborhood Replay Limitation

The full failure-neighborhood Horner diagnostic was attempted in two forms during this task: first across all resets, then narrowed to the final validated reset per baseline. Both replays were stopped manually after they did not return in practical time. The bounded diagnostic artifacts above are therefore real helper/stage outputs on the same two baseline configurations at an early validated reset, but they are not a completed high-degree t≈6.473/t≈7.496 failure-neighborhood measurement.

Decision consequence: because the bounded diagnostic shows no material direct-vs-Horner range change and the high-degree failure-neighborhood diagnostic is too expensive in the current clean-room implementation, the opt-in h10 Horner reset run was not executed.

