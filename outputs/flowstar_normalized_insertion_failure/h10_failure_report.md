# Normalized Insertion H10 Failure Localization

Inputs: existing `outputs/flowstar_normalized_insertion_h10` CSV artifacts.
This is a post-hoc localization report; it does not rerun the reachability kernel.

## Direct Answers

### o4 `flowstar_style_o4_target_insert`
- Failure near t=`6.4730088058091901` with h_try=`0.0034038826916583933`.
- Failed dimension: `y`.
- Residual width vs target: x `1.0841545978090407e-05` / `0.0002`, y `0.0001931656680284411` / `0.0002`.
- Shift or width? `y:positive_shift`.
- Dominant term: `symbolic missing`.
- Did failure happen after a width-ratio jump? `False`.
- Did failure happen after a cluster of step rejections? `False`.

### o6 `flowstar_style_o6_candidate8_output6_insert`
- Failure near t=`7.4960392581387341` with h_try=`0.0036197592495462228`.
- Failed dimension: `y`.
- Residual width vs target: x `9.5106126089377504e-07` / `0.0002`, y `0.00021264219517733133` / `0.0002`.
- Shift or width? `y:width`.
- Dominant term: `symbolic missing`.
- Did failure happen after a width-ratio jump? `False`.
- Did failure happen after a cluster of step rejections? `False`.

## Comparison

Why does o6 go farther but become much wider? o6/candidate8 goes farther because higher candidate order validates more steps, but its normalized reset boxes accumulate much larger widths.
Why does o4 stay tighter but fail earlier? The order4 path keeps lower-order, closer-to-Flow* reset boxes, so widths remain smaller, but the final residual margin is exhausted earlier.
Which path should be prioritized for h10 parity? o4 for Flow*-settings parity; keep o6/candidate8 as the reachability stress path.

## Summary Rows

| run_id | last_validated_t | failure_t | h_try | failed_dimension | shift_or_width | dominant_term | width_ratio | rejection_cluster |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- |
| flowstar_style_o4_target_insert | 6.4730088058091901 | 6.4730088058091901 | 0.0034038826916583933 | y | y:positive_shift | symbolic missing | 14.049230879967741 | False |
| flowstar_style_o6_candidate8_output6_insert | 7.4960392581387341 | 7.4960392581387341 | 0.0036197592495462228 | y | y:width | symbolic missing | 101.26404571229142 | False |
