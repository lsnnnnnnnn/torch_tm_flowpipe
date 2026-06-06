# Right Map Scaling Diagnostics Report

Which right-map dimension drives width? `y`; peak x=`4.1966738620765964`, peak y=`17.687201886555048`.
At what time does right_map_scaling begin dominating? Peak persisted right-map range occurs near t=`7.496039258138734`.
Is range dominated by a few monomials or many terms? Persisted diagnostics do not include exact sparse monomial contributions; aggregate top-term rows show the largest recorded width channels.
Are high-degree terms the issue? Max recorded right-map degree is `6`; compare against term-count plot before concluding.
Does time variable evaluation contribute? no in persisted endpoint/right-map diagnostics; the right map after endpoint substitution has no local time variable.
Is current evaluation using a larger domain than Flow* normal evaluation would? Normal-vs-old persisted ranges shrink: no or unavailable; old peak=`21.883875748631645`, normal peak=`21.883875748631645`.
Is o6 wide because of more terms or larger coefficients/scales? Max term-count row is `flowstar_style_o6_candidate8_output6_insert` at t=`0.125`; scale and width columns in the CSV show the o6 range/scale channel dominates.

## Peak Rows

| run_id | t_hi | current_range_sum | normal_range_sum | reset_width_sum | ratio_to_flowstar | terms_y | degree_y |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flowstar_style_o6_candidate8_output6_insert | 7.496039258138734 | 21.883875748631645 | 21.883875748631645 | 21.90299970246541 | 101.26404571229142 | 27.0 | 6.0 |
| flowstar_style_o6_candidate8_output6_insert | 7.493626085305703 | 21.430226201099863 | 21.430226201099863 | 21.449424726931547 | 101.26404571229142 | 27.0 | 6.0 |
| flowstar_style_o6_candidate8_output6_insert | 7.4904085215283285 | 20.855154188141835 | 20.855154188141835 | 20.874455644223552 | 96.51148855675044 | 27.0 | 6.0 |
| flowstar_style_o6_candidate8_output6_insert | 7.488263479010079 | 20.485320041309905 | 20.485320041309905 | 20.504688610068996 | 96.51148855675044 | 27.0 | 6.0 |
| flowstar_style_o6_candidate8_output6_insert | 7.485403422319079 | 20.01327362888817 | 20.01327362888817 | 20.03273383563185 | 94.79616753935854 | 27.0 | 6.0 |
| flowstar_style_o6_candidate8_output6_insert | 7.481590013397747 | 19.418791286259392 | 19.418791286259392 | 19.43837870610279 | 89.86958308721789 | 27.0 | 6.0 |
| flowstar_style_o6_candidate8_output6_insert | 7.479047740783525 | 19.0380584397948 | 19.0380584397948 | 19.057728456634372 | 89.86958308721789 | 27.0 | 6.0 |
| flowstar_style_o6_candidate8_output6_insert | 7.475658043964563 | 18.554777732809665 | 18.554777732809665 | 18.574560944941172 | 83.44172430187744 | 27.0 | 6.0 |

This report is diagnostic-only and does not claim exact Flow* parity.
