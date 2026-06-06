# Flowstar Insertion Width Attribution Report

Which component causes width growth? `right_map_scaling`; max component widths: {'insertion_truncation': 7.577728098712013e-05, 'right_map_scaling': 21.883875748631645, 'picard_residual': 0.00021478626710226962, 'output_range_evaluation': 21.8007462760906}.
Why does o6 run farther but widen much more? o6 reaches t=`7.496039258138734` with max reset width `21.90299970246541`, while o4 reaches t=`6.47300880580919` with max reset width `4.820920894732831`.
Why does o4 fail earlier but stay tighter? Its lower order/target path keeps reset widths smaller but leaves less residual slack near failure.
Does normalized right map degree/term count explode? max terms before/after insertion are reported in the CSV; dominant component is `right_map_scaling`.
Is the next fix source-map insertion Horner/intermediate range, symbolic queue, or polynomial range bounding? `right-map scalar alignment`.

## Component Maxima

| component | max_width |
| --- | ---: |
| insertion_truncation | 7.5777280987120131e-05 |
| right_map_scaling | 21.883875748631645 |
| picard_residual | 0.00021478626710226962 |
| output_range_evaluation | 21.800746276090599 |

This report is diagnostic-only and does not claim exact Flow* parity.
