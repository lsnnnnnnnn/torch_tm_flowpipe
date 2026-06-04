# Flow* Benchmark PyTorch TM Failure Diagnostics

This is a diagnostic report for the PyTorch Taylor-model failure on the original Flow* Van der Pol benchmark grid. It is diagnosis only, not a new reachability algorithm.

## Run Summary

| run | status | validated segments | last validated t | last attempted t | failure reason |
|---|---|---:|---:|---:|---|
| `range_only_o4_s1` | `failed` | 39 | 0.63428960000000001 | 0.66643399999999997 | `non-finite residual interval` |
| `range_only_o4_s2` | `failed` | 84 | 0.72124010000000005 | 0.73193615000000001 | `non-finite residual interval` |
| `range_only_o4_s4` | `failed` | 174 | 0.75439784999999993 | 0.76028067499999996 | `non-finite residual interval` |
| `range_only_o6_s1` | `failed` | 40 | 0.66643399999999997 | 0.70179279999999999 | `non-finite residual interval` |
| `range_only_o6_s2` | `failed` | 86 | 0.74263219999999996 | 0.75439784999999993 | `non-finite residual interval` |
| `range_only_o6_s4` | `failed` | 176 | 0.7661635 | 0.77263459999999995 | `non-finite residual interval` |
| `range_only_o8_s1` | `failed` | 40 | 0.66643399999999997 | 0.70179279999999999 | `Picard remainder validation did not converge` |
| `range_only_o8_s2` | `timeout` | 73 | 0.56642665000000003 | 0.57850190000000001 | `wall-time cap reached during validation call` |
| `range_only_o8_s4` | `timeout` | 72 | 0.25754589999999999 | 0.26188957499999999 | `validation exception: wall-time cap reached during validation call` |
| `dependency_preserving_o4_s1` | `failed` | 33 | 0.49429260000000003 | 0.51243720000000004 | `non-finite residual interval` |
| `dependency_preserving_o4_s2` | `failed` | 69 | 0.5224167500000001 | 0.53239630000000004 | `non-finite residual interval` |
| `dependency_preserving_o4_s4` | `failed` | 142 | 0.54337385000000005 | 0.54886262500000005 | `non-finite residual interval` |
| `dependency_preserving_o6_s1` | `failed` | 31 | 0.46280189999999999 | 0.47779739999999998 | `non-finite residual interval` |
| `dependency_preserving_o6_s2` | `timeout` | 65 | 0.486045 | 0.49429260000000003 | `wall-time cap reached during validation call` |
| `dependency_preserving_o6_s4` | `timeout` | 66 | 0.234571 | 0.23816080000000001 | `wall-time cap reached during validation call` |
| `dependency_preserving_o8_s1` | `timeout` | 22 | 0.31610739999999998 | 0.32882660000000002 | `wall-time cap reached during validation call` |
| `dependency_preserving_o8_s2` | `timeout` | 22 | 0.1640424 | 0.17295834999999998 | `wall-time cap reached during validation call` |
| `dependency_preserving_o8_s4` | `timeout` | 21 | 0.081346594999999994 | 0.086379440000000002 | `wall-time cap reached during validation call` |

## Questions

- Did higher order delay failure? Yes: range_only substep 1: order 6 reached 0.666434 vs order 4 at 0.63429; range_only substep 2: order 6 reached 0.742632 vs order 4 at 0.72124; range_only substep 4: order 6 reached 0.766163 vs order 4 at 0.754398
- Did smaller substeps delay failure? Yes: range_only order 4: substep 4 reached 0.754398 vs factor 1 at 0.63429; range_only order 6: substep 4 reached 0.766163 vs factor 1 at 0.666434; dependency_preserving order 4: substep 4 reached 0.543374 vs factor 1 at 0.494293; dependency_preserving order 6: substep 2 reached 0.486045 vs factor 1 at 0.462802
- Did dependency-preserving still fail earlier than range-only? Yes. Dependency-preserving validated to 0.494293, while range-only validated to 0.63429.
- Is the blowup dominated by polynomial range width, interval remainder width, residual width, or non-finite arithmetic? Dominant signal: non-finite arithmetic; polynomial range width max 873.70913120795512, interval remainder width max 2.0482044227241504e+38, residual width max 1.6385635381793194e+38.
- What bottleneck does this suggest? The immediate bottleneck is non-finite arithmetic in validation, after earlier width growth.

The substep-factor runs split each original Flow* segment for PyTorch diagnostics only. They are not Flow* parity claims. No successful replacement algorithm is claimed here.
