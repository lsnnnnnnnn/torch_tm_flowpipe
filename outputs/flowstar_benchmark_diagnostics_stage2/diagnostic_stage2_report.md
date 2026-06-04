# Flow* Benchmark PyTorch TM Stage-2 Diagnostics

This report is diagnosis only. It is not a new reachability algorithm and not a Flow* parity claim.

## A. Dominant Van der Pol RHS Blowup Term

Dominant term: **polynomial_range * remainder**.

- x*x truncation: max 80031.309555607659
- x*x remainder multiplication: max 6.8715595480124722e+305
- (x*x)*y truncation: max 361301378805.49066
- polynomial_range * remainder: max 7.7410783012361911e+307
- remainder * remainder: max 4.6504375243797575e+307
- RHS aggregation: max 4.6504375243797605e+307
- interval polynomial range evaluation: max 9763427159.7739735

## B. Dependency Reset Window

Yes. Best last-validated times by reset window: K=1: 0.75439784999999993, K=2: 0.75439784999999993, K=4: 0.748515025, K=8: 0.74263219999999996, K=inf: 0.54337385000000005. 10 dependency-window rows hit the wall-time cap before failure; those rows are still useful as capped diagnostics, not success claims.

## C. Adaptive Bisection

No. Best run `range_only_o6_b8` reached t=0.76009683671874995, so width/remainder blowup still dominates before beating t=0.7661635.

## D. Validation Parameter Tuning

`dependency_preserving_o6_s1` best reached t=0.46280189999999999 (delta 0 over default). `range_only_o8_s1` best reached t=0.66643399999999997 (delta 0 over default).

## E. Next Minimal Implementation Target

Pick exactly one: **symbolic remainder handling**.
