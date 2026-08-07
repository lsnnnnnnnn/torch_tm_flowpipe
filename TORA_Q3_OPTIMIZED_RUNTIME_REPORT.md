# TORA-Q3 optimized runtime report

All formal GPU timings below use the matched CROWN software stack. Cold compile/warm-up is reported separately and excluded from steady samples.

## Outcomes

- eager B48 one-step median: `0.680230` s
- compiled B48 one-step median: `0.508397` s (`4.621x`)
- compiled common-control T20 median: `105.480052` s (`4.854x`)
- compiled T20 IQR: `0.033973` s

P0, P1, and P2 pass. P3 and P4 do not reach the required 10x; no 10x claim is made.

## Gate table

| gate | observed | status |
|---|---:|---|
| P0 | frozen one-leaf hash exact; compiled first-call bitwise verification; CPU/CUDA predicate tests | PASS |
| P1 | 3 | PASS |
| P2 | 99.583897% | PASS |
| P3 | 4.621012x | FAIL |
| P4 | 4.854230x | FAIL |

The remaining concrete engineering boundary is a pure tensor kernel for natural range/remainder arithmetic across the whole K2 + ten-round phase. Compiling only the point sine/cosine core cannot remove the thousands of small interval kernels.
