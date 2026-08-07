# TORA-Q3 optimized runtime report

All formal GPU timings below use the matched CROWN software stack. Cold compile/warm-up is reported separately and excluded from steady samples.

## Outcomes

- eager B48 one-step median: `0.680230` s
- compiled B48 one-step median: `0.508397` s (`4.621x`)
- compiled common-control T20 median: `105.480052` s (`4.854x`)
- compiled T20 IQR: `0.033973` s

P0, P1, and P2 pass. P3 and P4 do not reach the required 10x; no 10x claim is made.
The `4.621x` and `4.854x` values are matched-stack workload runtime ratios.
Because P3/P4 fail, they are not presented as a claim that the required GPU
acceleration gates passed.

## Protocol and resource evidence

- one-step eager and compiled: 10 measured repeats each
- common-control T20: one excluded complete warm-up plus 5 measured repeats
- T20 min/median/max: `105.443672 / 105.480052 / 105.528960` s
- T20 peak CPU resident memory: `1,487,536,128` bytes
- T20 peak CUDA allocation: `1,004,825,088` bytes
- compiled point-kernel graph breaks: `0`
- repeat status/checksum: stable across all five T20 repeats
- controller/serialization time inside the matched replay solver scope: zero

The cold compiled one-step warm-up was `21.312084` s; the point-kernel compile
and first call within it was `11.751797` s. Neither is included in steady
samples. Full stage medians, IQR, output hashes, and eager/compiled statistics
are in `outputs/tora_q3_perf_closure_20260806/optimized_kernel/summary.json`.

## Gate table

| gate | observed | status |
|---|---:|---|
| P0 | frozen one-leaf hash exact; compiled first-call bitwise verification; CPU/CUDA predicate tests | PASS |
| P1 | 3 | PASS |
| P2 | 99.583897% | PASS |
| P3 | 4.621012x | FAIL |
| P4 | 4.854230x | FAIL |

The remaining concrete engineering boundary is a pure tensor kernel for natural range/remainder arithmetic across the whole K2 + ten-round phase. Compiling only the point sine/cosine core cannot remove the thousands of small interval kernels.
