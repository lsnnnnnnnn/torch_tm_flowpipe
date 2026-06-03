# Flow* Benchmark Parity

This document describes the Van der Pol Flow* original benchmark parity audit in `outputs/flowstar_benchmark_parity/`.

The audit compares three artifacts:

- Original Flow*: `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp`
- Generated Flow*: a C++ harness generated from the parsed original parameters
- PyTorch TM: range-only TM propagation over the original Flow* segment time grid

The original benchmark parameters are parsed from Flow* source files, not guessed. The original benchmark uses adaptive stepsize defaults from `flowstar-toolbox/Continuous.cpp`: min step `0.002`, max step `0.1`, fixed Taylor order `4`, remainder estimation `[-1e-4, 1e-4]`, and cutoff `[-1e-10, 1e-10]`.

Flow* GNUPLOT rectangles are segment boxes. They are not final-time endpoint boxes, so Flow* rows use `endpoint_box_available=false` and do not report endpoint ratios. Parity reporting is limited to last-segment and tube widths. Runtime columns keep algorithm runtime separate from plot generation.

The original benchmark PNG references are:

- `/srv/local/shengenli/flowstar/images/benchmarks/vanderpol_t_x.png`
- `/srv/local/shengenli/flowstar/images/benchmarks/vanderpol_t_y.png`

No Flow* source patch is used. No CROWN, no auto_LiRPA, no Jacobian bounds, no sin/cos support, no hybrid automata, no Flow* Python binding, no NN controller workflow, and no new algorithm are added.
