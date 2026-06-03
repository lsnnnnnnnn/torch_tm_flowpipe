# Flow* Benchmark Parity Report

This is a Flow* original benchmark parity audit for the plant-only polynomial Van der Pol benchmark. It is not a new reachability algorithm.

## Original Parameters

- ODE: `x' = y`, `y' = (1 - x^2) * y - x`, `t' = 1`
- Initial set: `x in [1.1, 1.4]`, `y in [2.35, 2.45]`
- Horizon: `0` to `10.0`
- Flow* original step policy: adaptive, min `0.002`, max `0.1`
- Flow* Taylor order: fixed order `4`
- Remainder estimation: `[-0.0001, 0.0001]`
- Cutoff threshold: `[-1e-10, 1e-10]`
- Symbolic remainder queue size: `100`
- Flow* patch used: no
- Benchmark PNG inputs: `/srv/local/shengenli/flowstar/images/benchmarks/vanderpol_t_x.png` and `/srv/local/shengenli/flowstar/images/benchmarks/vanderpol_t_y.png`

## Runtime And Bounds

| tool | status | runtime | segments | last width sum | tube width sum | endpoint box | source |
|---|---|---:|---:|---:|---:|---|---|
| `original_flowstar` | `completed` | 1.01494 | 290 | 0.704713 | 9.56631 | `False` | `flowstar_original_gnuplot_segment_boxes` |
| `generated_flowstar` | `completed` | 0.46567 | 290 | 0.704713 | 9.56631 | `False` | `flowstar_generated_gnuplot_segment_boxes` |
| `torch_tm` | `failed` | 6.18656 | 40 | 2.64069e+182 | 2.64069e+182 | `True` | `torch_tm_range_only_segment_on_flowstar_time_grid` |

`generated_flowstar` was generated from the parsed parameters and run through the repository Flow* toolbox runner. Its last-segment width sum is 0.704713.

Generated Flow* vs original Flow*: segment count match is `True` and max absolute parsed segment-field difference is `0`.

PyTorch TM status is `failed` with last reached time `0.666434` against requested horizon `10`.

## Semantics

Flow* GNUPLOT rectangles are segment boxes. They are not final-time endpoint boxes. Therefore `endpoint_box_available=false` for both Flow* rows, endpoint widths are blank, and no endpoint ratio is reported.

Only last-segment and tube widths are reported for Flow* parity. Plot generation time is not included in algorithm runtime.

## Scope Guard

No CROWN, no auto_LiRPA, no Jacobian bounds, no sin/cos support, no hybrid automata, no Flow* Python binding, no NN controller workflow, and no new algorithm were added.
