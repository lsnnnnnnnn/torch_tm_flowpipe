# Original Flow* Van der Pol Parameters

- Source: `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp`
- ODE: `x' = y`, `y' = (1 - x^2) * y - x`, `t' = 1`
- Initial set: `x in [1.1, 1.4]`, `y in [2.35, 2.45]`, `t = 0`
- Time horizon: `[0, 10.0]`
- Safe set: `y - 2.75 <= 0`
- Step policy: `adaptive`, min `0.002`, max `0.1`
- Taylor order policy: `fixed`, order `4`
- Remainder estimation: `[-0.0001, 0.0001]` for each declared variable
- Cutoff threshold: `[-1e-10, 1e-10]`
- Symbolic remainder queue size: `100`
- Plot commands:
  - `plot_setting.plot_2D_interval_GNUPLOT("./", "vanderpol_t_x", result.tmv_flowpipes, setting)`
  - `plot_setting.plot_2D_interval_GNUPLOT("./", "vanderpol_t_y", result.tmv_flowpipes, setting)`
- Plot files: `vanderpol_t_x.plt`, `vanderpol_t_y.plt`
- EPS files: `vanderpol_t_x.eps`, `vanderpol_t_y.eps`
- Benchmark PNG files: `/srv/local/shengenli/flowstar/images/benchmarks/vanderpol_t_x.png`, `/srv/local/shengenli/flowstar/images/benchmarks/vanderpol_t_y.png`

No Flow* source patch was used.
