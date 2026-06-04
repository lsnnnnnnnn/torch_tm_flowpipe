# Flow* Benchmark PyTorch TM Stage-3 Diagnostics

This directory contains Stage-3 diagnostic artifacts for an experimental symbolic remainder prototype on the Flow* Van der Pol benchmark segment grid.

## What This Stage Tested

Stage-3 compared symbolic-remainder diagnostic runs against a local `range_only_o6_s4_baseline`, using queue sizes 4, 8, and 16 for `range_only` and `dependency_window_2` variants.

## Best Run

The best symbolic run was `dependency_window_2_symbolic_o4_s4_q4`, which reached t = 0.1441472. The local baseline row `range_only_o6_s4_baseline` reached t = 0.7661635.

## Main Conclusion

The symbolic prototype reduced local ordinary interval-remainder interaction, but it was too slow and did not beat baseline. Most symbolic runs hit wall-time caps.

## Why This Is Not A Flow* Parity Claim

This is diagnostic-only. It is not part of the supported default API, did not reach horizon 10, did not improve the benchmark objective, and does not claim Flow* parity or a new algorithm.
