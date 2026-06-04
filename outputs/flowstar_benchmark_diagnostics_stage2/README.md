# Flow* Benchmark PyTorch TM Stage-2 Diagnostics

This directory contains Stage-2 diagnostic artifacts for the PyTorch Taylor-model failure on the original Flow* Van der Pol benchmark segment grid.

## What This Stage Tested

Stage-2 tested RHS blowup attribution, dependency reset windows, adaptive bisection, and validation parameter tuning after the Stage-1 failure runs.

## Best Run

The best Stage-2 run was `range_only_o6_b8`, which reached t ~= 0.7600968. It did not beat the Stage-1 best fixed run, `range_only_o6_s4`, at t ~= 0.7661635.

## Main Conclusion

The dominant Van der Pol RHS blowup term was polynomial_range * remainder, with remainder * remainder interactions also contributing in `x*x*y`. Dependency windowing, bisection, and validation tuning did not remove the long-horizon failure.

## Why This Is Not A Flow* Parity Claim

These artifacts localize a failure mode. They do not reach horizon 10, do not match Flow* output, and do not implement or validate a new algorithm.
