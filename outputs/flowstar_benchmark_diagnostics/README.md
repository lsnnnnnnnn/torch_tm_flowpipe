# Flow* Benchmark PyTorch TM Failure Diagnostics

This directory contains Stage-1 diagnostic artifacts for the PyTorch Taylor-model failure on the original Flow* Van der Pol benchmark segment grid.

## What This Stage Tested

Stage-1 varied PyTorch TM mode, Taylor order, and substep factor:

- `range_only` and `dependency_preserving`
- orders 4, 6, and 8
- substep factors 1, 2, and 4

## Best Run

The best fixed diagnostic run was `range_only_o6_s4`, which validated 176 diagnostic subsegments and last validated t ~= 0.7661635 before failing on the next attempted segment at t ~= 0.7726346.

## Main Conclusion

Higher order and smaller substeps only marginally delayed failure. The PyTorch TM prototype still failed far before the Flow* horizon 10 objective, with non-finite residual interval failure dominating the final stage of the run.

## Why This Is Not A Flow* Parity Claim

These runs split the original Flow* segment grid for PyTorch diagnostics only. They failed before horizon 10 and do not provide a successful replacement algorithm or a Flow* parity result.
