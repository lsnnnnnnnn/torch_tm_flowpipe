# Batched TM GPU Microbenchmark Report

This report is diagnostic-only. It does not claim a new reachability algorithm, and it does not use the Flow* C++ probe as an implementation route.

## Run Metadata

- Output directory: `outputs/batched_tm_gpu_microbench`
- PyTorch version: `2.5.1+cu121`
- CUDA available: `True`
- CUDA device: `Tesla V100-SXM2-16GB`
- dtype: `float64`
- OK rows: `210`
- Skipped rows: `105`

## Direct Answers

- At batch=1, is PyTorch GPU slower than CPU? Yes for 12/15 measured batch=1 CUDA rows; 3/15 were faster than torch CPU.
- What batch size is needed before GPU wins, if any?

| operation | first CUDA batch with speedup > 1.0 |
| --- | --- |
| interval_affine_map | 8192 |
| poly_coeff_add | 32 |
| poly_coeff_mul_trunc | 1 |
| tm_range_bound | 32 |
| fixed_picard_tm_step | 1 |

- Which operation dominates torch CPU runtime? fixed_picard_tm_step (1.32e+03 ms summed over largest measured batches)
- Which operation dominates torch CUDA runtime? tm_range_bound (14.8 ms summed over largest measured batches)
- Are current data structures tensorizable, or are Python dict/sparse loops blocking GPU? The current production `Polynomial`/`TaylorModel` path uses Python dictionaries keyed by exponent tuples and scalar tensors, so Python object and sparse-loop overhead blocks real GPU use. Existing sparse Python rows ran at 9.7e-05x to 0.091x of torch dense CPU throughput for the measured scalar batches.
- What representation change is needed for real GPU use? Use a canonical monomial basis per `(dim, order)`, store coefficients as batched dense or blocked-sparse tensors, precompute multiplication/truncation scatter plans, batch interval domains and remainders, and keep all hot-path arithmetic on device tensors.
- Is the project still justified as PyTorch-native, or should plant remain Flow* C++? Dense batched kernels show clear CUDA speedups at realistic batch sizes.

## Final Recommendation: GPU_PATH_PROMISING

Allowed recommendation values are `GPU_PATH_PROMISING`, `NEEDS_REPRESENTATION_REDESIGN`, and `STOP_PYTHON_PLANT_TM_FOR_SPEED`.
