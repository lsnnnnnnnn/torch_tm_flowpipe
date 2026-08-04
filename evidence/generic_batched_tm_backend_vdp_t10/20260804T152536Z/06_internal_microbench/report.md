# Batched TM GPU Microbenchmark Report

This report is diagnostic-only. It does not claim a new reachability algorithm, and it does not use the Flow* C++ probe as an implementation route.

## Run Metadata

- Output directory: `outputs/generic_batched_tm_backend_vdp_t10/20260804T152536Z/06_internal_microbench/production_cpu_cuda`
- PyTorch version: `2.5.1+cu121`
- CUDA available: `True`
- CUDA device: `Tesla V100-SXM2-16GB`
- dtype: `float64`
- OK rows: `60`
- Skipped rows: `15`

## Direct Answers

- At batch=1, is PyTorch GPU slower than CPU? Yes for 5/5 measured batch=1 CUDA rows; 0/5 were faster than torch CPU.
- What batch size is needed before GPU wins, if any?

| operation | first CUDA batch with speedup > 1.0 |
| --- | --- |
| tm_affine_map | no CUDA win measured |
| polynomial_add | no CUDA win measured |
| tm_mul_trunc | 128 |
| tm_range_bound | no CUDA win measured |
| picard_validate_step | no CUDA win measured |

- Which operation dominates torch CPU runtime? picard_validate_step (175 ms summed over largest measured batches)
- Which operation dominates torch CUDA runtime? picard_validate_step (203 ms summed over largest measured batches)
- Does the production representation expose tensorized work? Yes: these dense rows call `BatchedPolynomial`, `BatchedTaylorModel`, and `dense_picard_validate_step` directly. The sparse reference remains dictionary based. Existing sparse Python rows ran at 0.000887x to 0.577x of torch dense CPU throughput for the measured scalar batches.
- What representation is measured? A canonical complete monomial basis with batched coefficient/domain/remainder tensors and cached multiplication/integration routes; VDP order 4 uses three variables and 35 slots.
- Is the project still justified as PyTorch-native, or should plant remain Flow* C++? The dense batched representation is operational, but its measured CUDA advantage is too narrow for an end-to-end GPU claim.

## Final Recommendation: DENSE_GPU_PATH_LIMITED

Allowed recommendation values are `GPU_PATH_PROMISING`, `DENSE_GPU_PATH_LIMITED`, `NEEDS_REPRESENTATION_REDESIGN`, and `STOP_PYTHON_PLANT_TM_FOR_SPEED`.
