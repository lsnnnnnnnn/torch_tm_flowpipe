# Batched Dense GPU B-Line Report

## Scope

This report combines the prior dense kernel evidence with the new many-box plant and NNCS workloads. It is a B-line report: fixed-step dense Taylor-model tensor experiments, not Flow* parity work, not adaptive reachability, and not h10 rescue.

The dense plant helper is explicitly named `fixed_euler_tm_step_vdp`; the older `fixed_picard_step_vdp` name remains only as a backwards-compatible alias. The experiments do not implement Flow* Picard integration.

## Prior Microbench

`outputs/batched_tm_gpu_microbench/gpu_microbench_report.md` showed that dense tensor kernels are a plausible GPU path. Batch=1 CUDA was usually slower than CPU, but large batches produced CUDA wins. The key representation lesson remains: production sparse `Polynomial` / `TaylorModel` objects are not the GPU route; a canonical dense or blocked-sparse monomial basis with precomputed scatter plans is.

## Dense Prototype

`outputs/batched_dense_tm_prototype/dense_tm_demo_report.md` was a fixed-step toy demo. It showed the dense path could beat scalar sparse loops and that CUDA became useful at larger batches, but it was not a Flow* replacement and did not provide adaptive validation or parity.

## Many-Box Plant Demo

New outputs:

- `outputs/batched_dense_many_box_flowpipe_demo/many_box_summary.csv`
- `outputs/batched_dense_many_box_flowpipe_demo/many_box_report.md`

Direct results from the completed CPU/CUDA sweep:

- Dense CPU beats the scalar loop on checked scalar batches.
- CUDA first beats dense CPU at batch 512.
- Sample containment passed: 0 violations in sampled fixed-Euler trajectories.
- Dominant operation: rhs construction, with dropped range bounds and range bounds visible inside that total.
- Merged dropped-term bounding reduced width; observed merged/termwise max-width ratios were about 0.91 to 0.94.
- `split2` range bounds tightened boxes, but the median measured cost was about 3.19x interval mode, so it is situational rather than a default throughput setting.

Many-box recommendation: `GPU_PATH_CONTINUE`.

## NNCS Demo

New outputs:

- `outputs/batched_dense_nncs_demo/nncs_summary.csv`
- `outputs/batched_dense_nncs_demo/nncs_report.md`

Direct results from the completed CPU/CUDA sweep:

- The batched NNCS loop ran end-to-end on CPU and CUDA for affine and ReLU-IBP controllers.
- CUDA first beats dense CPU at batch 2048, in the affine controller row.
- Closed-loop sampled containment passed: 0 violations.
- Dominant operation: plant step.
- Controller bound overhead was not dominant: average overhead was about 6.3%, with the largest row about 16.7%.
- Plant overhead dominated: average plant-step fraction was about 74.2%.

NNCS recommendation: `GPU_PATH_CONTINUE`.

## Decision

Final B-line decision: `GPU_PATH_CONTINUE`.

The path is worth continuing because the dense representation exposes real batch/GPU speedups, the many-box plant workload has a clear CUDA crossover, and the NNCS loop runs end-to-end with controller-bound overhead below plant propagation overhead.

The next engineering pressure point is not more Flow* archaeology. It is tighter and cheaper remainder/range bounding for wider boxes, plus richer controller linear bounds after the dense plant representation is less remainder-bound.
