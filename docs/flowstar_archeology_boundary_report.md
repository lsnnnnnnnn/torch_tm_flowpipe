# Flow* Archaeology Boundary Report

## Executive Conclusion

Flow* archaeology has reached a boundary useful enough to stop. The current exposed mismatch is inside Flow* Picard-ctrunc raw remainder accumulation before the x0 add. Further progress would require deeper Flow* source instrumentation around intermediate ranges, Expression/Horner evaluation, and remainder accumulation internals; that is microscope work, not project-level progress for the batched/GPU Taylor-model path.

This report freezes the A-line diagnosis and defines the B-line handoff. It does not claim Flow* parity beyond the evidence already collected, and it does not add new Flow* mechanisms.

## A-Line Evidence Chain

- Original/generated Flow* parity is exact for the audited generated model path.
- Short horizon / h5 normalized insertion can be width-close.
- h10 late width is not close, and the failure is not merely a timeout artifact.
- no_queue, split, and v2 queue variants did not rescue h10.
- same-t/h acceptance differs: Flow* rejects h=0.025 while the PyTorch path accepts.
- same-source full-step tube comparison is width-close.
- Polynomial range evaluation is the same across the audited paths.
- raw ctrunc returned remainder differs.
- The internal intermediate range audit localizes the exposed raw y_hi gap to accumulated remainder before x0 add.

## Stop Rules

- No new Flow* probe work unless explicitly requested.
- No h10 reruns by default.
- No new symbolic queue variants.
- The C++ Flow* probe is frozen as microscope-only.
- Do not modify local Flow* source as part of the B-line path.

## B-Line Next

- Use a dense batched Taylor-model representation for GPU-shaped tensor kernels.
- Build a many-box plant workload with fixed-step, fixed-order explicit Euler-style TM propagation.
- Build a simple NNCS/control-bound workload with batched state boxes, controller bounds, and plant updates.
- Compare scalar loop, dense CPU, and dense CUDA timings where available.
- Check sampled containment as sanity only, not proof.

## Relevant Outputs

- `outputs/flowstar_internal_intermediate_ranges_audit/`
- `outputs/flowstar_raw_remainder_partition_audit/`
- `outputs/flowstar_raw_ctrunc_residual_audit/`
- `outputs/flowstar_validation_candidate_decomposition_audit/`
- `outputs/flowstar_same_source_endpoint_tube_compare/`
- `outputs/flowstar_vdp_width_trajectory_audit/`
