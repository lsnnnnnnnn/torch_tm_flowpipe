# Torch DiffReach fixed-basis equivalence

Date: 2026-08-10

This is the required canonical filename for the fixed-support qualification.
The complete report is
[`TORCH_FIXED_SUPPORT_DIFFREACH_EQUIVALENCE_20260810.md`](TORCH_FIXED_SUPPORT_DIFFREACH_EQUIVALENCE_20260810.md).

The result is **qualified**:

- exact seven-slot support `[1, t, xi0, xi1, t^2, t*xi0, t*xi1]` with SHA256
  `0ae11ee9d911d45e42294df74ef2896ecb9aeb9f3d7851c09ea90e2bb2631f5e`;
- pinned float64 polynomial Picard, every DR-RP round, masks, retained
  intervals, endpoint, tube, and symbolic carry are bit-exact on the frozen
  fixture;
- CPU/CUDA decisions match; no runtime DiffReach/JAX/CROWN-Reach dependency;
- B64 T10 completes 1,000 steps, with all 128,000 initial and 1,280,000 later
  component masks true;
- ordinary Torch/CUDA float64 remains `empirically sampled only`; a 2-ULP
  companion envelope is `independently outward replayed for exact benchmark
  workload`, not a universal directed-rounding proof.

The stock full-driver endpoint differs by at most `3.6633409964e-6` because the
upstream launcher enables JAX x64 but leaves several builders at their default
float32 dtype. Operator equivalence is bit-exact when both paths use explicit
float64; completion and all available decisions agree.

Machine evidence is in
`outputs/mainline_realignment_20260810/20260810T025910Z/02_fixed_support/`.

