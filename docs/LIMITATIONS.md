# Limitations

- Float64 Torch and JAX computations are not formal real-arithmetic proofs.
- Sampling-based nonlinear trajectory containment is a deterministic
  regression sanity check, not proof-grade enclosure evidence.
- Flowstar is the external interval/Taylor-model reference; the local patch
  worktree is explicitly versioned and is not described as stock upstream.
- Native order numbers do not imply matched basis, validation, reset, or
  arithmetic across tools.
- Tightened and raw endpoints are intentionally not mixed.
- Configuration-level peak memory is unavailable; long-lived-process
  `ru_maxrss` is not used.
- The Apple Silicon formal environment has no NVIDIA CUDA device. CUDA checks
  must carry an explicit skip reason and are not merged with historical Linux
  timing.
- The supported formal runner covers the versioned selected practical
  configurations. Historical diagnostic matrices remain recoverable from
  archive tags but are not headline results.
