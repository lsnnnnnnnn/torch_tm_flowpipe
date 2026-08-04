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
- The dense flowpipe is hybrid: normalized insertion and cross-step right-map
  composition remain sparse/CPU boundary work. It is S3, not full-dense S5.
- The authoritative VDP order-4 lane stops at T=6.3172908799330765 because the
  y remainder self-map fails before h_min; T=10 is not closed.
- CUDA correctness is exercised, but batch-1 CUDA is slower and only multiply
  wins at batch 128 in the recorded V100 microbenchmark. No end-to-end GPU or
  cross-tool speedup is claimed.
