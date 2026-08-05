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
- Natural range still stops the authoritative VDP order-4 lane at
  T=6.3172908799330765. The validated four-leaf terminal fix closes that step,
  but the final proactive fresh lane stops at T=6.397083942944808 with y
  self-map margin `-1.99995911680722e-5`; T=7.5 and T=10 remain unclosed.
- Subdivision uses safeguarded float64 arithmetic and complete-cover tests, not
  a machine-checked hardware-independent directed-rounding proof. Sampling is
  only an independent sanity check.
- The production policy is intentionally limited to four leaves on the
  attributed `polynomial_truncation` context. Frozen depths through the 64-leaf
  cap do not improve the later terminal, and no second policy adjustment or
  reduced h_min was used.
- CUDA correctness is exercised, but batch-1 CUDA is slower and only multiply
  wins at batch 128 in the recorded V100 microbenchmark. No end-to-end GPU or
  cross-tool speedup is claimed.
