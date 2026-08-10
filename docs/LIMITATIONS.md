# Limitations

- S1 is now integrated from `t=0`, but it does not reproduce the entire frozen
  historical prefix. Its sound common prefix ends after boundary 164 at
  `t=4.738198114669049`; the next proposed historical step is rejected. No
  terminal causal-improvement claim follows from the earlier empty-history
  local split.
- The endpoint and tube structured images are outward enclosures, not claimed
  exact Jacobian images. Complete degree-two through degree-four cross terms
  are placed in the nonlinear residual.
- The L1/L2 off-schedule half-step at the first divergence is diagnostic only.
  It is not committed, not counted as boundary 165, and not a fresh adaptive
  horizon result.
- Terminal A/B, fresh horizon, +0.5 promotion, T10, K32, and the integrated S1
  second-system run were prohibited by the prefix stop gate. Their table rows
  say `not_run_after_stop`; absence of a run is not interpreted as failure or
  success on those later gates.
- The schema-v2 checkpoint qualifies the complete boundary-164 prefix state,
  not the unavailable historical terminal prestate.

- Native rows use different representations, partitions, validators, step and
  carry policies, output objects, and numerical backends. They are not a winner
  table.
- The pinned stock Flow* VDP completes, but a clean scalar-affine MPFR oracle
  finds under-enclosure up to `3.4938679727147814e-10`; that build is
  `unsound/ineligible on a demonstrated counterexample` for the pinned native
  build and demonstrated workload. This is not a claim that the Flow* abstract
  algorithm is unsound.
- Stock DiffReach exposes initial DR-RP masks but not every later retain mask in
  its public result. It also mixes default float32 builders with JAX x64.
- Ordinary Torch/JAX/CUDA float64 is not universally directed-rounded. The
  fixed-support 2-ULP companion qualification applies only to the exactly
  replayed workload.
- Inductor fullgraph execution changes reduction arithmetic. It completes B64
  T10 but is performance-only and cannot support a same-ordinary-semantics
  speedup claim. Its first warm call performs additional lazy compilation.
- The fixed outward reference is CPU float64 and safeguarded only under the
  declared IEEE/PyTorch backend assumptions. It fails closed before T1; it is
  not a T10 formal certificate.
- The outward profiler records 369 CUDA kernel events per compiled logical
  step, but the prerefactor object CUDA kernel-launch count was not captured.
- Sampling is a regression sanity check, never a proof.
- The complete Torch baseline is hybrid: dense Picard/range/validation with
  sparse CPU normalized insertion and outer scheduling. It stops at
  `6.397083942944808`; T10 is partial.
- The complete-carry candidate is an experimental non-default lane. It is sound
  as an exact set-preserving clone but stops at `0.04345468750000001` and does
  not preserve the previously passing T=.1 certificate.
- The candidate's dense carry primitive is batch-generic through B512, but the
  adaptive multi-step complete scheduler remains B1. Batch figures state this
  kernel boundary and are not multi-step certificates.
- Host inclusion gates and audit extraction synchronize CUDA. The measured V100
  is slower at every tested batch; no GPU advantage is claimed.
- The preceding closure package's checkpoint predates S1; its local
  empty-history split remains attribution-only. The current package supersedes
  it only through the explicitly measured 164-boundary common prefix.
- NAV/DR15 is absent from the pinned DiffReach tree. The generality result uses
  the specified harmonic/Riccati fallback and has no navigation property or
  controller claim.
- Flow* observer replays are diagnostic counterfactuals, not production
  dependencies or modified native results.
- Endpoint, last-segment tube, and full-horizon/prefix tube are kept separate.
  Missing stock objects are `UNAVAILABLE`, never fabricated.
- TORA complete-Q3 is frozen historical stress-test evidence. No TORA-specific
  controller, formula, or support enters the generic source modules.
