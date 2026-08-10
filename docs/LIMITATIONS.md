# Limitations

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
- Flow* observer replays are diagnostic counterfactuals, not production
  dependencies or modified native results.
- Endpoint, last-segment tube, and full-horizon/prefix tube are kept separate.
  Missing stock objects are `UNAVAILABLE`, never fabricated.
- TORA complete-Q3 is frozen historical stress-test evidence. No TORA-specific
  controller, formula, or support enters the generic source modules.
