# Status

Current branch: `codex/flowstar-scalar-affine-correctness-closure-20260804`, based
on the verified native-reproduction tip
`438ee68fd71fa6182eb66cac17229e20dd3cb7d3`.  The launch document appended an
extra `f` to this 40-character remote tip; the resolved ancestry is recorded in
the closure start state.  The native evidence run is
`outputs/native_reproduction_no_adapters/20260804T081205Z`; the scalar-affine
closure run is
`outputs/flowstar_scalar_affine_correctness_closure/20260804T131445Z`.

The native reproduction phase is complete:

- Xiangru CROWN-Reach/Flow* B12 TORA is `reproduced_exact` at T=20;
- Xiangru complete-Q3 B48 is `reproduced_with_declared_tolerance` at T=20;
- Xiangru DiffReach CPU failed behavior is byte-exact; its GPU path has an
  `environment_failed` cuDNN float64 convolution backend error before step one on
  the available V100, not a native algorithm rejection;
- historical B12/B24 Q3 references remain `source_identity_unknown` because the
  saved source was dirty without a patch;
- clean stock Flow* official VDP and upstream DiffReach official VDP complete T=10,
  with no upstream raw reference artifact;
- our exact order-4 Torch command does not reach T=10 and does not reproduce the
  prior natural failure boundary: the two adaptive lanes hit their 300-second wall
  caps at T=5.904687 and T=6.049038, classified `runtime_timeout` rather than a
  mathematical solver rejection.

No adapter, rewritten ODE, generated harness or endpoint repair is counted as a
native reproduction. The clean-stock scalar-affine diagnosis stays in the
separate diagnostic registry. Its generated observer first loses strict analytic
containment at accepted remainder refinement 2 in unmodified
`Continuous.cpp:1013-1029`; the official public-API route also under-encloses at
its accepted right time. Outcome F is complete, but the Flow* correctness gate is
open and primary comparison eligibility remains false. See
[`FLOWSTAR_SCALAR_AFFINE_CORRECTNESS_CLOSURE.md`](FLOWSTAR_SCALAR_AFFINE_CORRECTNESS_CLOSURE.md).

Xiangru's external PyTorch Taylor-model implementation is identified and audited
as the complete-Q3 implementation at clean `27d29050...`.  Its surrounding
NNCS/controller path remains outside this repository's plant-only numerical core.

The pre-change baseline was `283 passed, 2 skipped`; final validation is
`299 passed, 2 skipped`. The two skips are optional external backend tests whose
`FLOWSTAR_ROOT` and `DIFFREACH_ROOT` variables are deliberately not configured;
the clean GCC 11 scalar binaries were compiled and run separately. Xiangru
exact-27d full pytest
executes with 111 passed, 4 skipped and one missing-historical-artifact failure;
the absent ignored `run.json` is not fabricated. Registry validation passes for
11 native and two diagnostic rows, all 401 prior native-run checksums verify, and
the scalar closure manifest verifies; see the run evidence for command records.
