# Status

Current branch: `codex/flowstar-scalar-affine-correctness-closure-20260804`, based
on the verified native-reproduction tip
`438ee68fd71fa6182eb66cac17229e20dd3cb7d3f`.  The native evidence run is
`outputs/native_reproduction_no_adapters/20260804T081205Z`; the scalar-affine
closure run is recorded separately under
`outputs/flowstar_scalar_affine_correctness_closure/`.

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
native reproduction.  The scalar-affine recheck stays in the separate diagnostic
registry and still finds Flow* empirical containment violations.

Xiangru's external PyTorch Taylor-model implementation is identified and audited
as the complete-Q3 implementation at clean `27d29050...`.  Its surrounding
NNCS/controller path remains outside this repository's plant-only numerical core.

Primary repository full pytest passes: 285 tests.  Xiangru exact-27d full pytest
executes with 111 passed, 4 skipped and one missing-historical-artifact failure;
the absent ignored `run.json` is not fabricated.  Registry validation and final
checksum verification are required before release/push; see the run manifest for
the final command records.
