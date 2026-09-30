# Flowpipe plotting validation — 2026-10-01

The exporter was validated only against already-saved artifacts; no solver or
GPU experiment was started. The local artifact directory is:

`/Users/shengenli/Documents/ChatGPT/verification/output/flowpipe_plot_validation_20260930/v2_20261001`

The machine-readable receipt is
`docs/evidence/flowpipe_plot_validation_20261001.json`; it binds the exact
generator/verifier source, raw-source manifests, commands, environment, and
all output hashes.

## Results

- Unit/contract suite: 23 tests passed (19 plotting tests and 4 ARCH-COMP
  manifest/source-inventory contract tests).
- Independent raw-to-geometry check: PASS for all displayed frames.
  - native root1-B2: 1000 frames reconstructed from 2000 binary records;
  - P2 time-state: 40 frames/40 union-hull boxes;
  - P2 state-state: 3 frames/3072 lane-preserving boxes;
  - P3 sparse time-state: 8 frames/8 union-hull boxes;
  - P3 sparse state-state: 8 frames/8192 lane-preserving boxes.
- Each `.render.json` receipt re-hashed its geometry, MATLAB, PNG, and PDF
  artifacts successfully.
- Visual inspection covered the P3 `t,x3` endpoint plot, P3 `x1,x2` endpoint
  plot, and native root1-B2 `t,x3` tube plot. Their footers remained visible,
  the endpoint-only target was drawn at `t=5`, the unprojectable target was
  disclosed on the state-state plot, and native acceptance was labelled
  unknown.
- MATLAB/Octave executables were not available. `.m` generation and static
  contract tests passed, but this is not a MATLAB runtime claim.

## Artifact SHA-256

Each row lists geometry, MATLAB, PNG, and PDF hashes in that order.

| Artifact stem | Geometry | MATLAB | PNG | PDF |
|---|---|---|---|---|
| `native_root1_b2_t_x3_tube` | `c2d39e4d44104eeb6423512b5a13a8b6393b9737a7d8e412cea870a5c22702df` | `46cc565be9ca8e6ced3727c246ad0affb651038c6206ec8bf86c39cecd77bcfb` | `8dfac6d2acff3157156d021b2b83af84f290aab7f3dc9724ed52d817ae797bf0` | `59931020b4f98684180f533c557608b97340386d9caea507ec0c8de40b6a064b` |
| `p2_40_t_x3_tube` | `a0b11b58baa8540f03afa0911d8e730eff34bb88bb27144f52a06c9a9196f5b9` | `1f1d8123d8e34475a0280dcd77828b7a978ef573646d5f1c5037e8d20b9dd7c5` | `8a1134f07d49425757019bd493d153546273056147b8b7972dae09aad9ce29ce` | `477dc1276827d7aad676b8611d4ce21206cddb568e16b4b5ed497c8ab013fd29` |
| `p2_x1_x2_endpoint` | `3d76f212d12fabfb0f2a479546a0d3dbcb33f35b54d234a3b1d28a655a01d6c1` | `f0398c0b6de318c461952259b9eff435c3c3e3ffa77bd2cb4ae7f4e58d5477c6` | `dec69a1fb12aaea36bab073bb76a9ac859ddc6a2259c27d64d6a350007ba3903` | `ca3f8e16de7adcfcead699876dc03f603a8e796246634cc0114e3692abfd861c` |
| `p3_sparse_t_x3_endpoint` | `9dcbd35a72d1fdddd5b141d2fa4d3b926240ce2fab6cbd705c63557b636d8e89` | `7517afefb688968d0b11265579299ab92c66cf3d9c202b0633affe0396c94ef4` | `d0315fd62dedbd6539cffc81f0ce6997347662318409abcf300add6ef7abae58` | `e6f9dde94bd0adeeac7f7bf5ac1d7f19682d743e93922a35d3817d1578cf0586` |
| `p3_sparse_x1_x2_endpoint` | `8b6eeaf268b5e1eee1de78b64f109c90e454bd43198bcab763e369b771ea70e7` | `2b56999b43b20bb87a5f24691924d8653ed3a64e8177adb79e10d27364709641` | `b0a884fee21d3c96f3bc3262bb54d1725c00a705097061e55cfebd155760c61f` | `a62316c1fff439308c453169f8f8bf2187250b0453bfc8729ef79f80738652f8` |

## Evidence interpretation

- P2 has all 40 saved observers, constant 1024 lanes, identity-bound
  sidecars/INPUT/RESULT, and a matched `h=0.005`, 40-step contract. It is still
  only the explicitly bounded 40-step prefix and has
  `fullbatch_qualification=false`.
- Every observed P2 and P3 `.pt` file, including display-omitted files, has a
  persisted hash entry. P2's per-field step-size/horizon/lane declarations are
  identity-bound; P3's adjacent declarations remain explicitly unbound.
- P3 has only eight saved projection snapshots. The adjacent RESULT reports a
  completed 1000-step/all-lane run, but the observer files have no identity
  sidecars; plots therefore label that RESULT as adjacent and **unbound**.
  Projection-unobserved 992/1000 is not interpreted as solver failure.
  Its adjacent `steps.jsonl` is likewise unbound because RESULT does not hash
  it; the render receipt does not promote its `advance_s` sum to a bound time.
- Native root1-B2 has all 1000 range-record steps and its INPUT numerical
  contract matches, but `ranges.bin` is not hash-bound to the adjacent
  RESULT. The RESULT is labelled unbound, and the geometry says only that two
  records exist per step. It does not claim lane acceptance or full-1024
  completion; `original_full1024_completed=false` remains visible.
- The current implementation is an axis-aligned box projection. It does not
  preserve Flow* octagonal direction correlation; native-octagon parity is a
  remaining Stage B extension rather than a completed claim.
