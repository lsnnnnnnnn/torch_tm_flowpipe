# Flowpipe plotting validation — 2026-10-01

The exporter was validated only against already-saved artifacts; no solver or
GPU experiment was started. The local artifact directory is:

`/Users/shengenli/Documents/ChatGPT/verification/output/flowpipe_plot_validation_20260930/v5_affine_20261001`

The machine-readable receipt is
`docs/evidence/flowpipe_plot_validation_20261001.json`; it binds the exact
generator/verifier source, raw-source manifests, commands, environment, and
all output hashes.

## Results

- Focused unit/contract suite: 26 tests passed (23 plotting tests and 3
  ARCH-COMP plot/source-audit contract tests). The plotting suite includes the
  canonical state-state route for time-scoped regions, rejects boolean
  `expected_steps`, and independently recomputes ACC's outward affine interval
  image while rejecting altered transforms or raw boxes.
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
  the endpoint-only target was drawn at `t=5` with its time scope in the
  legend, the unprojectable target was disclosed on the state-state plot, and
  native acceptance was labelled unknown.
- MATLAB/Octave executables were not available. `.m` generation and static
  contract tests passed, but this is not a MATLAB runtime claim.
- The five saved artifact sets below remain archived QUAD projections. ACC v3
  has no identity-bound numerical series yet, so its affine-threshold support
  is test-verified but is not presented as an ACC execution result.

## Artifact SHA-256

Each row lists geometry, MATLAB, PNG, and PDF hashes in that order.

| Artifact stem | Geometry | MATLAB | PNG | PDF |
|---|---|---|---|---|
| `native_root1_b2_t_x3_tube` | `c2d39e4d44104eeb6423512b5a13a8b6393b9737a7d8e412cea870a5c22702df` | `b1516249d9573bc471f3962c5bdfd4bf3f72c207462b68c185500926ed3f11d2` | `5b3986f907ac30f94af4f727bd947bea7849389b9347a439280d36d796a56fad` | `c98f222ba65a614f738e1b4927a3624aab443e0fe898a1f9bed155bd1f06a4fd` |
| `p2_40_t_x3_tube` | `a0b11b58baa8540f03afa0911d8e730eff34bb88bb27144f52a06c9a9196f5b9` | `1f1d8123d8e34475a0280dcd77828b7a978ef573646d5f1c5037e8d20b9dd7c5` | `8a1134f07d49425757019bd493d153546273056147b8b7972dae09aad9ce29ce` | `5eda9191715d031bd2b8cc8c4487e5902f60587c9e59bf839716ecb0735d26d5` |
| `p2_x1_x2_endpoint` | `3d76f212d12fabfb0f2a479546a0d3dbcb33f35b54d234a3b1d28a655a01d6c1` | `f0398c0b6de318c461952259b9eff435c3c3e3ffa77bd2cb4ae7f4e58d5477c6` | `dec69a1fb12aaea36bab073bb76a9ac859ddc6a2259c27d64d6a350007ba3903` | `b8e92583c961229d99809a677a00155595d4ef27e8e2b41993b5d73e8aa0abfc` |
| `p3_sparse_t_x3_endpoint` | `9dcbd35a72d1fdddd5b141d2fa4d3b926240ce2fab6cbd705c63557b636d8e89` | `3fff47bcb38983849372074301919e02e748044c3aaa6228f5f7727e0b345c00` | `5d21b760306a6cadf326472bf301246b703f6bb158f8d90b1566dac7c665de68` | `8ec08b76fe9a7f7c75ff48dcece1e329b2d086bef41db46795a618baa5b6fd41` |
| `p3_sparse_x1_x2_endpoint` | `8b6eeaf268b5e1eee1de78b64f109c90e454bd43198bcab763e369b771ea70e7` | `c11663ab910fa92c279643d27220567d1d8cb3b34f5528e4f318df1217ba05a6` | `972517fd0e267e7333a6226c442a89a2f1e161a1ec60fa9b765a72b7d78636b3` | `ac2059daeba920f016c1e22da3539a8c703532793f7dd23992b7a23fb5e593cf` |

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
- Raw state projections remain axis-aligned boxes. V3 can map those boxes to
  one explicitly declared affine interval coordinate, but it still cannot
  recover source-coordinate correlation or preserve Flow* octagonal support
  directions; native-octagon parity remains an unfinished Stage B extension.
