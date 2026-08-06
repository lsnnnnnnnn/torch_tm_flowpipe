# Xiangru complete-Q3 baseline reproduction

## Outcome

The frozen homogeneous-TORA `b48_static / complete_q3` baseline was reproduced from an unmodified, clean detached worktree at Xiangru commit `27d29050a5f214b56f211ca9cb411e734ed80230`. The fresh cell is `VERIFIED`, completes all 200 fixed 0.1 s segments to certified horizon 20.0, has 48 accepted leaves at every recorded segment, no first failure, and no retry segments.

Against the committed author reference, all selected semantic/top-level fields are exactly equal. Across 2,850 non-runtime numeric fields the largest absolute difference is `1.1546319456101628e-13`; there are no differences outside the upstream-declared `1e-6` tolerance. This is classified as **numerically equivalent within documented tolerance**, not byte-exact, because floating-point fields and native timings differ.

## Frozen run

| Item | Value |
|---|---|
| Xiangru source | `27d29050a5f214b56f211ca9cb411e734ed80230`, clean detached worktree |
| DiffReach source | `dd628eb443b517d6415de93e7035b4baef73963e` |
| Method | `complete_q3`, dynamic compiled tensor engine |
| Benchmark | homogeneous TORA closed-loop NNCS, B48 static initial partition |
| Controller | original homogeneous ReLU TORA ONNX SHA256 `52a50c6b...`; transformed input model SHA256 `bb80479c...` |
| Initial set | `[.6,.7] × [-.7,-.6] × [-.4,-.3] × [.5,.6]`, held `u1=[0,0]` |
| Step/control | fixed `h=0.1`; controller update every ten segments (1 s) |
| Horizon | 200 segments, 20 s |
| Arithmetic | PyTorch float64, CUDA GPU 0 (V100 16 GB) |
| Validation | two polynomial Picard rounds; seed 0.01; ten DR-RP remainder rounds; full-tube property |
| Process result | exit 0; cold process wall 151.02014847565442 s |

The top-level status `PARTIAL_VALIDATION` is the upstream matrix-scope label: this command intentionally selects one cell. It is not a partial-horizon result. The selected cell itself is `VERIFIED` at 20.0 s, and CPU/CUDA replay plus source/config validity checks pass.

## Timing scope

The fresh upstream timing reports compile/warm separately excluded (`141.06641076132655` s), controller total (`1.3335942178964615` s, including initial controller `0.7779296152293682` s), plant dynamics (`1.0712888417765498` s), validation separately excluded (`1.923620163463056` s), solver excluding validation (`2.689615928567946` s), and total including validation (`4.613236092031002` s). Environment/setup, model loading, and serialization are not separately available and are not inferred from the total.

These are native closed-loop GPU timings from one required exact reproduction. They are not a repeated formal performance benchmark and are not compared with Torch plant-only CPU timing.

## Evidence

- Entrypoint and call chain: `outputs/xiangru_q3_matched_audit_20260806/xiangru_reproduction/q3_entrypoint_map.md` and `q3_source_callgraph.md`.
- Exact command/config/controller inventory: `original_commands.sh`, `original_config_snapshot/`, and `original_artifact_inventory.json`.
- Raw run streams, exit code, resource usage, environment, config and source hashes: `fresh_q3_b48_t20/`.
- Author/fresh summary and strict diff: `reproduction_summary.csv` and `reproduction_diff.json`.

No Xiangru algorithm or source file was modified. The isolated reproduction worktree remains clean.
