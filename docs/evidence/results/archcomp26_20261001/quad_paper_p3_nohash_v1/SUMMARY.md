# ARCH-COMP 2026 paper QUAD — latest P3 diagnostic

This is a new run of the user-selected **2026 paper equations**, using the fixed official Torch ONNX saved on the server. It does not reuse or restart the historical author-contract P3 job. The source contract is `quad_paper.yaml`; the isolated executable and its no-digest adapters are saved beside this summary. The remote evidence root is `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/quad_paper_p3_nohash_v1`.

## Numerical route and scope

- 16 variables; physical initial set `[-0.4,0.4]^6 × {0}^6`, partition `8×8×8×2×1×1 = 1024`; 50 control periods of 0.1 s, 20 ODE substeps of 0.005 s per period. The runner checks the paper x2/x4/x5 expressions before every stage. Endpoint property: `0.94 ≤ x3(T=5) ≤ 1.06`.
- Shared Xiangru CROWN-Reach GPU driver, strict sparse mode; current P3 working order, P2 ODE point order, P4 validation; geometric reciprocal, strict endpoint and control injection, host K20 full symbolic remainder, trig direct reuse, chunk128 weighted/valid validation and working graph cleanup. The original numerical engine and strict modules are loaded; copies of metadata/graph adapters remove only digest identity gates. The `START.json` records the selected source modules and precompiled CUDA `.so` paths, sizes and modification times.
- Physical GPU3, CPUs 14–17. Eight existing CUDA shared objects were preloaded; CUDA extension JIT and SHA-256 calls were guarded to raise. No content digest or checksum comparison was performed in this attempt. The controller saved copy had already been directly byte-compared with the fixed official 2026 Torch ONNX in the separate contract audit.
- **Qualification:** This is a numerical diagnostic of the latest P3 route. `--strict` names the engine mode and the driver printed `VERIFIED` at the terminal checker. `end_to_end_strict_certificate=false` is recorded explicitly because the complete floating-point CROWN/controller and integrated NNCS proof chain was not independently established by this run. The saved binary identities were not reverified under the user's no-hash instruction.

## Executed stages

| Directory | Coverage | Outcome | Outer wall | Peak CUDA reserved |
| --- | ---: | --- | ---: | ---: |
| `smoke1_001` | 1 box, 0 steps | Failed before first substep: runner bound the Horner export at the wrong Python attribute; traceback retained. | 4.477 s | 121,634,816 B |
| `smoke1_002` | 1 box × 20 steps | 20/20 accepted; P3/validation4 and trig route exercised. | 13.552 s | 1,061,158,912 B |
| `batch1_001` | 1024 boxes × 20 steps | 20,480/20,480 lane-steps accepted; weighted/valid and graph maintenance counters active. | 36.064 s | 9,292,480,512 B |
| `batch2_001` | 1024 boxes × 40 steps | 40,960/40,960 lane-steps accepted; control refreshed at steps 0 and 20. | 60.384 s | 9,745,465,344 B |
| `full50_001` | 1024 boxes × 1000 steps | **1,024,000/1,024,000** lane-steps accepted; `broken=0`; driver `VERIFIED`. | **1,357.555 s** | **12,138,315,776 B** |

The first smoke failure was a runner plumbing error. The corrected runner was uploaded before `smoke1_002`; the failed directory was preserved. Short-stage `FALSIFIED` lines reflect the terminal T=5 checker applied at T=0.1 or T=0.2 and are not full-horizon property results.

## Full-run evidence

- `full50_001/RESULT.json`: outer exit 0, no timeout, wall 1,357.5549509264529 s. `full50_001/data/RESULT.json`: driver exit 0, elapsed 1,350.5255287094042 s, 50 NN refreshes, 1000/1000 accepted substeps, 1,024,000/1,024,000 accepted lane-steps, `metrics_broken=0`, peak CUDA allocated 8,273,138,688 B, reserved 12,138,315,776 B, peak RSS 5,936,856 kB.
- Terminal hull at T=5: `x3 ∈ [0.9584732146312492, 1.0256989633476477]` (width 0.06722574871639841), within `[0.94,1.06]`. The raw driver stdout says `VERIFIED`; all terminal variables are in `data/metrics.json`.
- `data/observations.jsonl` contains exactly 1000 consecutively numbered records. Every record reports 1024 accepted lanes, status 0 for all lanes and no rejected lanes. Each also records separate tube and endpoint bounds, each pooled across the 1024 boxes, for all 12 physical states at that substep. Across the 1000 tubes, x3 ranges from −0.40932024282160917 to 1.4526780123354797; the property above concerns only the T=5 endpoint. The last observed SR length is 1000; end-of-run cleanup resets length to 0 and advances epoch to 1000.
- Weighted chunk128: 1000 refinements, 16,000 graph-map calls, 2,048,000 target rows, zero padded rows. Valid chunk128: 1000 calls, 8000 blocks. These are execution counters, not a separate numerical proof.

For comparison, the separately executed new Huan and Xiangru P2 parity paper-contract runs both ended with x3 `[0.967434441417146, 1.015176258384569]` and 0 broken lanes. The P3 diagnostic terminal interval is wider. These are individual runs on different GPU devices and routes, not a controlled speed comparison.
