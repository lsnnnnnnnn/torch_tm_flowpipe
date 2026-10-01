# ARCH-COMP26 Double Pendulum less — new Huan / Xiangru runs

Date: 2026-10-01. These are new, separate attempts under the selected continuous 2026 contract: the official less-robust controller, `[1,1.3]^4` split into 225 boxes, 20 held-control periods of 0.05 s, five ODE substeps per period, and whole-time safety `[-1.7,2]^4`. The existing 2024-path ONNX was directly compared byte-for-byte with the fixed 2026 file in the contract audit; no content digest was calculated here. The old suite jobs were not restarted.

## Execution and coverage

| Method | New remote directory under `runs/archcomp26_20261001/author_dp_less_v1/` | Process wall | Driver plus observer | Core time | Coverage | Saved tube / endpoint within safe band |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Huan | `huan_full20_001` | 9.539174 s | 8.669310 s | 5.277640 s | 225 × 100 accepted | yes / yes |
| Xiangru | `xiangru_full20_001` | 8.387790 s | 7.488113 s | 4.095443 s | 225 × 100 accepted | yes / yes |

Both use the shared Xiangru controller driver with each author's plant engine, strict plant mode, **Taylor order 4** from the config, box/same-slope CROWN, official native input layout, RPC-float32 transfer, 0.01 s ODE substeps, GPU2, and CPU affinity 10–13. The raw `payload/START.json` from these first attempts accidentally calls the method “P2”; this refers to neither the actual config nor the metrics. Both record `ode_order: 4`, and the launcher source has since corrected the descriptive label. The raw files were preserved.

The first Huan smoke directory `huan_smoke1_001` stopped before numerical work because the historical split grid's last decimal edge is one binary64 step above the literal `1.3`. The check was corrected to accept that covering edge in a new directory `huan_smoke1_002`; Xiangru's `xiangru_smoke1_001` also completed. Each successful smoke had 225 × 5 accepted box-steps and an in-band saved tube. Neither smoke is a full result.

The full `ranges.bin` files each contain 22,500 records, one for every lane `0–224` and step `1–100`, with finite ordered intervals and h=0.01. The two files are directly byte-equal. This is unsurprising because the engines share substantial numerical source and the same controller driver; it is **not** independent proof. Each method has one process-time sample, so these times do not establish a stable speed ranking. The Huan/Xiangru NN injection still uses unqualified round-to-nearest coefficient arithmetic and CROWN floating bias; strict plant mode does not give an end-to-end floating-point NNCS certificate.

## Absolute interval results at T=1

State order is `(θ₁, θ₂, θ̇₁, θ̇₂)`. Huan and Xiangru have identical saved values:

| State | Endpoint union | Union width | Per-box width mean | Per-box width maximum | All-time tube union |
| --- | --- | ---: | ---: | ---: | --- |
| θ₁ | [1.2907643185625681, 1.7767949692884832] | 0.4860306507259151 | 0.19294536891664482 | 0.3338059300069445 | [0.9994352468167139, 1.9043182183289598] |
| θ₂ | [-0.1242823143233622, 0.4444010607820787] | 0.5686833751054409 | 0.19291889956967087 | 0.3376921859300584 | [-0.12429717252964027, 1.3579187620502111] |
| θ̇₁ | [-0.8653775761581333, -0.012398204543126833] | 0.8529793716150065 | 0.23721578744999255 | 0.807804620928524 | [-0.8654135741639509, 1.7008731991207624] |
| θ̇₂ | [-1.6639905719267576, -0.5542026971342128] | 1.109787874792545 | 0.31715285944167376 | 1.109787874792545 | [-1.667565096901283, 1.3187574886455504] |

The independent [range scanner](../../../../../tools/archcomp26_scan_dp_ranges_nohash.py) produces each run's `INDEPENDENT_INTERVAL_SCAN.json`. It found 2,927 component records with endpoint lower below the same-step tube lower, and 368 with endpoint upper above the tube upper. The largest discrepancy is `5.551115123125783e-15`. The cause of this small observer mismatch has not been isolated; do not claim exact endpoint-in-tube containment for these two runs. Both tube and endpoint intervals independently lie in the safety band, with far larger margins. The native DP less saved ranges have zero such mismatches under the same scanner.

Original supervisor `START.json`/`RESULT.json`, stdout/stderr, child configuration, metrics, result and range records are in each local run subdirectory. The remote originals remain in place. The [new no-hash launcher](../../../../../tools/archcomp26_dp_author_nohash.py) and supervisor record paths, sizes, environment and timing without calculating a content digest.
