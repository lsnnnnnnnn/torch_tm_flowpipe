# ARCH-COMP 2026 Double Pendulum less-robust: P3 DP port attempt

Date: 2026-10-01. This records a **new GPU diagnostic attempt**, including its failure. It does not fill a successful 2026 four-method result cell. No content hash, SHA-256, CUDA extension build, or `git rev-parse` was used.

A subsequent tighter, independently directed affine-ReLU attempt is recorded in [ARCHCOMP26_DP_P3_AFFINE_DIAGNOSTIC_20261001.md](ARCHCOMP26_DP_P3_AFFINE_DIAGNOSTIC_20261001.md). The failed interval-forward attempt below remains preserved.

## Identity and contract

The common benchmark contract is [ARCHCOMP26_DOUBLE_PENDULUM_LESS_CONTRACT_20261001.md](ARCHCOMP26_DOUBLE_PENDULUM_LESS_CONTRACT_20261001.md). The run uses the continuous four-state ODE, fixed less-robust ONNX, full `[1,1.3]^4` physical initial box partitioned `5×5×3×3` into 225 boxes, 20 controller periods of 0.05 s, five 0.01 s plant substeps per period, and whole-tube safety `[-1.7,2]^4`. The server model path under `CROWN-Reach/ARCH-COMP2024` had been compared directly byte-for-byte with the fixed official 2026 less-robust ONNX in the contract audit. This attempt did not repeat that comparison.

The numerical source is the 2026-09-30 P3 7-variable DP port with work order 3, point order 2, validation order 4 (`solution_plus_one`), geometric reciprocal five-route Python/CUDA candidates, strict endpoint and control injection, trig direct reuse, and full symbolic remainder. The 2026-09-30 QUAD 1024-lane qualification does **not** transfer to DP. The shared Xiangru `crown_reach.py` driver was used. Its controller layout is `native`, relaxation `same-slope`, and transport `native-f64`; the historical Huan/Xiangru DP diagnostics used `rpc-float32`. The P3 core lacks the shared driver's `COMPOSE_PARENT_ASSEMBLY` *reporting* attribute, so this port records that metrics field as `null`, without changing the numerical dispatch.

Local new sources: [`tools/archcomp26_dp_p3_nohash.py`](../tools/archcomp26_dp_p3_nohash.py), [`tools/archcomp26_dp_interval_controller.py`](../tools/archcomp26_dp_interval_controller.py), and [`tools/archcomp26_dp_controller_preflight.py`](../tools/archcomp26_dp_controller_preflight.py). They reside in the local branch workspace and an isolated server directory; no frozen engine source was edited. Exact runner/source copies were placed inside each completed remote run directory before later edits. The run directory root is:

`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_p3_port_v1/`

The server execution used physical `CUDA_VISIBLE_DEVICES=3`, CPU affinity `14–17`, existing prebuilt CUDA `.so` files, and an outer [no-hash runner](../tools/run_archcomp26_nohash.py) that captured commands, selected environment, logs, exit code, timeout, and wall time in new directories. The DP runner makes SHA-256 and CUDA `load_inline` calls fail immediately if accidentally reached. No old experiment directory was overwritten or relaunched.

## Controller certificate issue and independent enclosure

The shared driver reads auto_LiRPA's binary64 `lbias` and `ubias` directly, with no outward rounding. On the first DP box's actual initial sparse hull, its first output has `L=-0.4004669243588208`, `U=-0.4004669243588209`: `U−L=-1.1102230246251565e-16`. The hull is `[0.9999999999999999,1.0600000000000003]` in the first two variables and `[0.9999999999999999,1.1000000000000003]` in the next two. The same inversion occurs with batch sizes 1 and 225, so it is not a single-batch-shape issue. Two CROWN bias entries are inverted over all 225 initial boxes. The strict injection correctly rejected these original bounds in `smoke1_001` and `smoke1_002`.

For a separate diagnostic, `archcomp26_dp_interval_controller.py` verifies the fixed ONNX graph is exactly `MatMul→Add→ReLU→MatMul→Add→ReLU→MatMul→Add`, with weight shapes `4×25`, `25×25`, `25×2` and float32 initializers. Each stored float32 number is exactly representable in float64. The evaluator follows that graph using the pre-existing `flowstar_gpu.interval` directed dot/add/sub primitives on binary64 values. For the same CROWN linear map `T`, it computes an independent enclosure of the exact graph function minus the exact stored map `T·x`; those directed residual endpoints replace the unrounded auto_LiRPA bias endpoints. This is a local controller enclosure. It does not by itself prove the entire DP flowpipe or all surrounding floating-point operations end to end.

Controller-only preflights under `controller_preflight_001` (raw boxes), `_002` (actual initial sparse hull), and `_003` (independent directed residual) did no plant integration. `_003` found all directed intervals finite and ordered for B=1 and B=225. At the first box its two residual widths are `0.7912828366079631` and `0.8752500389718019`; over all 225 boxes, maximum width is `0.9277244972312`. Deterministic lower/mid/upper point checks using 100-digit Decimal arithmetic found all 6 B=1 and 24 selected B=225 output residual values inside the interval. These samples are checks, not an exhaustive proof of the interval implementation.

## Observed attempts

| Server directory | Scope | Observed result |
| --- | --- | --- |
| `smoke1_001`, `smoke1_002` | 1 box × 1 period with original CROWN bias | Fail closed at first control injection; second attempt identifies `L>U` in output 0. No plant substep ran. |
| `smoke1_003` | 1 box × 1 period with directed controller residual | All 5 plant substeps accepted and inside the safety box. Driver then failed while writing metrics because the P3 core lacks a reporting-only field. |
| `smoke1_004`, `smoke1_005` | 1 box × 1 period; metrics compatibility and corrected coupling metric | Both complete, 5/5 substeps accepted and their four-state tubes inside. The five JSONL observations are byte-identical between the two attempts. `smoke1_005` reports the actual directed residual coupling width. |
| `smoke225_001` | 225 boxes × 1 period | Complete: 1125/1125 lane-substeps accepted; all observed four-state tubes inside the safety box; `broken=0`. Driver time 3.529 s; outer wall 7.438 s including startup and JSONL export. Peak CUDA allocated 183,675,392 B, reserved 933,232,640 B. |
| `full225_001` | **Formal new diagnostic attempt:** 225 boxes × 20 periods | **Incomplete.** Period 1 accepts all 225 boxes for 5 substeps. At period 2's first substep, all 225 lanes become `FAILED_CONTRACTION` (status 1); 0 are accepted. The driver then ends with exit code 0, but the experiment result explicitly records `status=incomplete`, 1125 accepted lane-substeps against 22,500 required, 225 broken branches, and no valid 20-period verdict. Driver time 11.866 s; outer wall 15.764 s. Peak CUDA allocated 183,675,392 B, reserved 1,029,701,632 B. |

The failed sixth-step `tube`/`endpoint` fields in `full225_001/data/observations.jsonl` are the **frozen last accepted state**; their inclusion inside the safe box says nothing about the rejected new time interval. The `accepted=false` and `status=1` fields identify invalid per-step enclosures. The local runner was subsequently clarified to tag `tube_endpoint_valid` and require acceptance in per-step safety counts for any future attempt; the archived run source and raw observations remain unchanged.

## Why the full attempt stopped

The independent interval controller enclosure is much wider than the original CROWN affine bias. In `full225_001`, directed control residual widths at period 1 averaged `[0.78824,0.86028]` and had maxima `[0.84799,0.92772]`. At period 2 they averaged `[5.61296,7.14621]`, with maxima `[6.11939,7.88298]`; the injected remainder maxima were about `[6.12118,7.88515]`. The first post-injection plant step then failed the P3 self-map contraction check for every lane, even with the configured eight self-map retries and recentering. The authoritative P3 status enum defines 1 as `FAILED_CONTRACTION`, not a safety counterexample or a division-domain failure. No exact per-component subset margin was saved, so the precise numerical contraction threshold cannot be inferred from these logs.

For scale only, the **old**, separate 225-box Huan/Xiangru diagnostics recorded mean original same-slope CROWN bias widths around `[0.000920,0.006064]` at period 1 and `[0.004451,0.041876]` at period 2. These use `rpc-float32` and a different plant state after period 1; the second-period numbers are not a matched-state comparison. The first-period comparison shows the interval fallback loses two to three orders of magnitude of controller correlation. A one-ulp `min/max` repair of original CROWN bias would be tight but cannot certify its unbounded floating-point error; it was not used.

## Remaining concrete work

A full result needs a tight **verified affine** controller residual for `NN(x)−T·x`, retaining ReLU correlations and accounting outwardly for slopes, intercepts, matrix products, and bias sums. The current interval forward evaluator is mathematically conservative but too wide for period 2. Reusing auto_LiRPA's unrounded bias with one-ulp padding has no established rounding-error bound. The subsequent affine diagnostic linked above implemented that controller-local direction and made a new 225-box attempt. No validation cap, box count, or horizon was relaxed in this interval-forward attempt.

The unsuccessful `full225_001` is retained as the actual new P3 DP attempt; it must not be reported as a successful 2026 less-robust row or blended with the old `suite_v1/ours` result (which uses the older `engine_linear_leaf_v2`).
