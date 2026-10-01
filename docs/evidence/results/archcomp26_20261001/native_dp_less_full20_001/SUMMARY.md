# ARCH-COMP 2026 Double Pendulum less-robust — native full run

Run date: 2026-10-01. This is the **new** continuous-contract attempt, not the 2026-09-23 native timeout. No hash calculation or comparison was performed in this run or audit.

## Identity and completion

Remote run directory: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_less_full20_001` (abbreviated `R` below). The source evidence remains there: `R/START.json`, `R/RESULT.json`, `R/server.log`, `R/native.log`, `R/controller_rpc.jsonl`, and `R/ranges.bin`. On 2026-10-01, these six saved files were copied to this local directory without calculating a content digest. `COPY.json` records their exact remote and local paths, remote-listed and local byte sizes, and UTC copy times. The copied `ranges.bin` is 3,420,000 bytes; byte identity was not independently verified by a digest.

`START.json` records instance `double-pendulum-less-robust-continuous`, method `native`, contract label `archcomp26-dp-less-full20-all225`, start `2026-10-01T08:37:01.656825Z`, `CUDA_VISIBLE_DEVICES=1`, CPU affinity `6–9`, RPC port `5100`, and a 3600 s timeout. The native binary is the saved `suite_build/archcomp/double_pendulum_less_robust/matched_threads4`, which has `int steps = 20` in its adjacent source and requests port 5100. The separate `observed_server.py` listens on 5100; its model path resolves to the server's less-robust ONNX copy, previously compared directly byte-for-byte with the fixed official 2026 controller. The isolated Python RPC overlay is at `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_less_rpc_overlay_preflight_001/rpc_overlay`.

`RESULT.json` reports `status=completed`, exit code `0`, no timeout, end `2026-10-01T08:55:28.799700Z`, and full supervisor wall time **1107.127423 s** (including RPC startup and shutdown). `native.log` contains exactly `Step 0` through `Step 19`, `VERIFIED`, and solver-reported `time cost: 1100.101000` s. `server.log` has 20 successful POST responses; `controller_rpc.jsonl` has 20 requests, each with 225 four-state input boxes. The owned server and solver PIDs were `3052714` and `3052766`; both exited and port 5100 was released. The one-period 225-box plumbing smoke is a separate directory, `native_dp_less_smoke1_001`, and is not counted as a full benchmark result.

## Independent range-file scan

The read-only scanner used the saved `arch_ranges.h` record layout: little-endian `lane:uint64`, cumulative `step:uint64`, local interval length `h:float64`, then for each physical state four `float64` bounds in order `(tube_lo,tube_hi,endpoint_lo,endpoint_hi)`. That is `24+32×4=152` bytes per record. It found exactly **22,500 unique `(lane,step)` records**: lanes `0–224`, each with every step `1–100`; each step has 225 lanes, `h=0.01` throughout, and there were no duplicate, nonfinite, reversed, or endpoint-outside-tube intervals. This is 225 boxes × 20 control periods × 5 ODE steps per period. The scan found **zero** whole-tube coordinate records outside the official `[-1.7,2]` safety band, consistent with the author checker’s `VERIFIED` line.

Physical state order is `[θ₁, θ₂, θ̇₁, θ̇₂]`. Whole-time tube unions use **all 22,500 records**. The T=1 endpoint unions and per-box widths use **only cumulative step 100**, one interval from each of the 225 lanes:

| State | All-time tube union | T=1 endpoint union | T=1 union width | T=1 per-box width mean | T=1 per-box width max |
| --- | --- | --- | ---: | ---: | ---: |
| θ₁ | [0.9999996919513104, 1.9046282475863074] | [1.2931267868053968, 1.7771271322473556] | 0.4840003454419588 | 0.19436106403259815 | 0.32901199826617344 |
| θ₂ | [-0.12483376545118097, 1.3577873248942236] | [-0.12483376474059484, 0.44168221003359853] | 0.5665159747741934 | 0.19434017388097222 | 0.32931871755154307 |
| θ̇₁ | [-0.8405067547489368, 1.6969121773641698] | [-0.8405067511825203, -0.044472647141243334] | 0.796034104041277 | 0.23414675502031784 | 0.743550142361634 |
| θ̇₂ | [-1.618490164931325, 1.3000427612783] | [-1.618490164931325, -0.6019052691880047] | 1.0165848957433203 | 0.3158350652162677 | 1.0165848957433203 |

These widths measure saved interval enclosures, not sampled trajectories. The verdict is the historical CROWN-Reach/Flow* checker applied to the 2026-matched less-robust equations, full box, controller map, 0.05 s hold schedule, and continuous safety property. The separate saved-interval scan checks internal consistency and the safety band; it does **not** independently prove floating-point neural-network soundness or a complete end-to-end NNCS theorem. Method settings include 0.01 s fixed ODE substeps, Taylor order 4, and the author symbolic-remainder procedure. The first-period result and full-run timing must be reported separately.

## No-hash box projections

The independent `flowpipe_plot_nohash` entry parsed the local copy of all 22,500 records and wrote MATLAB `.m`, PNG, PDF, geometry JSON, and path/size/configuration `.render.json` files under `plots/` for:

- `dp_less_native_225x100_t_theta1_tube`: all 100 tube steps, one union-hull box per step;
- `dp_less_native_225x100_t_theta1_endpoint`: all 100 endpoint steps, one union-hull box per step;
- `dp_less_native_225x100_theta1_theta2_endpoint`: six displayed steps (`1,5,25,50,75,100`) with 225 separate lane boxes each; all 100 steps were still scanned for record coverage.

Each geometry names the remote origin and local source paths, copies the local `START.json` as an explicitly unbound run-configuration declaration, and records adjacent `RESULT.json` status as **unbound** run evidence. The initial box is `[1,1.3]^4`; the plotted Safe region is `[-1.7,2]^4` for **all continuous time `t∈[0,1]`**, not an endpoint target. The geometry marks `accepted_lanes=null`: `ranges.bin` has no accepted/status field, so `complete=true` means 225 records were found at a step, not solver acceptance or certification. The plots are axis-aligned box projections, not Flow* octagons. MATLAB scripts were generated but not executed.
