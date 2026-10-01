# ARCH-COMP 2026 Double Pendulum more-robust — native full attempt

Run date: 2026-10-01. This is a **new** continuous, all-box attempt. The saved historical more-robust entry used a single corner and the less-robust controller; it is not this run. No checksum or content digest was calculated.

## Contract and run identity

- Remote original: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_more_full20_001`.
- This directory holds copies of `START.json`, `RESULT.json`, native/server/RPC logs, `ranges.bin`, and port/PID evidence. `COPY.json` records the source paths and local byte sizes. The remote originals remain in place.
- Official fixed 2026 controller: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_more_prep_001/controller_double_pendulum_more_robust.onnx`, with float32 `[N,4] → [N,2]` graph and no graph preprocessing. The separate server reads this exact path from recorded `DP_MODEL`; it does not select the historical less controller.
- New source/build: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_more_build_001`. Local `../native_dp_more_build_001/` holds source patches, server script, build receipts, and the 225-box ledger. Its physical ODE is the directly checked official `dynamics_dp.m`; sample-and-hold control uses 4 state inputs and outputs `(T1,T2)`.
- Full initial box `[1,1.3]^4`, partitioned `5×5×3×3=225`. The new binary exported `initial_boxes.json` with those 225 seven-variable boxes, in the same numeric order as the saved less-robust all-box ledger. The last three entries are the auxiliary clock and held controls, initialized to zero.
- Native method settings: fixed ODE step `0.005 s`, Taylor order 4, cutoff `1e-6`, symbolic-remainder queue 1000. Intended control period `0.02 s × 20 = 0.4 s`. Each continuous tube is checked against `[-1.5,1.5]^4`.
- GPU 1, CPUs 6–9. RPC server PID `3086695` owned `127.0.0.1:5100` at start; solver PID `3086750` used the matching 5100 binary. An isolated overlay supplied the cached RPC Python packages.

## Outcome

`RESULT.json` says the process **completed** with exit code 0, no timeout: start `2026-10-01T10:00:27.147262+00:00`, end `10:12:23.141013+00:00`, supervisor wall `715.978257 s`. Solver-reported time is `710.229 s`.

The solver attempted control periods `Step 0` through `Step 15`, then printed 45 `Unknown.` lines and final **`UNKNOWN`**. It did **not** complete periods 16–19. The process-completion status is not a safety verdict. There are 16 successful RPC calls, each with 225 four-state input boxes and 225 coefficient responses. The `UNKNOWN` rows cannot be entered as a full-horizon verification, falsification, or 20-period speed/width result.

## Independent raw-range scan

The instrumented `arch_ranges.h` layout is a 152-byte little-endian record: lane, cumulative ODE step, local `h`, then four physical states each with `(tube_lo,tube_hi,endpoint_lo,endpoint_hi)`. The 2,188,800-byte file contains exactly **14,400 unique `(lane,step)` records**: every lane 0–224 at every cumulative step 1–64, with `h=0.005`. There were no missing, duplicate, nonfinite, reversed, or endpoint-outside-tube records. Steps 1–60 cover the 15 completed safe control periods, `t∈[0,0.3]`. Steps 61–64 are the attempted period ending at `t=0.32` and carry the undecided property status.

An independent scan found no saved tube crossing `[-1.5,1.5]^4` through step 60. In the attempted period, only the lower bound for `θ̇₁` crosses `-1.5`: 3 lanes at step 61, 22 at step 62, 43 at step 63, and 45 at step 64. The affected lanes at step 64 are exactly 180–224, the high `θ₁` initial slice. These interval crossings are **uncertainty**, not a certified counterexample; the author checker likewise reported 45 `Unknown.` and no `Unsafe.`.

For the common accepted prefix, whole-time tube unions use steps 1–60; endpoint unions and per-box widths use the 225 intervals at step 60 (`T=0.3`). State order is `[θ₁, θ₂, θ̇₁, θ̇₂]`:

| State | Tube union through T=0.3 | Endpoint union at T=0.3 | Endpoint union width | Per-box endpoint width mean | Max |
| --- | --- | --- | ---: | ---: | ---: |
| θ₁ | [0.9667007870326361, 1.4082946248415773] | [0.9667007870326361, 1.271785133823919] | 0.30508434679128293 | 0.07387703180979621 | 0.08016687460638194 |
| θ₂ | [0.8162794718975555, 1.3234454236986024] | [0.8162794936069183, 1.0909160100466846] | 0.2746365164397663 | 0.06714920664922272 | 0.07353709445788248 |
| θ̇₁ | [-1.4881522439996342, 1.4862091487512852] | [-1.4881522428584502, -1.0110230305169263] | 0.47712921234152383 | 0.15976717071389565 | 0.18935854587664003 |
| θ̇₂ | [-1.456079758084754, 1.3000382302165772] | [-1.456079758084754, -0.9706595749168455] | 0.48542018316790847 | 0.17764641469926434 | 0.2063867870077748 |

The one-period all-225-box plumbing smoke is separately saved in `../native_dp_more_smoke1_001/`: one RPC, 900 range records, zero tube-band crossings, `VERIFIED` only through `T=0.02`. Neither the smoke nor the common prefix establishes the full `T=0.4` property. The checker is the existing CROWN-Reach/Flow* code; the independent range scan checks its saved enclosures and property band, and is not an independent floating-point NN soundness proof.

## Saved plots

`plots/dp_more_native_theta1dot_status.png` and `.pdf` show the 225-lane union of saved `θ̇₁` whole-step tubes. The saved `plot_native_more_status.py` and adjacent `.json` reproduce and describe it without a digest. Blue steps 1–60 are the author-checker accepted prefix through `T=0.30`; orange steps 61–64 are saved enclosures from the `UNKNOWN` period, with 3, 22, 43, then 45 lanes crossing the `-1.5` band; gray steps 65–80 have **no data** and are not interpolated. The source is the copied native `ranges.bin` and `native.log`, whose remote origin is recorded in `COPY.json`. A generic no-hash geometry, MATLAB script, PNG, PDF, and render receipt are also saved as `plots/dp_more_native_t_theta1dot_tube.*`; its geometry explicitly lists unobserved steps 65–80, but its one-series color does not encode the checker transition. Both figures are axis-aligned box projections, not Flow* octagons or independently certified neural-network evidence.
