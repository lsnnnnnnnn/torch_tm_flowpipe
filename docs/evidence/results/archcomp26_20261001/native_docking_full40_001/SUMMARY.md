# ARCH-COMP 2026 Docking — isolated native Flow* full attempt

This is a **new full initial box** attempt, not a rerun of an old point box. The saved outer receipt is `failed`, exit 2, wall **9.139415808022022 s**. Its numerical horizon completed: **40/40** one-second control periods, **400/400** 0.1-second Flow* tubes, and **40** controller RPC calls. The full-time radial property result is **UNKNOWN**, not safe or unsafe.

## Contract and entry

- Source contract: `output/flowstar_latest_20260930/repo/docs/ARCHCOMP26_DOCKING_BALANCING_SOURCE_CONTRACT_20261001.md`, fixed official 2026 Docking specification and ONNX. Initial physical box (one unsplit lane): `(sx,sy)∈[70,106]^2`, `(vx,vy)∈[-0.28,0.28]^2`. Auxiliary `t,Fx,Fy` start at zero. The model is the saved `/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/Docking/model.onnx`, previously directly compared byte for byte to the fixed official model; this run did no new content digest or model comparison. The ONNX itself includes preprocessing/postprocessing; raw four physical states are sent and two physical forces are used.
- Continuous plant: `sx'=vx`, `sy'=vy`, `vx'=2*0.001027*vy+3*0.001027^2*sx+Fx/12`, `vy'=-2*0.001027*vx+Fy/12`. Control is held for each 1 s period; the saved method uses fixed 0.1 s, Taylor order 3, cutoff 1e-6, remainder estimate `[-0.01,0.01]`, symbolic remainder queue 1000.
- Every full-time tube is checked with outward MPFR intervals for `q = sqrt(vx²+vy²) − 0.2 − 0.002054*sqrt(sx²+sy²)`. Decimal constants use directed-rounding string constructors. A tube is certified safe only when `q.upper ≤ 0`; any positive upper bound is **uncertainty**, never a counterexample. Uncertainty is accumulated without stopping numerical integration. Flow* numerical noncompletion would stop the run.
- Local source: `output/flowstar_latest_20260930/repo/tools/archcomp26_docking_native_nohash.cpp`, `build_archcomp26_docking_native_nohash.py`, and `run_archcomp26_docking_native_pair.sh`. Remote build root: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_docking_build_002`; complete raw run: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_docking_full40_001`. GPU 2, CPU 10–13, exclusive RPC port 5104. Saved server PID 3422800 owned 5104; native PID 3422864. The outer timeout budget was 1800 s and did not fire.

## Saved evidence and audit

`START.json` and `RESULT.json` are the untouched outer receipts; `native.log`, `server.log`, `controller_rpc.jsonl`, `safety.tsv`, `ranges.bin`, `initial_boxes.json`, and `listeners_at_start.txt` are the copied raw files. `ranges.bin` consists of 400 records: little-endian lane, step and local step size, followed by each physical state's full-time tube lo/hi and right endpoint lo/hi. `safety.tsv` has one record per tube; `global_substep=s` denotes the closed local tube at `t∈[(s−1)/10,s/10]`. `SLOPE_CHECK_ALL.json` is a separate read-only replay of all 40 saved RPC input boxes: CROWN's lower and upper affine slope tensors were elementwise equal in every one (`max_abs_slope_difference=0`). It did not rerun the plant. `audit_saved.py` parses the original receipts and generates `AUDIT.json`; it confirmed exact row counts, lane/step sequence, model response dimensions, finite bounds, and the retained `failed` outer status.

| Physical state | T=40 endpoint lo | T=40 endpoint hi | Endpoint width | Full-time tube union lo | Full-time tube union hi |
|---|---:|---:|---:|---:|---:|
| sx | -56.10811233900303 | 217.47556332674057 | 273.5836756657436 | -56.10811233900303 | 217.47556332778632 |
| sy | -54.04233135534693 | 219.8138264025062 | 273.8561577578531 | -54.04233135676655 | 219.8138264025062 |
| vx | -6.512224614228157 | 5.880721649716058 | 12.392946263944214 | -6.512224614228157 | 5.880721649716058 |
| vy | -6.433739482845608 | 5.9554995887926525 | 12.389239071638261 | -6.433739482845608 | 5.9554995887926525 |

The initial-box radial margin was `[-0.5079082336541199,-0.0073558285335368345]`, certified negative. Already at the first tube `t∈[0,0.1]`, the saved interval was `[-0.50799175157193943,+0.016082965472827226]`. The last tube was `[-0.83512702112234516,+8.9543472273715725]`, also the maximum saved upper bound. All 400 tube intervals cross zero: **0** certified safe tubes, **400** inconclusive tubes, **0** tubes with strictly positive lower bound. These are conservative box evaluations, not a point trajectory or an unsafe witness.

The one-period plumbing smoke has a separate run directory `../native_docking_smoke1_001/` and untouched `RESULT.json`: exit 2 / `failed`, wall **4.025295520201325 s**, 10/10 numerical tubes, 1 RPC, radial `UNKNOWN`. The full attempt did not reuse its results. The converted ONNX/CROWN RPC and Flow* interval plant constitute this method's saved diagnostic; there is **no independent end-to-end floating-point ONNX certificate**. No content-hash or SHA-256 check was performed in this audit.
