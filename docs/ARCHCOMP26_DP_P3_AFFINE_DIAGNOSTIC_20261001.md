# ARCH-COMP 2026 DP less-robust: directed affine controller attempt

Date: 2026-10-01. This is a new 225-box numerical **attempt** for the fixed continuous benchmark, following the failed interval-forward controller diagnostic in [ARCHCOMP26_DP_P3_DIRECTED_DIAGNOSTIC_20261001.md](ARCHCOMP26_DP_P3_DIRECTED_DIAGNOSTIC_20261001.md). Its 20-period result is incomplete and cannot fill a successful 2026 result cell. It used no content digest, SHA-256, CUDA JIT build, or `git rev-parse`.

## Contract and implementation

The common benchmark contract is [ARCHCOMP26_DOUBLE_PENDULUM_LESS_CONTRACT_20261001.md](ARCHCOMP26_DOUBLE_PENDULUM_LESS_CONTRACT_20261001.md): fixed official 2026 less-robust ONNX, continuous four-state ODE, `[1,1.3]^4` split into 225 boxes, 20 sampled control periods of 0.05 s, five 0.01 s plant substeps per period, and whole-time safety `[-1.7,2]^4`. The server's 2024-path controller had already been compared directly byte-for-byte with the fixed 2026 ONNX. No comparison or digest was repeated at execution.

The unchanged P3 DP numerical port uses work order 3, point order 2, validation order 4 (`solution_plus_one`), the geometric reciprocal candidate, strict injection and endpoint, trig direct reuse, and full symbolic remainder. Its server directory is:

`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/dp_p3_affine_v1/`

The new local sources are [archcomp26_dp_affine_controller.py](../tools/archcomp26_dp_affine_controller.py), [archcomp26_dp_interval_controller.py](../tools/archcomp26_dp_interval_controller.py), and [archcomp26_dp_p3_nohash.py](../tools/archcomp26_dp_p3_nohash.py). Each completed remote run directory contains a copy of the exact runner and both controller modules used for that attempt, plus the outer `START.json`/`RESULT.json`, stdout, stderr, generated config, inner result, and per-step observations. Nothing in a frozen engine source tree was edited. Physical GPU 3 and CPU 14–17 were used; GPU 3 was checked free before the full attempt.

## Controller-local directed enclosure

The fixed ONNX is checked to have only `MatMul→Add→ReLU→MatMul→Add→ReLU→MatMul→Add`, with float32 weights of shapes `4×25`, `25×25`, and `25×2`. Each stored float32 number is exactly representable in binary64. For every neuron the new abstraction keeps interval coefficients `A`, interval bias `b`, and interval remainder `R` so that the exact real network value lies in `A·x+b+R` for every input `x` in the CROWN hull. Linear layers propagate all three parts using the existing directed interval dot/add operations. If a ReLU preactivation has directed range `[l,u]` crossing zero, it selects a stored binary64 `α∈[0,1]` and adds the directed residual bound

`ReLU(z)−αz ∈ [0, max(−αl, (1−α)u)]`.

If the range is entirely positive or nonpositive, the ReLU is passed through or zeroed exactly. The final directed control residual is `(A−T)·x+b+R`, where `T` is the same float64 CROWN linear map used by the original driver. The unrounded auto_LiRPA bias interval is retained only for diagnostic inversion counts; it is not injected. Driver coupling-width metrics now describe the injected directed residual. This establishes a controller-local enclosure under the stated graph and interval-arithmetic contracts; it is **not** a complete end-to-end proof of the DP flowpipe, specification checker, or floating-point implementation.

Controller-only preflight `dp_p3_port_v1/controller_preflight_004` used the driver's actual initial sparse hull, with no plant integration. Both B=1 and B=225 affine residuals were finite, ordered, and contained inside the looser independent interval-forward residuals (2/2 and 450/450 outputs). The first box has no crossing ReLU in either hidden layer and affine residual widths `[3.44e-15,4.52e-15]`. Across 225 boxes, maximum residual width is `0.06025086831469739` and median `0.009260120098194213`, versus `0.9277244972312` maximum for the interval-forward fallback. The first and second hidden layers saw 182 and 67 crossing preactivations among their 5625 lane-neuron entries. The preflight saved per-layer maxima/means for coefficient, bias, and remainder interval widths. Deterministic 100-digit Decimal evaluation checked 18 B=1 and 72 selected B=225 output residual values inside the affine bounds; point samples are supporting checks, not exhaustive proof.

An additional controller-only `controller_preflight_005` selected the worst-width initial lane 96 as well as lanes 0, 1, 10, and 224. Its 18 B=1 and 90 B=225 deterministic Decimal output checks all stayed inside the same affine bounds; the smallest observed slack was about `1.54e-15`. This adds an adversarial-width sample and does not change the mathematical proof boundary.

## New runs and exact stop

| Server directory under `dp_p3_affine_v1` | Result |
| --- | --- |
| `smoke1_001` | First box × one period: 5/5 accepted substeps; every valid four-state tube inside the safety box. Outer exit 0. |
| `smoke225_001` | All 225 boxes × one period: 1125/1125 accepted lane-substeps; every valid four-state tube inside the safety box; `broken=0`. Peak CUDA allocated 183,675,392 B and reserved 891,289,600 B. |
| `full225_001` | **225 boxes × 20 periods attempted. Incomplete.** It recorded 60 of the required 100 substeps, with 13,478 accepted lane-substeps in that prefix. The first contraction failure is substep 56, lane 20. By substep 60, six lanes have status `1 = FAILED_CONTRACTION`; 219 remain active. The strict control-boundary endpoint requires every lane active and fails closed with `ValueError` at the period-12 handoff. Outer exit 1. No further period was run. |

The full attempt's directed controller residual maximum widths by period 0–11 were `0.0603, 0.1169, 0.1795, 0.0699, 0.0684, 0.1091, 0.1373, 0.1702, 0.2117, 0.3622, 0.7272, 1.5511`. Its inner wall was 26.902 s and outer wall 27.988 s. A final `metrics.json` was not written because the endpoint raised before the shared driver's post-run metrics block; exact full-attempt peak CUDA memory and final aggregate hull therefore remain unavailable. The per-step JSONL, controller audit, traceback, exact command, and copied source remain available.

The whole-time safety containment check becomes undecidable **before** the contraction failure. At substep 48, accepted lane 96 has fourth physical-state tube `[-1.7017804287873026,-0.8735910587945372]`, which crosses the safety lower bound `-1.7`. By substep 60, 127 still-active lanes have tubes that are not wholly inside the safety box. The saved last-step tube ranges do not show an entire variable interval outside the box, so this record establishes neither a 1-second safety proof nor a counterexample. The old `suite_v1/ours` run used a different engine and remains separately labeled legacy evidence.

## Remaining blockers

1. The present affine relaxation still widens enough that some accepted tubes cross the safety boundary at `t≈0.48 s`; the complete 1-second property is not certified. A stronger verified controller relaxation or a documented property-refinement method is needed, with the same initial 225 boxes and full horizon.
2. Six lanes eventually fail P3 self-map contraction; no per-component subset margins were persisted in this attempt, so the exact contraction margin cannot be stated. The strict endpoint correctly stops rather than carrying failed lanes through a control refresh.
3. The controller-local directed proof and its Decimal samples do not replace an end-to-end audit of P3 reciprocal, trig, symbolic remainder, endpoint, and safety composition for the 7-variable DP contract.

No validation cap, initial-box count, or time horizon was relaxed. Neither incomplete full attempt is a successful latest-method ARCH-COMP 2026 result.
