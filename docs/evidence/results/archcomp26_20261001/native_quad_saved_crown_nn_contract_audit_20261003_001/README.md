# QUAD first-call saved CROWN/NN contract: read-only gate design

This is a **read-only contract audit**, not a neural-network certificate or a new solver attempt. It reads the [already saved 1,024-box first RPC](../native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json) and the frozen server construction recorded by [the native builder](../../../../../tools/build_archcomp26_quad_paper_native_nohash.py). No Flow*, CROWN, ONNX inference, plant integration, or remote job was run for this audit. The original native octagon production gate remains **CLOSED**.

## Saved contract and missing upper-slope evidence

The first-call launcher selects `quad_controller_3_64_torch.onnx` from the CROWN-Reach-GPU ARCH-COMP2024 QUAD directory; the locally saved [model and 2026 contract audit](../../../../ARCHCOMP26_QUAD_PAPER_CONTRACT_DECISION_20261001.md) identify a 12-input, 3-output controller with four `Gemm` layers and three hidden `Sigmoid` layers. The frozen `crown.py` source selected by the native builder constructs `T` from `A['lA']`, `u_min` from `A['lbias']`, and `u_max` from `A['ubias']`. It requests `activation_bound_option='same-slope'`, but does not save `A['uA']` or an independently checked equality `uA=lA`. Consequently the saved RPC alone does **not** establish that `u_max` is an upper bias for the **same** matrix `T` injected into the plant. This is a missing certificate field, not evidence that the saved bounds are false.

The native [injection code](../native_quad_allbox_firststep_recenter_gate_20261003_003/quad_allbox.cpp) converts every saved `T`, `u_min`, and `u_max` entry to binary32 before constructing the control Taylor model. The independent NN question is therefore, for every saved RPC input box `X` and output `j`, whether the **actual selected ONNX mathematical map** satisfies

```text
float32(u_min[j]) <= f_j(x) - sum_i float32(T[j,i]) * x_i <= float32(u_max[j])
for every x in X.
```

Here every stored binary32 weight and bias in the ONNX graph is interpreted as its exact real value. This checks the real-valued network map represented by the file. A separate operation-by-operation roundoff envelope would be needed for a claim about a particular float32 inference runtime.

## Why independent plain intervals cannot close this gate

The small [saved-RPC reader](audit_saved_rpc.py) checks the 1,024 by 12 input grid and 1,024 by 3 coefficient grid and reproduces [AUDIT.json](AUDIT.json). It applies the native binary32 transport before computing `width(T·X) = sum_i |T_i| width(X_i)`. For the first box and first output:

| Saved quantity | Width |
| --- | ---: |
| `T·X` | 3.3901568215 |
| `[u_min,u_max]` after binary32 transport | 0.06709861755 |
| Width ratio | 50.525 |

Across all 1,024 boxes, first-output `width(T·X)` is **3.387991–3.391097** while saved residual width is **0.065733–0.068608**. If `f(X)` and `T·X` are enclosed separately as ordinary intervals, the width of their interval difference is at least `width(T·X)`. Thus that method cannot fit the saved first-output residual interval for **any** of the 1,024 boxes, even if its separate `f(X)` enclosure had zero width. This proves a limitation of that checker, not a violation by any neural-network output. The other two outputs have much smaller `T·X` widths, but their NN enclosures have not been evaluated here.

## Smallest independent certificate attempt

1. Parse the selected ONNX graph with explicit checks for its input/output layout, four `Gemm` weight/bias shapes and attributes, three `Sigmoid` nodes, and complete wiring. Lift each stored binary32 parameter exactly; use the already saved RPC input intervals and native binary32-transported `T`, `u_min`, `u_max`. Confirm the controller file selected on the server corresponds to the saved local model by direct byte comparison if needed, without a digest.
2. Start with **one existing box**. Propagate a correlated affine enclosure `A*x+b+R` through each linear layer using directed Decimal or MPFR interval operations. At a sigmoid preactivation interval `Z`, choose a fixed exact slope `alpha` and strictly enclose the scalar relaxation error `sigmoid(z)-alpha*z` for all `z in Z`; add that error to `R` while retaining the original input symbols in `A`. Directed `exp` bounds and monotonic sigmoid bounds, followed by derivative-sign tests or interval bisection for extrema of the scalar error, suffice. A bounded search must return `UNDECIDED` on exhausted precision or subdivisions.
3. At the final layer, subtract the saved **transported** `T*x` within the correlated representation and require each of the three resulting residual intervals to lie in its saved transported bias interval. Preserve per-output numeric margins and first undecided location. If one box passes, apply the same checker to all **1,024 saved boxes**; every box must pass. If a bound is too broad, refine only that box's six nonzero input axes and prove the subboxes cover it before joining their residual bounds. A broad independent interval outside the saved bias is `UNDECIDED`, never a network counterexample by itself.

The repository's [DP affine residual checker](../../../../../tools/archcomp26_dp_affine_controller.py) supplies a useful `A*x+b+R` pattern, but its activation rule is for `ReLU`. The existing [smooth-controller Taylor model code](../../../../../research/gpu_verified_20260930/source/engine/src/flowstar_gpu/nn_tm.py) explicitly leaves point coefficient roundoff outside its validity claim; the [existing sigmoid interval primitive](../../../../../research/gpu_verified_20260930/source/engine/src/flowstar_gpu/transcendental.py) uses empirically calibrated `exp` ULP slack. Neither is by itself the proposed independent strict sigmoid certificate.

The saved RPC (1.6 MB) and local ONNX model (about 55 KB) provide the inputs for a CPU-only streaming check of one box and, if it succeeds, all 1,024 first-call boxes. No new GPU, CROWN server, or plant run is intrinsically needed. Runtime and whether an independently chosen affine sigmoid relaxation fits the **tight saved biases** are unknown until the one-box certificate is attempted. Even 1,024 successful residual checks would cover **only the first control call's NN input boxes** and would not qualify later controls, the remaining 19 plant steps of the first period, or the full `T=5` reach-and-remain claim.
