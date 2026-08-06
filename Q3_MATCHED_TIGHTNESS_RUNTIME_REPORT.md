# Q3 matched tightness/runtime audit

# FORMAL MATCHED COMPARISON NOT AUTHORIZED

## Plain-language answers

1. **What is Xiangru Q3?** A dense complete total-degree-3 Taylor support over local time plus five TORA state/control parameters, with 84 monomials, two polynomial Picard rounds and ten interval-remainder Picard rounds.
2. **Did the original Q3 baseline reproduce?** Yes. The fresh clean-commit run exits 0, verifies all 48 leaves for 200 segments, and reaches T=20. Its 2,850 checked non-runtime numbers agree with the author artifact within the documented `1e-6` tolerance, with maximum absolute error `1.155e-13`.
3. **Plant-only or NN/controller?** Closed-loop. It includes the frozen homogeneous ReLU TORA NN, auto_LiRPA bounds, and a controller update every one second.
4. **Is Torch order-3 semantically the same?** Only the elementary total-degree retention predicate agrees. The dense/sparse basis, Picard count, remainder algorithm and actual candidate order/model do not.
5. **Which benchmarks truly match?** None across Torch and Xiangru. Torch and stock Flowstar share a Van der Pol family, but that does not match Xiangru's TORA baseline.
6. **Was the Torch trace mislabel fixed?** Yes. Actual lifecycle objects are recorded and hashed; wrong-object tests fail closed, call-44 metadata is corrected, and instrumented/uninstrumented numerical outputs agree.
7. **Did both sides complete the same horizon?** No common benchmark was run. Xiangru completes its native T20 TORA contract; the existing Torch candidate targets T10 VDP.
8. **Does tightness mean endpoint or tube?** Neither is compared cross-tool here. The schemas keep endpoint and full-step tube distinct; Xiangru's certificate uses the full tube.
9. **What runtime stages/devices are included?** Xiangru reports CUDA float64 compile/warm, controller, plant dynamics, validation and totals. Torch's available VDP lane is CPU plant-only. Setup/loading/I/O are unavailable separately and are not guessed.
10. **Can we legally say which is tighter or faster?** No. The sole decisive blocker is the absence of an existing Torch lane implementing the exact TORA plant, state/control representation, frozen NN/controller-update contract, B48 initial set and T20 workload.
11. **What is the single next priority?** Implement an exact, independently validated native Torch homogeneous-TORA closed-loop lane—including the frozen controller and output contract—then rerun Gates 1–5 before any winner table.

## Gate result

Gate 1 fails on model, dimension, coordinates, controller and initial set. Gate 2 independently fails full algorithmic order equivalence. Gates 3–5 are `NOT_RUN`, not passed or waived. Consequently:

| Result | Status |
|---|---|
| Formal endpoint tightness | `N/A` |
| Formal tube tightness | `N/A` |
| Formal Torch/Xiangru runtime ratio | `N/A` |
| Formal three-way ranking | `N/A` |
| Case | **B** |

No trajectory overlay, interpolated interval endpoint, width ratio, speed ratio, or winner table was generated.

## Native Xiangru timing, not a comparison

The one fresh exact reproduction records 141.0664 s compile/warm separately excluded, 1.33359 s controller, 1.07129 s plant dynamics, 1.92362 s validation, 2.68962 s solver excluding validation and 4.61324 s total including validation. Cold process wall is 151.020 s on CUDA GPU 0. This is a native implementation observation from one reproduction, not the minimum five-repeat short-runtime protocol and not an algorithm-speed claim.

Authoritative gate, runtime and provenance evidence is under `outputs/xiangru_q3_matched_audit_20260806/`; `manifest.sha256` covers every artifact except itself.
