# ARCH-COMP 2026 QUAD: paper-contract decision

Decision date: 2026-10-01. The user selected the equations printed in the 2026 AINNCS report for the new four-method QUAD main comparison. The official benchmark repository's different equations are a separate comparison contract. Historical Huan speed and reachability results retain their original author-code contract and must not be relabeled as results for the paper equations.

## Primary sources and scope

- [2026 AINNCS report](https://easychair.org/publications/paper/GsKW), section 3.9, printed pages 97–98 (PDF pages 13–14). The saved reading copy is `/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_latest_20260930/reference/ARCH_COMP26_AINNCS.pdf`; page 97 was checked visually, since text extraction scrambles mathematical layout.
- [Official QUAD dynamics at fixed benchmark commit `d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/QUAD/dynamics.m) and [instance specification](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/QUAD/Specifications.txt), read directly from that commit.
- Historical author execution source: `/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/sources/native/submit/CROWN-Reach/archcomp/Quadrotor/quad.cpp`, lines 53–64 for the ODE, 77–96 for the initial box and partition, and 250–261 for the endpoint target test. The saved author YAML is `research/gpu_verified_20260930/source/benchmark/quad_author_resolved.yaml`.

The report describes 12 physical states in this order: `x1` north position, `x2` east position, `x3` altitude, `x4` longitudinal velocity, `x5` lateral velocity, `x6` vertical velocity, `x7` roll, `x8` pitch, `x9` yaw, `x10` roll rate, `x11` pitch rate, `x12` yaw rate. The controller has three outputs `u1,u2,u3`; the report describes three hidden layers of 64 sigmoid neurons and an identity output, sampled every 0.1 s. The plant parameters are `g=9.81`, `m=1.4`, `Jx=Jy=0.054`, `Jz=0.104`, `tau_psi=0`.

## Paper dynamics selected for the new main comparison

Here `s_i=sin(xi)` and `c_i=cos(xi)`. The executable transcription below follows the printed equations, including the printed sign in `x9'`; it does not substitute a textbook quadrotor model. The printed `sin(x8, )` in `x4'` is a punctuation error; the intended scalar call `sin(x8)` is also used in the repository source.

```text
x1'  = c8*c9*x4 + (s7*s8*c9 - c7*s9)*x5 + (c7*s8*c9 + s7*s9)*x6
x2'  = c8*s9*x4 + (s7*s8*s9 + c7*c9)*x5 + (c7*s8*s9 - s7*c9)*x6
x3'  = s8*x4 - s7*c8*x5 - c7*c8*x6
x4'  = x12*x5 - x11*x6 - g*s8
x5'  = x10*x6 - x12*x4 + g*c8*s7
x6'  = x11*x4 - x10*x5 + g*c8*c7 - g - u1/m
x7'  = x10 + s7*tan(x8)*x11 + c7*tan(x8)*x12
x8'  = c7*x11 - s7*x12
x9'  = (s7/c8)*x11 - (c7/c8)*x12
x10' = ((Jy-Jz)/Jx)*x11*x12 + u2/Jx
x11' = ((Jz-Jx)/Jy)*x10*x12 + u3/Jy
x12' = ((Jx-Jy)/Jz)*x10*x11 + tau_psi/Jz = 0
```

The report and official instance file give the full initial set `[-0.4,0.4]^6 × {0}^6`, horizon 5 s, and 50 controller intervals of 0.1 s. The target region is `x3 ∈ [0.94,1.06]`. The report says the controller should reach and remain in this region within 5 s; the instance file says stabilize in 5 s. The historical author code checks **only the end-of-time flowpipe** with `isInTarget`. Record that endpoint check as the historical checker, and record the paper's reach-and-remain wording separately until its temporal checker is fixed from the 2026 submitted execution source. Since the initial altitude includes values below 0.94, a claim of membership for every time in `[0,5]` is impossible for this initial set.

The report specifies the full initial set, not the numerical partition. The author code partitions dimensions `x1,x2,x3,x4,x5,x6` into `8,8,8,2,1,1` (1024 boxes). A new shared partition may use that partition for comparability, but it must be identified as an execution choice and applied to the same full initial set across methods.

## Exact differences from the official repository and historical author source

The official fixed-commit `dynamics.m` and the saved historical `quad.cpp` have the same expressions in these three places:

| Derivative | Selected 2026 paper equation | Official repository and historical author equation |
| --- | --- | --- |
| `x2'` | `c8*s9*x4 + (s7*s8*s9 + c7*c9)*x5 + (c7*s8*s9 - s7*c9)*x6` | `c8*s9*x4 + (s7*s8*s9 - c7*c9)*x5 + (c7*s8*s9 + s7*c9)*x6` |
| `x4'` | `x12*x5 - x11*x6 - g*s8` | `x12*x5*x11*x6 - g*s8` |
| `x5'` | `x10*x6 - x12*x4 + g*c8*s7` | `x10*x6 - x11*x6 - g*s8` |

These are distinct plant dynamics, so historical widths, verification verdicts and timings cannot serve as results for the newly selected paper model. `x12'=0` is consistent with the paper because `Jx=Jy` and `tau_psi=0`; it is not another discrepancy.

## Controller selection and remaining execution decisions

The fixed official QUAD directory contains two ONNX candidates, [`model.onnx`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/QUAD/model.onnx) and [`quad_controller_3_64_torch.onnx`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/QUAD/quad_controller_3_64_torch.onnx), plus `model.mat`. The repository's [`load_controller.m`](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/QUAD/load_controller.m) accepts an `onnxfile` argument and can write `model.mat` and `model.onnx`; it does not select a default input. The saved historical CROWN-Reach `crown.py:35–36` explicitly selects `quad_controller_3_64_torch.onnx` from its ARCH-COMP2024 tree.

Direct ONNX graph inspection on 2026-10-01 resolves their **mathematical controller map**. `model.onnx` has MATLAB layout `float32 [N,1,1,12] → [N,3]`, with a `Sub` input-mean node, four `Conv` affine layers, and three `Sigmoid` nodes. `quad_controller_3_64_torch.onnx` has PyTorch layout `float32 [1,12] → [1,3]`, with four `Gemm` affine layers and three `Sigmoid` nodes. The 12 MATLAB input means are exactly zero. After reshaping each Conv kernel to the corresponding Gemm matrix, **all four weight matrices and all four bias vectors compare element-by-element equal**, with maximum absolute difference zero. Four deterministic inputs from `[-0.4,0.4]^6×{0}^6` (zero, both all-boundary corners, and one mixed point) gave equal float32 outputs in the same NumPy affine/sigmoid evaluation. This proves the same real-valued 12-to-3 network from their graph operators and parameters; the four samples are a check, not a claim of all-input floating-point bitwise equivalence across different runtimes. The two ONNX files have different serialized bytes and tensor layouts. The official PyTorch ONNX is directly byte-for-byte equal to the saved current QUAD snapshot controller; no hash was calculated.

For the new paper-dynamics four-method main row, select `quad_controller_3_64_torch.onnx` with a documented `[1,12]` input and `[1,3]` output as **our execution choice**; retain the MATLAB Conv ONNX as an implementation comparison. The 2026 participants' exact file choices are still unverified. Freeze the exact paper-equation transcription in the executable model and inspect the actual parsed expressions. Freeze the property checker time semantics, the full-box partition, the control update schedule, and method-specific numerical settings. A first small smoke may verify plumbing only; it is not a completed 5 s benchmark result.

## Executable transcription preflight

The separate [paper-equation Huan P2 config](../benchmarks/archcomp26/configs/quad_paper_p2_huan.yaml) changes only the three differing right-hand sides from the archived author YAML and labels the selected torch ONNX map. It is a **new** contract file; archived QUAD measurements do not apply to it. On the server, the current `nncs_env` PyYAML parser read 16 variables, 16 right-hand sides, order 2, and 50 controller periods. The existing `ode_compiler.compile_ode(..., order=1)` accepted each changed right-hand side separately and the complete 16-output model (228 tape instructions, 519 cache slots). This was a static grammar/tape preflight without numerical integration, a property check, a GPU run, or a content digest.

## First new paper-contract Huan parity attempt

A one-box, one-period plumbing smoke and a separate full 1024-box, 50-period parity run completed on 2026-10-01. The local [full-run evidence and summary](evidence/results/archcomp26_20261001/quad_paper_huan_full50_001/SUMMARY.md) record the separate remote directory, copied config and launcher, 50 period metrics, and original stdout. The full run used GPU 2 and CPUs 10–13, preloaded the existing CUDA shared objects, blocked CUDA-extension JIT loading and the `hashlib.sha256` constructor, and made no content-digest check. The supervisor exited 0 after 94.583 s; the Huan driver reported 90.471 s, `B=1024`, all 50 periods with 1024 active boxes, 20 ODE substeps per period, and `broken=0`. Its T=5 final union for `x3` was `[0.967434441417146,1.015176258384569]`, inside the endpoint target `[0.94,1.06]`, and the existing driver printed `VERIFIED`.

That label is the Huan **endpoint checker** outcome for our selected paper-equation/Torch-ONNX execution choice. It does not resolve the report's reach-and-remain wording or establish an independent floating-point neural-network proof. The one-period smoke printed `FALSIFIED` at T=0.1 because it had not reached the terminal target; it is not a full-horizon verdict. The other three paper-contract method cells remain separate from this Huan result.
