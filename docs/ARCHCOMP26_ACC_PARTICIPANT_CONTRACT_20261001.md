# ARCH-COMP 2026 ACC: named participant-order contract

Audit date: 2026-10-01. This profile freezes the controller feature transform used by the saved CROWN-Reach participant source. The paper does not specify the sign of `v_rel`, so the profile is an explicit execution choice, not a claim that the paper uniquely defines that sign. No content digest or SHA-256 check was performed in this audit.

## Official sources and physical specification

- [AINNCS 2026 report](https://easychair.org/publications/paper/GsKW/download), section 3.1, printed page 90 (PDF page 6); saved reading copy `/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_latest_20260930/reference/ARCH_COMP26_AINNCS.pdf`.
- [Fixed 2026 ACC specification](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/ACC/Specifications.txt) and [dynamics](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/ACC/dynamicsACC.m), read directly at commit `d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26`.

The six physical states, in order, are `(x_lead,v_lead,a_lead,x_ego,v_ego,a_ego)`. The complete initial box is `[90,110]×[32,32.2]×{0}×[10,11]×[30,30.2]×{0}`. Lead command `a_c,lead=-2`, air-drag coefficient `0.0001`, and the sampled ego command give

```text
x_lead' = v_lead             v_lead' = a_lead
a_lead' = -2*a_lead - 4 - 0.0001*v_lead^2
x_ego' = v_ego               v_ego' = a_ego
a_ego' = -2*a_ego + 2*a_c,ego - 0.0001*v_ego^2
```

The control period is 0.1 s for 50 updates, horizon 5 s. The **all-time** safety halfspace is `x_lead - x_ego - 1.4*v_ego - 10 >= 0`, equivalent to Flow* safe constraint `-x_lead+x_ego+1.4*v_ego+10 <= 0`. It is not an axis-aligned safe box. The archived participant partition has no split variables: `B=1` is the entire initial box. Its numerical settings are Flow* fixed step 0.1, order 3, cutoff `1e-6`, remainder estimate `[-0.1,0.1]`, and symbolic-remainder queue 50. The seventh and eighth Flow* variables are an auxiliary clock `t'=1` and held ego command `a_c,ego'=0`.

## Controller contract and limit

The report gives five controller features `(v_set,T_gap,v_ego,D_rel,v_rel)` and scalar output `a_c,ego`, but never defines the sign of `v_rel`. Both saved participant C++ submissions, `/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/sources/{native,xiangru}/submit/CROWN-Reach/archcomp/ACC/acc.cpp`, construct exactly

```text
[30, 1.4, v_ego, x_lead-x_ego, v_lead-v_ego] -> a_c,ego.
```

The fifth feature is explicitly `v_lead-v_ego` in both sources. The affine feature dependency must be retained when injecting controller bounds into the Taylor model; replacing it with five independent scalar ranges changes the contract. The saved `nncs_acc_adapter.py` computes feature Taylor models and interval roundoff enclosures for the fourth and fifth features. Its own scope says it does not certify the supplied auto_LiRPA/CROWN neural bounds.

The [fixed 2026 ONNX](https://github.com/Kiguli/ARCH-COMP2026/blob/d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26/benchmarks/ACC/controller_5_20.onnx) was saved locally as `/Users/shengenli/Documents/ChatGPT/verification/results/archcomp26_20261001/official_acc_controller_5_20.onnx` and in the isolated remote `runs/archcomp26_20261001/acc_prep_001/` directory. Its true external I/O is float32 `[1,1,1,5] → [1,1,1,1]`. The graph subtracts a five-element all-ones input mean, then has five 20-wide ReLU affine layers and a scalar linear output. The old server model file is directly byte-for-byte equal to this fixed model; no digest was calculated. The ONNX graph has no physical feature labels and does not itself settle the paper's relative-velocity sign.

## Previous execution and the VAR-tail boundary

The archived GPU config `research/gpu_verified_20260930/report/configs/arch_acc.yaml` and C++ source implement this participant-order profile. Six historical repetitions under `results/archcomp_review_20260923/evidence_v2/timing_v1/acc_r{0..5}_{huan,xiangru,native}/result.json` each report full completion; the Huan and Xiangru runs have `50/50` accepted one-box steps, and the native supervisors exit 0. These are historical results and must not be copied into a new no-digest comparison or restarted under their original run identity.

The historical native Flow* library has a separate known VAR-input truncation-tail issue. The saved isolated fix at `/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/native_var_tail_fix/expression.patch` changes `expression.h` at the `VAR_ID` evaluation and replay sites: it caches the polynomial truncation tail separately from the incoming remainder and adds the tail back during validation replay. The saved `RESULT.json` reports a targeted witness and four one-step ACC checks for that variant; its own scope expressly excludes a full benchmark or overall soundness qualification. A **new** full native ACC attempt must compile against this isolated patched library and record that correction as a changed numerical implementation. Old native times/verdicts remain historical reference, not proof for the patched variant.

## Executable next attempt

An isolated participant-order attempt may use the fixed 2026 ONNX, the one full initial box, the exact five-feature transform, 50 held controls, and the full-time halfspace checker. A one-period smoke must be labeled plumbing only. The full run must require 50 completed periods and accepted integration steps before any `VERIFIED` label; record all-time property bounds or safety statuses and the final six physical-state ranges. For native, pair the corrected Flow* library with a distinct RPC port and a server that loads the fixed saved model, record the controller calls and raw range trace, and preserve the old run files. The Huan/Xiangru no-digest attempts require the existing ACC Taylor-model feature adapter and preloaded compiled CUDA libraries so their launch path performs no content-digest check or extension build.
