# ACC participant-order ours/P3: new 50-period diagnostic

The remote source directory is `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/acc_p3_full50_001`; this local directory is its small raw-evidence mirror. The separate one-period plumbing run is at `../acc_p3_smoke1_001/`. Neither reuses an old ACC experiment directory. No content digest or SHA-256 check was performed.

Contract: [fixed ACC participant-order profile](../../../../ARCHCOMP26_ACC_PARTICIPANT_CONTRACT_20261001.md), with fixed 2026 ONNX, full one-box initial set, five inputs `[30,1.4,v_ego,x_lead-x_ego,v_lead-v_ego]`, 50 control periods at 0.1 s, ODE order 3 and one fixed 0.1 s substep per period, and all-time halfspace `x_lead-x_ego-1.4*v_ego-10≥0`. The saved [new P3 launcher](../../../../../tools/archcomp26_acc_p3_nohash.py) used working-P3 numerical source `engine_quad_normalization_center`, strict endpoint module, the existing ACC exact-feature Taylor-model adapter, and eight prebuilt CUDA libraries. The launcher blocks SHA-256 construction and CUDA extension builds, and records the source paths, binary paths/sizes/mtime, environment, GPU and CPU affinity in [START.json](START.json). The run used physical GPU 2 and CPUs 10–13. This is a new ACC port of the working P3 core, not the 1024-lane QUAD qualification.

## Completion and safety evidence

- [RESULT.json](RESULT.json): `completed`, driver exit 0, 50/50 accepted substeps, 50 safety events, 50 feature constructions and 50 control injections. Runner wall 7.934 s (one cold diagnostic process), peak CUDA allocated 102,582,784 bytes and reserved 184,549,376 bytes. The driver's internal `time cost` line in [console.log](console.log) was 4.724 s.
- [ranges.jsonl](ranges.jsonl) has 50 consecutive one-lane records containing six physical-state tube intervals and direct local-domain endpoint intervals. [safety.jsonl](safety.jsonl) has 50 consecutive safe-set range evaluations. Every author unsafe-expression upper bound is nonpositive; its maximum is `-16.43485856971698`.
- Independently evaluating `x_lead_lo-x_ego_hi-1.4*v_ego_hi-10` on each saved **tube** box gave minimum lower bound **16.43485856971698 > 0**. It agrees with the author's all-time halfspace checker. This independent scan uses the saved numerical range boxes and does not independently certify the controller bounder.

## Local-domain recorded ranges

The endpoint intervals below come from the recorded local-domain Taylor-model prestate observer. They must be compared with the same observer definition across methods. The driver's terminal `HULL` lines use its separate endpoint routine and are retained in the console log, not silently substituted for these values.

| State | Full `t∈[0,5]` tube union | Direct T=5 endpoint interval | Endpoint width |
| --- | --- | --- | ---: |
| `x_lead` | `[89.98931255496008,250.05224032449735]` | `[229.04422885752757,250.0420950004359]` | 20.99786614290832 |
| `v_lead` | `[22.81833789484313,32.20137808961348]` | `[22.818361270318363,23.016844291993454]` | 0.19848302167509146 |
| `a_lead` | `[-2.0352223428763376,0.04123928916398211]` | `[-2.0289468032793545,-2.0282480738017417]` | 0.0006987294776128472 |
| `x_ego` | `[9.989803087339478,159.44518497064522]` | `[154.27949206972832,159.44256124676247]` | 5.163069177034146 |
| `v_ego` | `[27.269353118304767,30.203125216444086]` | `[27.269390730715905,29.127437020064843]` | 1.8580462893489376 |
| `a_ego` | `[-1.0806947908949769,0.045079206950296116]` | `[-1.0795664185608114,0.017330960153620165]` | 1.0968973787144316 |

This is an author checker result under an explicitly named participant input profile. The 2026 paper does not define the relative-velocity sign, and this run is not an end-to-end independently certified floating-point neural-network proof. One cold process time is not a five-run steady-state performance estimate.
