# ARCH-COMP 2026 QUAD paper contract — Xiangru original P2 parity

This is a new execution of the **2026 paper equations** selected by the user. It does not reuse the historical author-contract QUAD result. The source contract and the paper-versus-code equation differences are in `docs/ARCHCOMP26_QUAD_PAPER_CONTRACT_DECISION_20261001.md` in the local repository.

## Identity

- Remote root: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/quad_paper_xiangru_v1`.
- This directory is the compact copy of the entire remote run root: source config, isolated runner, preflight and full run records, raw logs, metrics, and all 1000 acceptance observations.
- Engine: `/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream`; shared Xiangru CROWN-Reach GPU driver at `src/flowstar_gpu/integrations/crown_reach.py`.
- Contract: 16 variables; initial `[-0.4,0.4]^6 × {0}^6`, partition `8×8×8×2×1×1 = 1024`; paper x2/x4/x5 equations; official Torch ONNX saved server copy; order 2; `h=0.005`; 20 ODE steps per 0.1 s control period; 50 periods to `T=5`; terminal target `0.94 ≤ x3 ≤ 1.06`.
- Method: sparse original Xiangru engine, parity (no `--strict`), box/same-slope CROWN, native float64 coefficient transport, native ONNX input layout, CROWN NN mode. The three existing CUDA `.so` modules were preloaded; extension JIT and SHA-256 routines were disabled. No content digest was computed. GPU3, CPUs 14–17; 14 GiB PyTorch allocator cap.
- The local `quad_paper.yaml` was directly byte-compared with the new Huan paper-run config and was identical. The controller saved server copy was directly byte-compared with the selected official 2026 Torch ONNX in the prior contract audit; this run did not repeat that comparison.

## Executed stages

| Stage | Coverage | Accepted lane steps | Outer wall | Driver elapsed | Peak CUDA reserved | Outcome |
|---|---:|---:|---:|---:|---:|---|
| `smoke1_001` | 1 box × 20 steps | 20/20 | 4.978 s | 0.927 s | 65,011,712 B | Completed, 0 broken |
| `batch1_001` | 1024 boxes × 20 steps | 20,480/20,480 | 5.228 s | 1.022 s | 3,573,547,008 B | Completed, 0 broken |
| `batch2_001` | 1024 boxes × 40 steps | 40,960/40,960 | 5.630 s | 1.427 s | 6,058,672,128 B | Completed, 0 broken |
| `full50_001` | 1024 boxes × 1000 steps | 1,024,000/1,024,000 | **108.018 s** | **103.686 s** | **14,979,956,736 B** | Completed, 0 broken; `VERIFIED` |

The 1000 records in `full50_001/data/observations.jsonl` are consecutively numbered 1–1000; every one reports 1024 accepted lanes, status 0 for all lanes, and no rejected lane. The driver also recorded 50 controller updates. Peak CUDA allocated was 9,352,376,832 B. `full50_001/RESULT.json` is the outer process receipt; `full50_001/data/RESULT.json` is the inner driver and acceptance receipt.

The terminal union at `T=5` has `x3 ∈ [0.967434441417146, 1.015176258384569]`, width `0.04774181696742297`, inside the target. The driver printed `VERIFIED`; all 16 terminal hulls are in `full50_001/data/metrics.json`. The exact final-hull JSON values match the separately run new Huan paper P2 parity result; both methods share much of the underlying GPU numerical source, so this equality is a reproducibility observation, not independent correctness evidence. Huan's separate single-run wall was 94.583 s and driver time 90.471 s on GPU2; these are two individual process timings on different devices, not a stable speed ratio.

The short-stage driver printed `FALSIFIED` because its terminal checker applied the `T=5` target to `T=0.1` or `T=0.2`. Those short runs are resource and acceptance diagnostics only. The complete run's target check is the relevant result. The existing parity/CROWN floating-point path has no independent end-to-end NNCS proof. Stepwise tube and endpoint hulls were not exported in this run; the acceptance trace and final endpoint hull are retained.
