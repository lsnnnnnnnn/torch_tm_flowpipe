# ARCH-COMP 2026 QUAD paper contract — Huan P2 parity

This is a **new** execution of the 2026 paper equations. The historical Huan author-contract run was not restarted or used as this result. The selected controller is the official Torch ONNX; the choice and the equation differences are recorded in `docs/ARCHCOMP26_QUAD_PAPER_CONTRACT_DECISION_20261001.md`.

## Identity

- Remote run: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/quad_paper_huan_full50_001`
- Local compact evidence: this directory (`quad_paper.yaml`, `launcher_paper.py`, `parity/`, `supervisor/`).
- Huan engine: `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/engine_huan_sr_chunk`; sparse engine, box CROWN, same-slope, parity mode, order 2, ODE h=0.005.
- CUDA device 2, CPU affinity 10–13. Three existing CUDA shared objects were preloaded; extension JIT loading was disabled. The launcher replaced `hashlib.sha256` with a function that raises before importing Torch or the driver. The run completed without triggering it. No digest computation or checksum comparison was made in this attempt.
- Supervisor started 2026-10-01T09:43:24.070802+00:00; exited 0 at 09:44:58.669416+00:00, wall 94.583 s. Driver elapsed 90.471 s. No timeout.

## Coverage and result

- Initial set: `[-0.4,0.4]^6 × {0}^6`, with physical splits `8×8×8×2×1×1 = 1024`.
- All 50 control periods `k=0…49` have 1024 active boxes; 20 ODE substeps per period, 1000 total. `broken=0`.
- At `T=5`, the final union for `x3` is `[0.967434441417146, 1.015176258384569]`, inside the paper target `[0.94,1.06]`; `t` is `[4.999999999999916, 4.999999999999916]` in the emitted hull. Driver printed `VERIFIED`.
- Peak CUDA allocated: 6,992,032,256 bytes; reserved: 11,806,965,760 bytes.

The `VERIFIED` label is the existing Huan endpoint checker applied to the selected paper contract. It is not an independent floating-point neural-network proof. `metrics.json` and the original driver stdout are retained for per-period widths and all final-state hulls.

The separate one-box, one-period plumbing smoke is in `../quad_paper_huan_smoke1_001/`. It exited 0 with `broken=0` and printed `FALSIFIED` at `T=0.1`; that short diagnostic is not a full-horizon property result.
