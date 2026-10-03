# Airplane continuous: all 64 binary-cover cells accepted one numerical step

The fixed 2026 continuous Airplane initial box has six uncertain states
`u,v,w,phi,theta,psi∈[0,1]`; the other initial coordinates are zero. The
[saved plan](SPLIT_PLAN.json) bisects each uncertain interval into closed
`[0,0.5]` and `[0.5,1]`, forming 64 cells whose union is the full initial
box. Earlier isolated receipts cover [`000000`](../native_airplane_binary6_firststep_20261002_007/README.md)
and [`111111`](../native_airplane_binary6_coverage_gate_20261002_008/README.md).
This run attempted **only the other 62 cells**. It did not restart either old
cell or the unsplit experiment.

The isolated [driver](driver.py) and [C++ source](build/archcomp/airplane/airplane_binary6_numeric_cover.cpp)
kept the same Airplane ODE, ONNX controller, first held control, Flow* order 3,
`h=0.01`, cutoff `1e-6`, and `[-0.01,0.01]` starting remainder as the two
earlier cells. The [build receipt](BUILD.json) reports success. The remote run
ID was
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_airplane_binary6_numeric_cover_20261003_001`.
The [coverage receipt](COVERAGE_GATE.json) records **62/62** new cells
numerically accepted, in 253.594 s for the isolated queue. Each cell saved
its own 16 original files under [`cells/`](cells/), including START/RESULT,
actual initial ledger, controller RPC, native/Picard log, safety row, and one
408-byte binary range record.

The independent [saved-file auditor](audit_saved_independent.py) reconstructed
the 64-cell partition directly from the six bit positions and confirmed that
the new cell IDs are exactly the complement of the two prior IDs. It required
all **992/992** new raw files, then independently parsed the prior two and
the new 62 receipts: one HTTP 200 controller call, exactly 19 successful
first-Picard inclusions, one finite `h=0.01` range record with its endpoint
inside the tube, and agreement among the actual box, RPC, native status,
safety row, outer result, and driver ledger. Its [result](INDEPENDENT_AUDIT.json)
is **64/64 cells with one accepted numerical 0.01 s segment**, including
**1216/1216** saved first-Picard inclusion rows. Re-run only this offline
audit, without a solver, with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B audit_saved_independent.py
```

The new cells split into **7 one-step SAFE** and **55 property Unknown**.
The SAFE bit IDs are `111000`, `011000`, `101000`, `110000`, `001000`,
`010000`, and `100000`. Including the two prior receipts gives **8 SAFE**
and **56 Unknown** over the 64-cell cover. An Unknown cell has an accepted
numerical segment whose enclosure is not wholly inside the safety set; its
outer `RESULT.json` can therefore be `failed/exit 2`. This is a property
stop, not a numerical Picard refusal and not an actual unsafe trajectory.

This evidence covers only the **first 0.01 s plant step per cell**, conditional
on the saved controller envelopes. It does not cover the remainder of the
first `0.1 s` held-control period, later controls, the `T=2 s` horizon,
full-box safety, or independent NN/CROWN and Flow* floating-point soundness.
The 16×4 benchmark matrix remains unchanged until a real full-horizon result
is available.
