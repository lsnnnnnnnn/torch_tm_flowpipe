# Airplane continuous: upper-corner first-step gate stopped at property Unknown

The preceding [`_007`](../native_airplane_binary6_firststep_20261002_007/README.md)
sample accepted one `h=0.01` numerical step for binary-cover cell `000000`.
This new isolated run used the **same 64-cell partition** of the official
full initial box and tested the opposite cell `111111` first: all six uncertain
initial coordinates `u,v,w,phi,theta,psi∈[0.5,1]`, with the other coordinates
zero. The [saved plan](SPLIT_PLAN.json) lists every cell; only `000000` and
`111111` have been sampled. The remaining 62 cells were not run.

The [generic entry source](airplane_binary6_coverage_gate.cpp) uses the
same frozen Airplane ODE, ONNX controller, six-output injection, Flow* order 3,
`h=0.01`, cutoff `1e-6`, and remainder `[-0.01,0.01]` as `_005`/`_007`.
Its only semantic change from `_005` is selecting the six initial intervals
from `AIRPLANE_CELL_BITS`. This run set `111111`; the actual [ledger](cells/111111/initial_boxes.json)
and one [CROWN RPC](cells/111111/controller_rpc.jsonl) record that box.
The [build receipt](BUILD.json), [driver](driver.py), and local [saved-file
audit](INDEPENDENT_AUDIT.json) retain the source and run checks. No earlier
job or frozen library was changed. The server identity is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_airplane_binary6_coverage_gate_20261002_008/cells/111111`.

The raw [RESULT](cells/111111/RESULT.json) is `failed/exit 2`, wall
**4.126563 s**, with no timeout. This outer failure is **not a Picard
numerical refusal**: the [native log](cells/111111/native.log) shows 19/19
first-Picard inclusions and **one accepted 0.01 s numerical segment**, saved
as one 408-byte [range record](cells/111111/ranges.bin). Flow* status
`3 = COMPLETED_UNKNOWN` and the independent box check `BOX_SAFE 0` cause the
driver to stop after that segment.

The [saved tube](cells/111111/safety.tsv) has upper bounds
`phi=1.0008947231602838`, `theta=1.0007017493894266`, and
`psi=1.0006611201867901`, each above the all-time safety limit `1`;
`cos(theta)` remains positive, lower bound `0.5397116711302705`.
The [independent audit script](audit_saved.py) parses the original range
record and confirms those tube bounds, one RPC, actual initial box,
19 Picard rows, and source change without rerunning a solver.

The [coverage-gate receipt](COVERAGE_GATE.json) calls the stop
`first_refusal`; here that label means **first failure of the combined
numerical-and-safety gate**. Numerically this cell's first step was accepted.
The property result is **Unknown**, because an enclosure crossing the safe
boundary alone is not an unsafe trajectory witness. No `T=2` run, all-cell
numeric acceptance claim, full-box safety claim, or counterexample is made.
