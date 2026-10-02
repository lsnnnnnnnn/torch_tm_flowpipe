# Airplane continuous native: widened remainder first-step trace

This is a **new isolated diagnostic** of the previously saved order-3
`[-1,1]` remainder profile. The original
[`native_airplane_fullbox_order3_rem1_smoke1_001`](../native_airplane_fullbox_order3_rem1_smoke1_001/RESULT.json)
was not restarted or altered. The copied [entry source](airplane_first_reject_trace.cpp)
uses the same unsplit official 19-coordinate box (12 physical states and seven
auxiliaries), frozen 12→6 ONNX, order 3, `h=0.01`, and `[-1,1]` preset
remainder. Its reach request was limited to **one** `0.01 s` step and one
CROWN RPC. No `T=2` job was launched.

The [copied-library patch](Continuous.patch) only prints each coordinate's
old remainder estimate, first Picard proposal, polynomial difference, and
inclusion flag in the actual `Expression<Real>` symbolic-remainder path.
The original refusal branch remains. The [build receipt](BUILD.json) records
successful compile, archive, and link steps. The [raw RESULT](run/RESULT.json)
is `failed/exit 2`, no timeout, wall **3.625502 s**. [Native output](run/native.log)
reports `4 = UNCOMPLETED_SAFE`, zero accepted/saved segments; [ranges.bin](run/ranges.bin)
is empty, so its `BOX_SAFE 1` is vacuous. There is no property result.

All 19 first-Picard proposals are finite and ordered. Five are **not**
contained by the old `[-1,1]` estimate:

| Coordinate | First proposed remainder |
| --- | ---: |
| `x` (first refusal) | `[-2.52122191537212, 2.5218805989602933]` |
| `y` | `[-2.4294572167945838, 2.4298688160039572]` |
| `phi` | `[-3.0477667710707669e20, 3.0477667710707669e20]` |
| `theta` | `[-2.7771020966823759e20, 2.7771020966823759e20]` |
| `psi` | `[-9.0629475684749672e19, 9.0629475684749672e19]` |

The other 14 proposals, including `z` at approximately
`[-0.652667,0.652766]`, fit. The default `[-0.01,0.01]` trace had a different
failed-coordinate set (`x/y/z/phi/theta`); widening the preset changed the
Picard proposals and did not make this full-box first step complete.

The [independent saved-file audit](INDEPENDENT_AUDIT.json), reproducible with
[audit_saved.py](audit_saved.py), parsed the raw logs and confirmed: one
identical initial box and **exactly identical** CROWN RPC across the old
widened run, the default-remainder trace, and this run; matching fixed model
and server paths; and a new entry source equal to the old widened entry after
only limiting the reach request to one step and updating segment count/output
labels. The raw [RPC](run/controller_rpc.jsonl), [initial box](run/initial_boxes.json),
[START](run/START.json), [server log](run/server.log), [driver](driver.py), and
[runtime audit](AUDIT.json) remain available.

This identifies a **numerical Picard self-map refusal in this widened
order-3 profile**. The very large angular intervals are interval proposals,
not trajectories. The run supplies no accepted flowpipe, actual unsafe
trajectory, complete-period result, `T=2` result, or runtime-ranking sample.
