# Airplane continuous native — first Picard refusal in the actual Real path

The saved 2026 full initial box is one unsplit 19-coordinate box: physical
`x=y=z=r=p=q=0`, `u,v,w,phi,theta,psi∈[0,1]`, and seven auxiliary
coordinates zero. This independent order-3 diagnostic called the fixed
12→6 Airplane controller once, then requested **one** `h=0.01` Flow* step.
It did not run the remainder of the first period or the `T=2` task. The
original three smoke attempts and their library/source were not changed.

The [copied-library patch](Continuous.patch) adds only a readout of every
coordinate's old Picard remainder estimate, newly proposed remainder and
polynomial-difference interval in the `Expression<Real>` symbolic-remainder
overload used by `ODE<Real>`. The original first-refusal branch remains in
place. The copied [Airplane entry](airplane_first_reject_trace.cpp) changes
the reach request from 0.1 to 0.01 second, checks one rather than ten segments,
and avoids printing a full-task `VERIFIED` label. The [build receipt](BUILD.json)
records three successful compile/archive/link steps. Before running, the
trace marker was confirmed in the new binary, and the run directory was empty.

The raw [RESULT](run/RESULT.json) is `failed/exit 2`, no timeout, wall
3.675378 s. [Native output](run/native.log) records status
`4 = UNCOMPLETED_SAFE`, zero accepted/saved segments and 19 ordered Picard
rows. The old preset remainder is `[-0.01,0.01]` for every coordinate. The
first non-inclusion is `x`; all non-inclusions are:

| Coordinate | Proposed first-Picard remainder | In old estimate? |
| --- | --- | --- |
| `x` | `[-0.069390255181559155, 0.070282484851510674]` | No |
| `y` | `[-0.067922303239625254, 0.068510490934515375]` | No |
| `z` | `[-0.034019498423153313, 0.034303339461087413]` | No |
| `phi` | `[-0.012568092636247242, 0.012495548690721597]` | No |
| `theta` | `[-0.010752397075873792, 0.010751150195381528]` | No |

The other 14 proposed intervals are included in their preset remainder
estimates. The [independent saved-log audit](INDEPENDENT_AUDIT.json), generated
by [audit_saved.py](audit_saved.py), re-parsed both `_004` and `_005` raw logs
and confirmed the same full initial box and exactly identical single CROWN
RPC, both status 4 and zero ranges, plus all 19 finite, ordered proposed
intervals and their inclusion flags. The [raw RPC](run/controller_rpc.jsonl),
[box ledger](run/initial_boxes.json), [launcher/server logs](run/launcher.stdout.log),
[empty range file](run/ranges.bin), [START](run/START.json), and [driver](driver.py)
remain available separately from the audit.

This identifies the **first numerical Picard self-map refusal for this
isolated order-3, full-box method setting**. It is not an actual unsafe
trajectory, a full-period flowpipe, a `T=2` result, or a claim that changing
the remainder estimate would produce a valid complete run. `_004` is the
earlier one-step alignment control; `_001`–`_003` stopped during preparation
without solver launches, so `_005` was used as a fresh run ID.
