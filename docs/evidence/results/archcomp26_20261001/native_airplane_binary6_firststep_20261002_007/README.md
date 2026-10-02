# Airplane continuous: one first-step sample from a full-box binary cover

The fixed 2026 continuous contract has six uncertain initial states
`u,v,w,phi,theta,psi∈[0,1]`; all other initial coordinates are zero.
`SPLIT_PLAN.json` partitions each uncertain interval into closed
`[0,0.5]` and `[0.5,1]`, giving 64 distinct boxes whose union is the
original full box. This is the smallest **uniform grid that bisects all six
uncertain axes once**; it is not proven to be the fewest boxes needed for
numerical acceptance. Only cell `000000`, with all six intervals `[0,0.5]`,
was run here. The other 63 cells have **not** been attempted.

The prior unsplit order-3, remainder `[-0.01,0.01]` first-step trace
[`_005`](../native_airplane_first_reject_trace_20261002_005/README.md)
rejected before saving a segment: first-Picard proposals in `x,y,z,phi,theta`
were outside the preset remainder. The separate unsplit `[-1,1]` trace
[`_006`](../native_airplane_rem1_first_reject_trace_20261002_006/README.md)
also rejected. A single-axis split has no saved feasibility evidence; the
position RHS couples velocities and angles, while `theta` additionally enters
the angular denominators. Order 3 was chosen because it reaches the Picard
step with bounded resources; the previous order-6 author entry encountered
a much larger monomial-table resource demand. These observations motivate a
bounded grid diagnostic, not a claim that this grid is optimal.

The new isolated [source](airplane_binary6_firststep.cpp) differs from
`_005` only in the six initial intervals. It uses the same frozen ODE,
12→6 controller and output mapping, one CROWN call, Flow* order 3,
`h=0.01`, cutoff `1e-6`, and preset remainder `[-0.01,0.01]`.
The [driver](driver.py) linked that source to `_005`'s already instrumented
copied library; no frozen or earlier-run file was modified. The [build
receipt](BUILD.json) and [actual initial ledger](run/initial_boxes.json)
identify the one sampled cell. The new remote run ID is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_airplane_binary6_firststep_20261002_007`.

The raw [RESULT](run/RESULT.json) is `completed/exit 0`, wall **3.775791 s**,
with one CROWN [RPC](run/controller_rpc.jsonl). The [native log](run/native.log)
shows all 19 first-Picard proposals inside their old remainder estimates,
status `2 = COMPLETED_SAFE`, and **one accepted 0.01 s segment**. The
[ranges.bin](run/ranges.bin) is one 408-byte record for the 12 physical
states, and [safety.tsv](run/safety.tsv) records this one tube inside the
four specified coordinate bounds with `cos(theta)` lower bound
`0.8773788041312982`. The [saved-file audit](INDEPENDENT_AUDIT.json),
reproducible with [audit_saved.py](audit_saved.py), checks the 64-cell plan,
the six-only source edit, actual ledger and RPC input, status, 19 Picard
rows, range record, and saved safety bounds without starting a solver.

This is a **single-subbox numerical first-step smoke**. It does not establish
first-step acceptance over the original full initial set, a complete 0.1 s
control period or `T=2` horizon, the all-time property, a strict NNCS
certificate, or a runtime comparison. A later [opposite-corner gate](../native_airplane_binary6_coverage_gate_20261002_008/README.md) sampled cell `111111` and stopped at property Unknown after one accepted numerical step. The other 62 cells remain unattempted.
