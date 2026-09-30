# ARCH-COMP26 four-way contract manifest

`manifest.json` freezes the 16 currently identified non-VCAS instances and the
official report/repository identities. It is deliberately an audit scaffold,
not a claim that the 2026 contracts have already been reproduced.

`official_assets.json` resolves a narrower question: it hashes the official
specification, dynamics, and controller file candidate(s) at the pinned
repository commit. It deliberately leaves multiple official controller files,
continuous/discrete semantics, non-ONNX ingestion, and every execution-policy
field unresolved instead of guessing.
`manifest.json` hash-binds both this inventory and its detached-checkout audit
receipt; preflight also requires their repository URL, commit, schema, and
zero-mismatch receipt fields to agree before an external source identity can
support a resolved contract.

`execution_matrix.json` is the machine-readable 16-by-4 result index. Its
complete default cell records support/blockers, commands, source and binary
identity, arithmetic, runtime/hardware, the frozen one-cold/ten-steady
measurement plan, run/failure state, and a hash-bound external result-record
pointer. Every instance explicitly has all four method cells and every current
cell is `not_started`; an empty override is not a missing result.

The v2 matrix deliberately keeps timing, property, certificate, width, and raw
sample evidence out of the 64-cell index. A terminal cell must instead bind an
`archcomp26-cell-result-v1` record conforming to
`result_record.schema.json`. That record preserves every cold/steady sample,
the timing boundary and phase times, requested versus validated extent,
accepted/rejected work and NN calls, first failure, property and certificate
outcomes, eligibility flags, endpoint/last-segment/full-tube absolute bounds,
union versus per-partition width summaries, and artifact hashes. A `completed`
cell is rejected unless its executable plan is complete, its requested extent
matches the resolved continuous horizon or discrete transition count, every
planned sample covers that same extent and work counts, and all three width
views bind the contract variable order and exact endpoint/last/full domains.
`accepted_steps` means scheduled plant advances along the validated prefix
(not lane-expanded work), while `nn_calls` binds the contract's declared total;
every cold/steady sample must repeat those same workload counts. Lane-expanded
diagnostics belong in bound artifacts with their own explicit label.
Eligibility flags must agree with the run, certificate, finiteness, soundness
class, and soundness scope; empirical, one-step, unknown, or demonstrated
unsound evidence cannot opt itself into cross-tool ranking. A genuinely
unsupported method remains representable as an evidence-backed `skipped`
terminal cell rather than a fabricated executable run.

Changing an instance's manifest status to `resolved` is not enough to open the
gate. It must bind an `archcomp26-instance-contract-v1` record conforming to
`instance_contract.schema.json`. The record supplies the shared mathematical
contract and repository-bound evidence coverage; method-specific arithmetic,
runtime, command, source, and binary identities stay in the corresponding
matrix cell. No current instance has such a resolved record.

Run the read-only structural check with:

```bash
PYTHONPATH=src python -m torch_tm_flowpipe.archcomp26_preflight
```

To inspect one future cell without launching it, add
`--preflight <instance> <method>`. The command contains no launcher. It rejects
the current matrix because experiments are paused and contracts, support,
commands, identities, and runtime budgets are not yet resolved. Nested cell
overrides must be complete objects, so a partial `run`, plan, or result pointer
cannot silently discard default fields. The check also verifies schema,
contract, result, and referenced artifact bytes without a launcher or a
third-party schema runtime.

`docs/ARCHCOMP26_EXECUTION_STATUS.md` is a deterministic, generated status
report rather than the final experiment report. It resolves all cell defaults,
keeps archived QUAD evidence outside the 2026 matrix, and exposes every empty
cell as `not_started`. Check it without rewriting it with:

```bash
PYTHONPATH=src python -m torch_tm_flowpipe.archcomp26_status_report \
  --check docs/ARCHCOMP26_EXECUTION_STATUS.md
```

`docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md` is the deterministic Chinese report
skeleton for the final 16-by-4 delivery. It reads only the current manifest,
resolved instance contracts, and hash-bound result records; archived Huan, P3,
and native timings never enter its current-result tables. The draft keeps one
section per benchmark and reserves separate full-horizon timing, absolute
endpoint/tube width, reproduction, plot, failure, and unresolved-contract
sections. Check it with:

```bash
PYTHONPATH=src python -m torch_tm_flowpipe.archcomp26_final_report \
  --check docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md
```

Passing `--final` is deliberately fail-closed. It refuses output while
experiments are paused, an instance contract is unresolved, the matrix is not
terminal, any support state remains unassessed, or a supported/unsupported run
state is inconsistent. Failed, timed-out, and evidence-backed unsupported
cells remain visible final outcomes; they are not silently deleted. DOCX/PDF
generation is deferred until this shared final gate passes.

On 2026-10-01, `refs/heads/main` still resolved to the pinned
`d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26` commit. All 50 inventory
references (41 unique files) were independently re-hashed from a detached
checkout with zero mismatches. The exact audit is bound to the inventory in
`evidence/official_assets_audit_20261001.json`. Portable tests verify that
receipt and its counts; they do not fetch or re-hash the external repository.
This source audit also confirmed that the repository itself does not settle
several execution questions. The report/source conflict ledger in
`evidence/contract_audits_20261001.json` freezes the evidence without choosing
an unsupported interpretation. It now links every one of the 16 manifest rows
to one of 11 evidence audits:

- ACC's six-state, five-second safety contract is known, but the sign
  convention for the controller input `v_rel` is not stated and the property
  is a halfspace rather than a plot box;
- Airplane continuous is fixed to two seconds because the report and instance
  specification agree; the top-level `[0,20]` row is conflicting metadata.
  For discrete time, report page 89 gives forward Euler, the Airplane step is
  0.1 seconds, and the index set is `0..20`; however, no executable transition
  source or exact control-application ordering is selected, so the continuous
  ODE cannot be substituted;
- QUAD contains two distinct ONNX candidates. The report and pinned MATLAB /
  historical-author contracts are both preserved, with exact `x2`, `x4`, and
  `x5` differences. No new matched run may start until one dynamics and one
  controller identity are explicitly selected;
- Single Pendulum has a two-state report/specification but its repository
  dynamics returns the extra derivative `dx(3)=1` without the clock state's
  initial value or interface role;
- TORA requires an explicit raw-versus-post-processed controller boundary so
  the remain offset is applied exactly once. Its two heterogeneous controller
  activation descriptions conflict, and their reach-checker semantics remain
  unresolved;
- Unicycle includes a bounded disturbance in the report, while the repository
  dynamics omits it and carries a malformed disturbance comment;
- NAV's report state order is `[x,y,theta,nu]`, while the repository dynamics
  implements its first derivatives as `x3*cos(x4), x3*sin(x4)`, implying the
  swapped order `[x,y,nu,theta]` and corresponding swapped control-output
  semantics. Both controller input/output contracts remain unresolved;
- Attitude Control contains two distinct ONNX candidates. The report first
  defines avoidance of an unsafe box, then says the goal is to show that the
  specification does not hold, so both controller identity and checker
  polarity remain unresolved;
- Balancing is the repository's CartPole instance. Its report uses a
  five-feature trigonometric controller expression and a closed `[8,10]`
  property interval, while the repository comment/ONNX use four raw inputs and
  the specification says `t > 8`;
- Docking's report supplies the 40-second horizon missing from the instance
  file. Its coupled nonlinear velocity/position constraint is not an
  axis-aligned box;
- Double Pendulum's two official controllers and properties are distinct. The
  legacy more-robust row used the less-robust controller and only the singleton
  corner `{1.3}^4`, so its old result cannot populate the official
  more-robust cell;
- heterogeneous TORA supplies text/MAT controller artifacts but no ONNX file.

Every row is `unresolved`. Continuous/hybrid rows refer to the complete ODE
execution-field profile. Airplane discrete instead has a discrete transition,
step-semantics, and controller-update profile; ODE integration/order/remainder
fields are explicitly not applicable. Its `dynamics.m` is retained only as an
unselected continuous-dynamics candidate, never as the discrete transition.
The `legacy_candidate` mapping only records which older 14-configuration row
might seed the audit. It never promotes an older configuration or result into
a 2026 run. In particular:

- Airplane discrete and Docking have no legacy row;
- Airplane's time semantics and QUAD's equations have unresolved source
  conflicts;
- the legacy Double Pendulum more-robust mapping is known wrong and remains
  excluded from the 2026 execution cell;
- old completed/timed results remain regression references, not new reruns.

Experiments are marked paused. Before any launch after explicit user
authorization, read-only recheck the original native QUAD job's actual
terminal state and do not duplicate it.
