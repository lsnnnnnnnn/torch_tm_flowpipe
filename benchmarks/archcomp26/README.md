# ARCH-COMP26 four-way contract manifest

`manifest.json` freezes the 16 currently identified non-VCAS instances and the
official report/repository identities. It is deliberately an audit scaffold,
not a claim that the 2026 contracts have already been reproduced.

`official_assets.json` resolves a narrower question: it hashes the official
specification, dynamics, and controller file candidate(s) at the pinned
repository commit. It deliberately leaves multiple official controller files,
continuous/discrete semantics, non-ONNX ingestion, and every execution-policy
field unresolved instead of guessing.

`execution_matrix.json` is the machine-readable 16-by-4 result scaffold. Its
complete default cell records support/blockers, commands, source and binary
identity, arithmetic, runtime/hardware, run/failure state, timing/memory, and
common-prefix/endpoint widths. Every instance explicitly has all four method
cells and every current cell is `not_started`; an empty override is not a
missing result.

Run the read-only structural check with:

```bash
PYTHONPATH=src python -m torch_tm_flowpipe.archcomp26_preflight
```

To inspect one future cell without launching it, add
`--preflight <instance> <method>`. The command contains no launcher. It rejects
the current matrix because experiments are paused and contracts, support,
commands, identities, and runtime budgets are not yet resolved. Nested cell
overrides must be complete objects, so a partial `run` or `metrics` update
cannot silently discard default fields.

On 2026-10-01, `refs/heads/main` still resolved to the pinned
`d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26` commit. All 50 inventory
references (41 unique files) were independently re-hashed from a detached
checkout with zero mismatches. The exact audit is bound to the inventory in
`evidence/official_assets_audit_20261001.json`. Portable tests verify that
receipt and its counts; they do not fetch or re-hash the external repository.
This source audit also confirmed that the repository itself does not settle
several execution questions. The report/source conflict ledger in
`evidence/contract_audits_20261001.json` freezes the evidence without choosing
an unsupported interpretation:

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
- Attitude Control also contains two distinct ONNX candidates;
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
- the Double Pendulum more-robust controller mapping is suspect;
- old completed/timed results remain regression references, not new reruns.

Experiments are marked paused. Before any launch after explicit user
authorization, read-only recheck the original native QUAD job's actual
terminal state and do not duplicate it.
