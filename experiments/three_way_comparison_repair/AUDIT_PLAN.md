# Audit plan and experiment-changing behavior inventory

The audit traces the exact adapter and upstream paths, reproduces each anomaly,
then changes one semantic choice at a time. “Matched” below means comparable
output meaning, not identical internal algorithms.

| Behavior | Torch | DiffReach | Flow* | Matched? | Visible in old report? | Consequence |
|---|---|---|---|---|---|---|
| Local order | complete degree 1 | affine flag or restricted quasi-quadratic | complete degree 2 in old harness | No | Partly | Never call stress rows “same order” |
| Output order | truncates to requested degree | upstream flag-dependent projection | TM setting order | No | Partly | Configuration caveat, not a semantic match |
| Step policy | fixed | fixed | old fixed; original VDP adaptive 0.002–0.1 | Only in A–E | Partly | Invalidates general failure claims |
| Cutoff | Torch truncation policy | operation-specific upstream thresholds | explicit interval cutoff | No | Yes | Sensitivity only |
| Candidate remainder | growth initializer | `init_remainder` | configured per state | No | Yes | Internal validation resource |
| Native refined remainder | returned validated residual | upstream Picard/remainder result | returned by `advance` | Yes as native output | No | Must be exported unchanged |
| Post-advance overwrite | none | none | old adapter assigned candidate after success | No | Described as workaround | Invalidates old Flow* bounds |
| Endpoint substitution | direct raw object now exposed | composed segment with time fixed to `h` | composed Flowpipe evaluated at endpoint | Yes | No | Primary endpoint contract |
| Endpoint-specific refinement | legacy `final_tm`, now supplemental | none found | none in called path | No | No | Invalidates old Riccati ranking |
| Box reset | Protocol C only | Protocol C only | Protocol C only | Yes in C | Yes | Controls representation but erases dependency |
| Normalization | native TM generators | upstream affine/symbolic normalization | normalized TM Flowpipe | No | Partly | Separate native rows |
| Symbolic remainder | none equivalent | window 100 | original VDP window 100 | No | Partly | Native-mode caveat |
| Dtype | Torch float64 CPU | JAX x64 CPU forced at constructor | MPFR interval backend | No | Yes | Soundness/runtime caveat |
| Rounding backend | IEEE tensor operations | IEEE/JAX interval-style operations | MPFR-directed intervals | No | Partly | Analytic tests do not equal proofs |
| Range bounding | Torch polynomial interval evaluation | `QuadTM.eval_interval` | Flow* interval evaluation/composition | No | Partly | Report algorithm-specific widths only |
| Failure handling | exception/category mapping | contraction/category mapping | structured instrumented return reason | Yes at schema level | No | Old horizon claims lacked a cause |

Evidence sequence:

1. Record all repositories, worktrees, versions, devices, and frozen hashes.
2. Reproduce stock/reinjected Flow*, Torch raw/tightened, and DiffReach raw
   endpoints plus tubes on the minimal cases.
3. Trace Flow*'s fixed-step/fixed-order `Expression<Real>` overload from first
   Picard inclusion through each refinement round and return site.
4. Compare the actual original Flow* Van der Pol executable, an
   identical-settings generated harness, and the repaired generic harness.
5. Apply the analytic, trajectory, schema, carry, parity, failure, and
   frozen-artifact gates before selecting Outcome A, B, or C.

The old report, result report, protocol, Flow* comparison adapters, and available
notes were included in the claim audit. No separate LaTeX deck or speaker notes
were found in the inspected tree.
