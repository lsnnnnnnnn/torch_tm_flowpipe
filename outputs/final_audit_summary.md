# Final Audit Summary

Generated on 2026-06-02 for the plant-only Van der Pol fixed-step/fixed-order audit cleanup.

## Git

- branch: `main`
- local HEAD: `c74de89e6ce0a278bda455d644e7abcabadbc1c4`
- remote origin/main: `c74de89e6ce0a278bda455d644e7abcabadbc1c4	refs/heads/main`
- git status --short at audit-summary generation:

```text
M outputs/flowstar_provenance_manifest.json
 M outputs/flowstar_provenance_manifest.md
```
- provenance source_tree_commit_used_for_generation: `c74de89e6ce0a278bda455d644e7abcabadbc1c4`
- provenance generation_worktree_status: `clean`
- provenance remote_origin_main_at_generation: `c74de89e6ce0a278bda455d644e7abcabadbc1c4	refs/heads/main`
- artifact_bundle_commit_note: `This manifest is generated from the clean source tree recorded in source_tree_commit_used_for_generation. The artifact bundle commit containing this refreshed manifest may be later.`

## Pytest

- install command: `conda run -n py11 python -m pip install -e ".[test]"`
- pytest command: `conda run -n py11 pytest -q`
- pytest result: `42 passed in 5.90s`
- final staged-artifact pytest rerun: `42 passed in 5.53s`
- final pre-commit pytest rerun: `42 passed in 6.29s`

## Authoritative CSV Checks

| file | LF/UTF-8/no NUL | data rows | line count | tool rows | Flow* status rows |
| --- | --- | ---: | ---: | --- | --- |
| `outputs/tm_order_audit_vdp_order2_8.csv` | `True` | 126 | 127 | <none>: 126 | <none> |
| `outputs/van_der_pol_diagnostics_by_order_v2.csv` | `True` | 126 | 127 | <none>: 126 | <none> |
| `outputs/flowstar_vdp_remainder_cutoff_sweep.csv` | `True` | 252 | 253 | flowstar: 252 | completed: 122, failed: 130 |
| `outputs/flowstar_vdp_plot_input_v2.csv` | `True` | 266 | 267 | flowstar: 252, torch_tm_flowpipe: 14 | completed: 122, failed: 130 |

Expected row-count checks: `tm_order_audit_vdp_order2_8.csv` has 126 data rows plus header; `van_der_pol_diagnostics_by_order_v2.csv` has 126 data rows plus header; `flowstar_vdp_remainder_cutoff_sweep.csv` has 252 Flow* rows; `flowstar_vdp_plot_input_v2.csv` is nonempty and contains both torch and Flow* rows.

## Flow* Generated C++ Audit

- generated C++ case count: `252`
- existing C++ case files: `252`
- all cases include `Continuous.h`: `True`
- all cases call `ode.reach(...)`: `True`
- all cases use `setting.setFixedStepsize`: `True`
- Flow* completed/failed count: `completed=122`, `failed=130`

## Semantics

- endpoint boxes for Flow* GNUPLOT rows: `False`
- endpoint ratios allowed: `False`
- ratio types retained: `last_segment, tube`
- endpoint/last_segment/tube semantics: Flow* GNUPLOT artifacts provide last-segment and tube boxes, not endpoint boxes; torch ratios are limited to matching last-segment/tube semantics.
- runtime_s source: `FLOWSTAR_RUNTIME_S internal reach time when present`
- compile/run wall-time columns separate: `True`

## Claim Boundaries

- plant-only: `True`
- fixed-step/fixed-order: `True`
- Flow* adaptive baseline: `False`
- full CROWN-Reach pipeline: `False`
- raw Taylor-model coefficient comparison: `False`
- new algorithm added: `False`
- excluded features: `CROWN`, `auto_LiRPA`, `Jacobian bounds`, `sin/cos`, `hybrid automata`, `Flow* core binding`, `Flow* rewrite`, `NN controller`.

## Next research target: polynomial range looseness diagnostic

Plant-only Van der Pol only: compare current polynomial interval range evaluation against a sampling/subdivision diagnostic. The goal is to quantify looseness before designing any new range bounding method. This is not a CROWN, auto_LiRPA, Jacobian-bound, Flow* binding, Flow* rewrite, hybrid-automata, sin/cos, or NN-controller task.
