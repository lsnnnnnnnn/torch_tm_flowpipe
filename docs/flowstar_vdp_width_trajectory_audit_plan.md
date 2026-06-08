# Flowstar Van der Pol Width/Trajectory Audit Plan

Goal: consolidate existing Flow*/PyTorch Van der Pol width and trajectory evidence into one authoritative audit without adding a new flowpipe mechanism, symbolic queue variant, or Flow* source patch.

## Initial Repository State

Captured before this audit implementation on branch `codex/flowstar-normalized-insertion`.

```text
git status --short --branch
## codex/flowstar-normalized-insertion...origin/codex/flowstar-normalized-insertion

git branch --show-current
codex/flowstar-normalized-insertion

git log --oneline -12
7101408 Publish batched TM GPU microbenchmark evidence
b2b1a6d Add batched TM GPU microbenchmark
4d8f1e7 Add Flowstar accepted step trace comparator
c07a0b8 Add three-way Flowstar symqueue v2 audit evidence
7e81d6a Add Flowstar symbolic queue v2 audit
346772e Add Horner insertion diagnostics
3778e90 Add normal-eval right-map diagnostics
c56f817 Add Flowstar width attribution and scalar fix run
a2ac372 Add split symbolic queue semantics diagnostics
82a814c Fix h10 failure target width diagnostics
ed23e61 Add normalized insertion symqueue diagnostics
f667f73 Add normalized insertion h10 diagnostics

git remote -v
origin git@github.com:lsnnnnnnnn/torch_tm_flowpipe.git (fetch)
origin git@github.com:lsnnnnnnnn/torch_tm_flowpipe.git (push)
```

## Scope

- Inventory existing parity, fixed-step, trajectory, normalized insertion, width attribution, symbolic queue, accepted-step trace, and GPU strategy artifacts.
- Parse existing CSV/MD files only by default; do not rerun expensive h10 experiments.
- Treat Flow* GNUPLOT rectangles as flowpipe segment boxes, not endpoint boxes.
- Disable endpoint ratios unless both tools explicitly provide endpoint boxes.
- Treat sampling trajectories and overlays as visual diagnostics only.
- Mark accepted-step ordinal trace attribution noncausal when Flow* and PyTorch accepted `t` or `h` differ.

## Outputs

- `outputs/flowstar_vdp_width_trajectory_audit/evidence_inventory.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/evidence_inventory.md`
- `outputs/flowstar_vdp_width_trajectory_audit/summary.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/width_comparison_ledger.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/trajectory_overlay_ledger.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/claim_boundary_checks.csv`
- `outputs/flowstar_vdp_width_trajectory_audit/report.md`

## Next Comparator Work

The accepted-step comparator should next emit `attempt_aligned_trace_diff.csv` and `forced_h_trace_diff.csv` so channel localization is causal rather than accepted-ordinal only.
