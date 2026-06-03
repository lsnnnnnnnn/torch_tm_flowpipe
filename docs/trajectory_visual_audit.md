# Trajectory Visual Audit

This audit is for Flow* and PyTorch TM trajectory and trend visualization on the plant-only polynomial Van der Pol benchmark. The goal is structured output, comparable segment semantics, and visual diagnostics for fixed-step/fixed-order runs. It is not a new reachability algorithm.

## Scope

- System: `dx/dt = y`, `dy/dt = y - x - x^2*y`
- Initial box: `x in [1.1, 1.4]`, `y in [2.35, 2.45]`
- PyTorch TM modes: `range_only` and `dependency_preserving`
- Flow* cases: `rem1e-4_cut1e-10` at `h=0.01`, `steps=10`, orders `4` and `2`; `rem1e-10_cut1e-15` at `h=0.0025`, `steps=10`, order `8`
- Comparison mode: fixed step and fixed order only

Run:

```bash
python experiments/trajectory_visual_audit.py --system van_der_pol --out-dir outputs/trajectory_audit
```

The script uses the existing generated toolbox C++ path. Generated cases include `Continuous.h`, call `ode.reach(...)`, use `setting.setFixedStepsize(h, order)`, and are compiled through the runner that links `-lflowstar` from `$FLOWSTAR_ROOT/flowstar-toolbox/libflowstar.a`.

## Flow* Patch Status

No Flow* source patch is used for this audit. The parser reads GNUPLOT segment rectangles emitted by the generated C++ cases, so patch path, patch sha256, rebuild command, and patched `libflowstar.a` sha256 are not applicable.

## Box Semantics

Flow* GNUPLOT boxes are flowpipe segment boxes. They are not final-time endpoint boxes. Therefore Flow* rows set `endpoint_box_available=false`, leave endpoint widths blank, and do not contribute endpoint ratios.

PyTorch rows record both final-time endpoint Taylor-model range boxes and segment Taylor-model range boxes. `last_segment_width_*` always means the final segment box. `tube_width_*` is the hull across all segment boxes.

## Runtime Semantics

Flow* `flowstar_internal_reach_s` is parsed from `FLOWSTAR_RUNTIME_S` printed by the generated case around `ode.reach(...)`. Flow* compile, run, and total wall times are recorded separately. PyTorch `runtime_s` measures the local `flowpipe_multi_step(...)` call for each mode/order case.

## Outputs

- `outputs/trajectory_audit/flowstar_structured_summary.csv`
- `outputs/trajectory_audit/flowstar_segments/*_segments.csv`
- `outputs/trajectory_audit/torch_structured_summary.csv`
- `outputs/trajectory_audit/torch_segments/*_segments.csv`
- `outputs/trajectory_audit/samples/*_samples.csv`
- `outputs/trajectory_audit/figures/*.png`
- `outputs/trajectory_audit/figures/contact_sheet_torch_orders.png`
- `outputs/trajectory_audit/figures/contact_sheet_flowstar_overlays.png`
- `outputs/trajectory_audit/figures/contact_sheet_width_trends.png`
- `outputs/trajectory_audit/flowstar_vs_torch_overlay_summary.csv`
- `outputs/trajectory_audit/visual_audit_report.md`
- `outputs/trajectory_audit/crosscheck_summary.csv`
- `outputs/trajectory_audit/crosscheck_summary.md`
- `outputs/trajectory_audit/README.md`

Sampling uses corners, center, and a 5x5 grid with RK4 trajectories. Sampling is visual diagnostic evidence only, not proof.

This audit is not Flow*_adaptive, not full CROWN-Reach, not CROWN, not auto_LiRPA, not a Jacobian-bound experiment, not sin/cos support, not hybrid automata, not a Flow* Python binding, and not an NN controller workflow.
