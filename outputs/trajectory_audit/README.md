# Trajectory Visual Audit

This directory is a structured-output and plotting audit for the plant-only Van der Pol benchmark. It is a trend/visual audit of existing fixed-step/fixed-order behavior, not a new algorithm.

## Scope

- System: `dx/dt = y`, `dy/dt = y - x - x^2*y`
- Initial set: `x in [1.1, 1.4]`, `y in [2.35, 2.45]`
- PyTorch TM modes: `range_only`, `dependency_preserving`
- Flow* backend: generated toolbox C++ linked against `$FLOWSTAR_ROOT/flowstar-toolbox/libflowstar.a`
- Flow* patch status: No Flow* source patch was used; patch path, patch sha256, rebuild command, and patched libflowstar.a sha256 are not applicable.

## Semantics

- Flow* GNUPLOT boxes are flowpipe segment boxes. They are not final-time endpoint boxes.
- `endpoint_box_available=false` for Flow* rows in this audit, so endpoint widths are blank and endpoint ratios are disabled.
- `last_segment_width_*` is the width of the final flowpipe segment box.
- `tube_width_*` is the hull over all segment boxes.
- PyTorch `endpoint_width_*` comes from the final-time Taylor model range box; PyTorch segment rows come from Taylor-model segment ranges.
- Flow* internal runtime is parsed from `FLOWSTAR_RUNTIME_S`; compile, run, and total wall times are recorded separately.
- Samples are RK4 trajectories from corners, center, and a 5x5 grid. They are visual diagnostics only and are not proof.

## Files

- `flowstar_structured_summary.csv`
- `flowstar_segments/*_segments.csv`
- `torch_structured_summary.csv`
- `torch_segments/*_segments.csv`
- `samples/*_samples.csv`
- `figures/*.png`
- `figures/contact_sheet_*.png`
- `flowstar_vs_torch_overlay_summary.csv`
- `visual_audit_report.md`
- `crosscheck_summary.csv`
- `crosscheck_summary.md`

This audit is fixed-step/fixed-order. It is not Flow*_adaptive, not full CROWN-Reach, not CROWN, not auto_LiRPA, not a Jacobian-bound experiment, not sin/cos support, not hybrid automata, not a Flow* Python binding workflow, not an NN controller workflow, and not a new algorithm.
