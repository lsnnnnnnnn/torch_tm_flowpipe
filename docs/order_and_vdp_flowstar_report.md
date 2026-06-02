# Order semantics, Van der Pol diagnostics, and Flow* fixed-order comparison

## Scope

This report is limited to diagnostics, experiment harnesses, metrics, plots, tests, and documentation for the existing `torch_tm_flowpipe` Taylor-model prototype. It does not add CROWN-Reach, auto_LiRPA, Jacobian bounds, sin/cos support, hybrid guards or jumps, Flow* bindings in the core library, or adaptive Flow* as the main baseline.

The Van der Pol benchmark is:

```text
dx/dt = y
dy/dt = y - x - x^2*y
x0 in [1.1, 1.4], y0 in [2.35, 2.45]
```

## Reproduction

Use the `py11` conda environment and the chenxin415 Flow* toolbox root:

```bash
cd /srv/local/shengenli/torch_tm_flowpipe
export FLOWSTAR_ROOT=/srv/local/shengenli/flowstar
conda run -n py11 python -m pip install -e '.[test]'
conda run -n py11 pytest -q

conda run -n py11 python experiments/tm_order_audit.py --systems van_der_pol --orders 2 3 4 5 6 7 8 --h-values 0.0025 0.005 0.01 --steps-values 1 5 10 --csv outputs/tm_order_audit_vdp_order2_8.csv
conda run -n py11 python experiments/diagnose_van_der_pol.py --orders 2 3 4 5 6 7 8 --h-values 0.0025 0.005 0.01 --steps-values 1 5 10 --csv outputs/van_der_pol_diagnostics_by_order_v2.csv
bash scripts/run_vdp_order_flowstar_study.sh
```

The generated C++ Flow* cases are linked against `flowstar-toolbox/libflowstar.a`.

## Order Semantics

The torch `order` parameter is a requested total-degree cutoff. Actual retained degree can be lower than requested because the ODE structure, truncation, validation, and endpoint substitution may not populate every degree.

The audit CSV now records the dependency reference explicitly:

| mode | dependency_scope | actual_degree_reference |
| --- | --- | --- |
| `dependency_preserving` | `original_initial_variables` | `degree_wrt_original_initial_vars` |
| `range_only` | `current_step_box_variables_after_collapse` | `degree_wrt_current_step_box_vars_not_original_initial_vars` |

This means `range_only degree` and `range_only actual_degree=1` are not evidence that range-only retained only degree 1 dependency with respect to the original initial set. It means each collapsed step restarts with current-step box variables, so the degree is measured in the fresh local box variables.

The Van der Pol audit has 126 rows: 2 torch modes x 3 step sizes x 3 step counts x 7 orders. `range_only` endpoint degree stays 1 in its local variables. `dependency_preserving` reaches degree 7 at requested order 8 in multi-step cases. No segment-local `tau` variable remains active after endpoint substitution.

Torch and Flow* are aligned on ODE, initial set, step size, step count, and requested fixed order. Their internal Taylor-model dependency representation is not identical, so the comparison is a harness-level fixed-order comparison, not a coefficient-by-coefficient equivalence check.

## Width Semantics

The comparison CSV now separates three box meanings:

| field group | meaning |
| --- | --- |
| `endpoint_width_sum/max` | endpoint box at final time, when a true endpoint box is available |
| `last_segment_width_sum/max` | range box over the final flowpipe segment |
| `tube_width_sum/max` | hull over all parsed flowpipe segment boxes |

For torch, endpoint uses `final_tm.range_box()`, last segment uses the last segment Taylor model over `tau in [0,h]`, and tube is the hull over all segments.

For Flow*, GNUPLOT rectangles are flowpipe segment boxes. They are not endpoint boxes. Flow* endpoint boxes were not available in these artifacts. The current parser therefore leaves Flow* `endpoint_width_*` blank, marks `endpoint_box_available=False`, records the last parsed GNUPLOT segment in `last_segment_width_*`, and records the GNUPLOT hull in `tube_width_*`. Flow* rows use `box_source=flowstar_gnuplot_last_segment_and_tube` when plot boxes are parsed, and no endpoint ratio is plotted or claimed.

A brief Flow* API search found endpoint-related members such as `Result_of_Reachability::fp_end_of_time`, `tmv_fp_end_of_time`, `tmv_flowpipes`, and `evaluate_time`/`intEval` hooks. A reliable generated-C++ endpoint printer was not a small, low-risk harness edit, so endpoint extraction remains documented as unavailable from GNUPLOT.

## Runtime Semantics

Flow* rows now distinguish:

| field | meaning |
| --- | --- |
| `flowstar_internal_reach_s` | parsed `FLOWSTAR_RUNTIME_S`, i.e. Flow* reachability clock time printed by generated C++ |
| `flowstar_wall_compile_s` | Python wall time spent compiling the generated C++ case |
| `flowstar_wall_run_s` | Python wall time spent running the compiled case |
| `flowstar_wall_total_s` | compile plus run wall time |
| `runtime_s` for Flow* | internal reach time when available, otherwise wall total |
| `runtime_s` for torch | Python algorithm wall time |

## Flow* backend provenance and claim boundaries

The corrected Flow* rows use the `chenxin415/flowstar` toolbox C++ static-library backend at `FLOWSTAR_ROOT=/srv/local/shengenli/flowstar`. Generated benchmark programs include `Continuous.h`, call `ode.reach(...)`, and are compiled through the local runner against `flowstar-toolbox/libflowstar.a`; detailed backend, git, compiler, static-library, generated-case, and representative-artifact hashes are recorded in `outputs/flowstar_provenance_manifest.md` and `outputs/flowstar_provenance_manifest.json`.

The comparison remains a plant-only fixed-step/fixed-order baseline over polynomial ODEs. It does not compare raw Taylor-model coefficients, does not report endpoint ratios for Flow* GNUPLOT-derived boxes, does not represent Flow* adaptive or best-tuned performance, and does not reproduce the full CROWN-Reach NNCS pipeline. Current torch-vs-Flow* ratios are limited to matching `last_segment` and `tube` box semantics, with Flow* `runtime_s` sourced from `FLOWSTAR_RUNTIME_S` internal reach time when available and compile/run wall times reported separately.

## Van der Pol Decomposition

For `h=0.01`, `steps=10`, dependency-preserving is wider than range-only because two effects move in opposite directions. Higher order reduces remainder width sharply, but interval evaluation of the retained polynomial becomes the dominant source of looseness.

| order | mode | final width | polynomial range | remainder width | remainder frac | quality |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | dependency_preserving | 1.108536 | 0.450401 | 0.658135 | 0.593698 | remainder_dominated |
| 3 | dependency_preserving | 0.808607 | 0.450432 | 0.358175 | 0.442953 | polynomial_range_dominated |
| 4 | dependency_preserving | 0.801724 | 0.703102 | 0.098622 | 0.123013 | polynomial_range_dominated |
| 5 | dependency_preserving | 0.783200 | 0.710039 | 0.073162 | 0.093414 | polynomial_range_dominated |
| 6 | dependency_preserving | 0.769764 | 0.753662 | 0.016102 | 0.020919 | polynomial_range_dominated |
| 7 | dependency_preserving | 0.769151 | 0.759914 | 0.009238 | 0.012010 | polynomial_range_dominated |
| 8 | dependency_preserving | 0.767345 | 0.764985 | 0.002360 | 0.003076 | polynomial_range_dominated |

The reversal is therefore not simply "dependency preservation is worse." Low orders are remainder dominated because the nonlinear term `x^2*y` multiplies propagated remainder uncertainty. Higher orders reduce that remainder, but the larger retained polynomial is still bounded by ordinary interval range evaluation, which can loosen enough that the final width stays above range-only on this horizon.

## Flow* Setting Sweep

The previous report was wrong to imply only orders 2 and 3 failed. Under the strict old setting, orders 2-6 all failed for every tested horizon, order 7 completed only 3/9 cases, and order 8 completed 6/9 cases.

Flow* status counts over 63 Van der Pol cases per setting:

| setting_label | completed | failed |
| --- | ---: | ---: |
| `rem1e-10_cut1e-15` | 9 | 54 |
| `rem1e-8_cut1e-15` | 24 | 39 |
| `rem1e-6_cut1e-12` | 38 | 25 |
| `rem1e-4_cut1e-10` | 51 | 12 |

Status by order, each cell is `completed/9`:

| setting_label | o2 | o3 | o4 | o5 | o6 | o7 | o8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rem1e-10_cut1e-15` | 0 | 0 | 0 | 0 | 0 | 3 | 6 |
| `rem1e-8_cut1e-15` | 0 | 0 | 0 | 0 | 6 | 9 | 9 |
| `rem1e-6_cut1e-12` | 0 | 0 | 3 | 8 | 9 | 9 | 9 |
| `rem1e-4_cut1e-10` | 0 | 6 | 9 | 9 | 9 | 9 | 9 |

Failure below order 4 is setting-dependent for order 3: it fails under the first three settings but completes 6/9 cases under `rem1e-4_cut1e-10`. Order 2 failed in all settings tested. Orders 4-6 were highly setting-dependent: strict settings made them fail, while looser settings often completed.

The Flow* last-segment widths at `h=0.01`, `steps=10` are about `0.65423` for completed orders/settings, with tube widths about `1.05781`. These are last-segment/tube GNUPLOT boxes, not endpoint boxes.

## Generated Artifacts

Primary CSVs:

- `outputs/tm_order_audit_vdp_order2_8.csv`
- `outputs/van_der_pol_diagnostics_by_order_v2.csv`
- `outputs/flowstar_vdp_by_order_rem1e-10_cut1e-15_v2.csv`
- `outputs/flowstar_vdp_by_order_rem1e-8_cut1e-15_v2.csv`
- `outputs/flowstar_vdp_by_order_rem1e-6_cut1e-12_v2.csv`
- `outputs/flowstar_vdp_by_order_rem1e-4_cut1e-10_v2.csv`
- `outputs/flowstar_vdp_remainder_cutoff_sweep.csv`

Plots:

- `outputs/van_der_pol_endpoint_width_vs_order.png`
- `outputs/van_der_pol_last_segment_width_vs_order.png`
- `outputs/van_der_pol_tube_width_vs_order.png`
- `outputs/van_der_pol_runtime_internal_vs_order.png`
- `outputs/van_der_pol_runtime_wall_vs_order.png`
- `outputs/van_der_pol_remainder_frac_vs_order.png`
- `outputs/van_der_pol_poly_vs_remainder_stacked_by_order.png`
- `outputs/flowstar_status_by_order_and_setting.png`
- `outputs/torch_over_flowstar_last_segment_width_ratio_by_order.png`
- `outputs/torch_over_flowstar_tube_width_ratio_by_order.png`
- `outputs/order_flowstar_status_table.md`

Auxiliary generated files:

- `outputs/flowstar_vdp_torch_h0.01_s10_v2.csv`
- `outputs/flowstar_vdp_plot_input_v2.csv`
- `scripts/run_vdp_order_flowstar_study.sh`

## Next Work

The highest-leverage follow-ups are symbolic remainder support, centered or normalized variables, Bernstein or subdivision range bounding for retained polynomials, an explicit maximum remainder budget, and a separate Flow*_adaptive baseline. Those are intentionally outside this fixed-order diagnostic pass.
