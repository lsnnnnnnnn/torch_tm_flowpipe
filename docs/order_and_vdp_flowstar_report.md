# Order semantics, Van der Pol diagnostics, and Flow* fixed-order comparison

## Scope

This report is limited to diagnostics, experiment harnesses, plots, and documentation for the existing `torch_tm_flowpipe` Taylor-model prototype.  It does not add CROWN-Reach, Flow* bindings, adaptive Flow*, hybrid transitions, branch-and-bound, sin/cos support, or sensitivity/Jacobian bounds.

The Van der Pol benchmark is:

```text
dx/dt = y
dy/dt = y - x - x^2*y
x0 in [1.1, 1.4], y0 in [2.35, 2.45]
```

## Reproduction

Use the `py11` conda environment:

```bash
conda run -n py11 python -m pip install -e '.[test]'
conda run -n py11 pytest -q
conda run -n py11 python experiments/tm_order_audit.py --all --csv outputs/tm_order_audit_all_systems.csv
conda run -n py11 python experiments/tm_order_audit.py --system van_der_pol --orders 2 3 4 5 6 7 8 --h 0.0025 0.005 0.01 --steps 1 5 10 --csv outputs/tm_order_audit.csv
conda run -n py11 python experiments/diagnose_van_der_pol.py --orders 2 3 4 5 6 7 8 --h-values 0.0025 0.005 0.01 --steps-values 1 5 10 --csv outputs/van_der_pol_diagnostics_by_order.csv
conda run -n py11 python comparisons/flowstar/compare_against_torch_tm.py --systems van_der_pol --orders 2 3 4 5 6 7 8 --h-values 0.0025 0.005 0.01 --steps-values 1 5 10 --flowstar-root /srv/local/shengenli/flowstar --csv outputs/flowstar_comparison_by_order.csv --flowstar-timeout-s 120 --no-plots
conda run -n py11 python experiments/plot_order_results.py --diagnostics-csv outputs/van_der_pol_diagnostics_by_order.csv --comparison-csv outputs/flowstar_comparison_by_order.csv --out-dir outputs --h 0.01 --steps 10
```

If `FLOWSTAR_ROOT` is not exported, either pass `--flowstar-root /path/to/flowstar` as above or run:

```bash
export FLOWSTAR_ROOT=/path/to/flowstar
```

## Taylor order audit

The torch `order` parameter is a requested total-degree cutoff.  Actual retained endpoint degree can be lower than the requested order because the dynamics, truncation, and endpoint substitution may not populate all degrees.

For Van der Pol over the requested grid, `outputs/tm_order_audit.csv` has 126 rows: two torch modes times 3 step sizes times 3 step counts times 7 requested orders.  The audit records `order_semantics=total_degree_cutoff`, final and flowpipe widths, actual degree, term counts, active variables, remainder radii, runtime, and whether local segment `tau` leaked after endpoint substitution.

Observed behavior:

| mode | actual endpoint degree pattern |
| --- | --- |
| range_only | final endpoint polynomials stay degree 1 for all requested orders 2-8, because each step collapses to a box and restarts with identity TMs |
| dependency_preserving | reaches degree 7 at requested order 8 for multi-step cases; requested order is a cutoff, not a guarantee of equal actual degree |

No `segment_tau_active_after_drop` cases were observed in the generated audit CSV.

## Van der Pol decomposition

For h=0.01 and steps=10, the diagnostic decomposition shows why dependency-preserving can look backwards:

| order | mode | final width sum | polynomial range width | remainder width | actual degree | term counts | runtime s |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 4 | range_only | 0.730158 | 0.730158 | ~0 | 1 | [1, 1] | 1.31 |
| 4 | dependency_preserving | 0.801724 | 0.703102 | 0.098622 | 3 | [6, 6] | 3.71 |
| 5 | range_only | 0.729515 | 0.729515 | ~0 | 1 | [1, 1] | 2.94 |
| 5 | dependency_preserving | 0.783200 | 0.710039 | 0.073162 | 3 | [6, 6] | 7.12 |
| 6 | range_only | 0.719083 | 0.719083 | ~0 | 1 | [1, 1] | 5.51 |
| 6 | dependency_preserving | 0.769764 | 0.753662 | 0.016102 | 5 | [12, 12] | 15.64 |
| 7 | range_only | 0.718068 | 0.718068 | ~0 | 1 | [1, 1] | 9.16 |
| 7 | dependency_preserving | 0.769151 | 0.759914 | 0.009238 | 5 | [12, 12] | 26.78 |
| 8 | range_only | 0.718028 | 0.718028 | ~0 | 1 | [1, 1] | 14.64 |
| 8 | dependency_preserving | 0.767345 | 0.764985 | 0.002360 | 7 | [20, 20] | 47.43 |

The hypothesis is supported but refined.  Higher order does reduce dependency-preserving remainder width sharply, from 0.0986 at order 4 to 0.00236 at order 8.  However, the interval evaluation of the larger retained polynomial grows looser, so polynomial range width increases from 0.703 to 0.765.  Range-only avoids propagated remainder multiplication by collapsing to a fresh box each step; it loses symbolic dependency, but its final range remains narrower on this horizon.

All torch rows in `outputs/flowstar_comparison_by_order.csv` are `validated` and had zero sampling containment failures.

## Flow* fixed-order comparison

The Flow* toolbox was found at `/srv/local/shengenli/flowstar`; `FLOWSTAR_ROOT` was not exported in the shell.  The harness compiles plant-only C++ cases against `flowstar-toolbox/libflowstar.a`, uses fixed step size and fixed Taylor order, and writes Flow* rows as `tool=flowstar, mode=fixed`.

Final Flow* status counts over 63 Van der Pol cases:

| status | count | meaning |
| --- | ---: | --- |
| failed | 54 | Flow* reported `FLOWSTAR_COMPLETED 0`, usually with `Flowpipe computation is terminated due to the large overestimation.` |
| completed | 9 | Flow* completed and generated parseable plot intervals |

Completed Flow* cases were only:

| order | h | steps | final width sum |
| ---: | ---: | ---: | ---: |
| 7 | 0.0025 | 1 | 0.415557 |
| 8 | 0.0025 | 1 | 0.415557 |
| 7 | 0.0025 | 5 | 0.438618 |
| 8 | 0.0025 | 5 | 0.438618 |
| 7 | 0.0025 | 10 | 0.466777 |
| 8 | 0.0025 | 10 | 0.466777 |
| 8 | 0.005 | 1 | 0.431219 |
| 8 | 0.005 | 5 | 0.477538 |
| 8 | 0.005 | 10 | 0.532188 |

Flow* did have problems below order 4: every order-2 and order-3 case failed explicitly.  Orders 4-6 also failed on this generated fixed-order setup.  This does not prove Flow* cannot handle Van der Pol in general; it says this fixed-order plant-only harness with the current generated settings fails by large overestimation for most cases and only completes at high order on the smaller horizons.

## Generated deliverables

Primary files:

- `outputs/tm_order_audit.csv`
- `outputs/van_der_pol_diagnostics_by_order.csv`
- `outputs/flowstar_comparison_by_order.csv`
- `outputs/van_der_pol_width_vs_order.png`
- `outputs/van_der_pol_runtime_vs_order.png`
- `outputs/van_der_pol_remainder_vs_order.png`
- `outputs/torch_over_flowstar_width_ratio_by_order.png`
- `outputs/order_flowstar_status_table.md`
- `docs/order_and_vdp_flowstar_report.md`

Additional useful files:

- `outputs/tm_order_audit_all_systems.csv`
- `outputs/flowstar_comparison_by_order_summary.md`
- `outputs/flowstar_status_by_order.png`
- `outputs/dependency_vs_range_ratio_by_order.png`
