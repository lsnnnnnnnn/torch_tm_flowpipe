# October 6 final selected-run figures

Nine Python-generated vector PDFs and matching PNGs. All nine final PNGs were individually reviewed and passed; [ALL_FIGURES_QA.json](ALL_FIGURES_QA.json) records the combined result, dimensions, paths and byte sizes.

| Figure | PNG | PDF |
|---|---|---|
| Eight selected internal-driver time pairs | [PNG](driver_selected_before_after.png) | [PDF](driver_selected_before_after.pdf) |
| TORA sigmoid: four-state endpoint widths | [PNG](tora_sigmoid_endpoint_widths.png) | [PDF](tora_sigmoid_endpoint_widths.pdf) |
| TORA sigmoid: four-state per-step tube widths | [PNG](tora_sigmoid_tube_widths.png) | [PDF](tora_sigmoid_tube_widths.pdf) |
| TORA tanh: four-state endpoint widths | [PNG](tora_tanh_endpoint_widths.png) | [PDF](tora_tanh_endpoint_widths.pdf) |
| TORA tanh: four-state per-step tube widths | [PNG](tora_tanh_tube_widths.png) | [PDF](tora_tanh_tube_widths.pdf) |
| QUAD: twelve-state endpoint widths | [PNG](quad_endpoint_widths.png) | [PDF](quad_endpoint_widths.pdf) |
| QUAD: twelve-state per-step tube widths | [PNG](quad_tube_widths.png) | [PDF](quad_tube_widths.pdf) |
| QUAD: time-height tube bounds | [PNG](quad_time_x3.png) | [PDF](quad_time_x3.pdf) |
| QUAD: pooled axis-aligned x1-x2 projection | [PNG](quad_x1_x2.png) | [PDF](quad_x1_x2.pdf) |

## Current selection and sources

The timing figure compares the eight final October 6 selections with the previous report choices. Each pair has its own scale. Exact recorded seconds and raw receipt fields remain in [FIGURE_SOURCES.json](FIGURE_SOURCES.json); the slower sigmoid choice is retained. Single shared-host samples do not establish a stable speed ranking.

The four TORA figures show all four physical states and all 500 saved steps for selected P3, Huan, Xiangru and native. State scales remain separate, coincident curves are not displaced, and callouts are terminal endpoint widths or maximum per-step tube widths. The selected P3 sigmoid and tanh runs both use cutoff 1e-8. [FIGURE_QA.json](FIGURE_QA.json) records their validation.

All four QUAD figures use `quad_paper_joint_full1000_001`, the final selected joint run, read from its own saved observations and final width-comparison receipt. They do not substitute the parent control-only run. [QUAD_SOURCES.json](QUAD_SOURCES.json) includes the actual source paths, joint qualification, contract, raw receipts, terminal values and per-step maxima; [QUAD_QA.json](QUAD_QA.json) records all four final visual reviews and 96,000 direct bound-field matches.

QUAD width plots show twelve states, three complete 1000-step curve sets, and Huan/Xiangru terminal-only endpoint markers. Time-height uses each saved tube bound on its substep and shows the x3 target [0.94,1.06] only at T=5. The x1-x2 view contains 1000 pooled coordinate rectangles for each available full method, with identical axes, initial boxes and terminal boxes. These axis-aligned boxes preserve neither per-lane nor octagon correlations. Huan/Xiangru terminal driver hulls remain distinct saved objects from P3/native pooled-observer curves. No author tube or trajectory is invented.

## Rebuild

The common plotter defaults to the formal report package: [widths_long.csv](../widths/widths_long.csv), [summary.json](../widths/summary.json), [sources.json](../widths/sources.json) and [timing_index.json](../timing/timing_index.json). Publishing duplicate `report_data/current/widths_long.csv/json` files is unnecessary. Selection is read from `research/p3_speed_tightness_20261006/report_data/selection.json`. The optional `--data` argument also accepts a separately rebuilt flat report-data folder.

From the repository root, with Python ReportLab and Poppler installed:

```sh
python3 -B tools/plot_archcomp26_optimization_20261006.py
python3 -B tools/plot_archcomp26_quad_optimization_20261006.py --candidate-run research/p3_speed_tightness_20261006/results/quad_paper_joint_full1000_001
```

This machine used `/Users/shengenli/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`; `--pdftoppm` can select an installed Poppler renderer. The scripts read saved data only. Python ReportLab creates PDFs with literal document IDs and disabled signature accumulators; prohibited digest constructors raise. Poppler renders the exact produced PDFs. No numerical solver or old checker is executed.

The twelve-panel QUAD PDFs are intended for zoomable reading; retain their full-size links and numerical tables when placing a reduced image in a report. Per-step tube width is not the width of the all-time union. Smaller interval width is neither containment nor an independent NNCS certificate.
