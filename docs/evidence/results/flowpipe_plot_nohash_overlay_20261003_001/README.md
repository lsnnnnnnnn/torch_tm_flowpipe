# No-hash saved-geometry overlay fixture — 2026-10-03

This is a **synthetic plotting fixture**, not an ODE/NNCS run or an
ARCH-COMP instance result. The two tiny `ranges.bin` files contain one lane:
`source_1` has only step 1/2 and `source_2` has steps 1–2/2 at `h=0.125`.
Both were exported separately through the no-hash native entry using the same
v1 [informational plot spec](synthetic_plot_spec.json), then their saved
geometries were overlaid without reading the ranges again. No solver or
content-digest operation was invoked.

[Combined PNG](overlay_t_x1_tube.png) · [PDF](overlay_t_x1_tube.pdf) ·
[MATLAB script](overlay_t_x1_tube.m) ·
[geometry JSON](overlay_t_x1_tube.geometry.json) ·
[render receipt](overlay_t_x1_tube.render.json).

The combined geometry retains both method labels and distinct coverage:
`A saved prefix` has `projection_unobserved_step_ranges=[[2,2]]`, while
`B saved horizon` has none. The legend labels record presence and acceptance
unknown, and the footer does not infer a solver failure for A's missing step.
The initial box, all-time Safe band, and endpoint-only Target at `t=0.25`
appear separately. The PNG was visually inspected and the PDF is one page.
MATLAB/Octave was unavailable locally, so the `.m` file was generated and
read statically, without a runtime claim.

The focused check ran:

```console
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/codex-plot-overlay-mpl-20261003 \
  PYTHONPATH=src /opt/anaconda3/bin/python3 -m unittest -v \
  tests/test_flowpipe_plot_nohash_overlay.py
```

Result: **1 test passed**. It also verifies refusal on duplicate series labels,
different plot specs, and an output path that would replace an input geometry.
The fixture is regenerated with
`tests.test_flowpipe_plot_nohash_overlay.build_fixture(Path(output_dir))`.
The receipt separates geometry-validation, MATLAB-generation, and Matplotlib
render wall times, and explicitly states that solver timing and run identity
were not checked. Those wall values describe this tiny render only.
