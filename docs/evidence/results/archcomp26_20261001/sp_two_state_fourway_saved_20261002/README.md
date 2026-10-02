# Single Pendulum named two-state profile: four saved full runs

This figure reads the already saved 100 accepted `0.01 s` substeps of each
method. It does not launch a solver or change any run. The **named two-physical-
state profile** is distinct from an official three-state MATLAB reproduction:
the official third-state initial/reset and closed-loop entry remain missing.

Sources: [P3 range JSONL](../single_pendulum_two_state_p3_full20_001/data/ranges.jsonl),
[Huan range JSONL](../single_pendulum_two_state_huan_full20_001/data/ranges.jsonl),
[Xiangru range JSONL](../single_pendulum_two_state_xiangru_full20_001/data/ranges.jsonl),
and [native binary ranges](../native_sp_two_state_full20_001/ranges.bin).
The [source script](../../../../../tools/archcomp26_sp_fourway_saved_figure_nohash.py)
checks each recorded status, complete 100-step sequence, finite ordered tube
and endpoint intervals, endpoint containment, and the saved `x1∈[0,1]` tube
condition for all 50 steps in the closed property window `t∈[0.5,1]`. It also
confirms Huan and Xiangru saved intervals are directly equal. This is an
audit of saved numerical outputs, not an independent enclosure proof.

The [single overlaid figure](fourway_saved_bounds.png) also has a [PDF](fourway_saved_bounds.pdf).
Upper panels show both tube boundaries for `x1` and `x2` on the same axes.
Lower panels show each saved endpoint boundary minus Huan's corresponding
boundary on `t=0.8–1`; solid lines are lower bounds and dashed lines upper
bounds. Huan/Xiangru coincide at zero in those panels. The green `x1` shading
is the target interval; the vertical mark at `t=0.5` starts the property
window. The [800-row saved interval table](saved_tubes.csv) and [eight-row
absolute bounds and widths](endpoint_and_window_widths.csv) give values
without graphical overdraw. A [MATLAB plotting example](fourway_saved_bounds.m)
uses that same CSV, but has not been executed in MATLAB.

At `T=1`, the saved `x1` endpoint widths are P3 `0.1416718662`, Huan and
Xiangru `0.1391393522`, and native `0.1391393524`; corresponding `x2` widths
are `0.1469736372`, `0.1444355459`, `0.1444355459`, and `0.1444355487`.
These are one run per method and different numerical implementations. Neither
small width differences nor completed saved ranges establish a stable speed
ranking or an independent end-to-end floating-point NNCS certificate.

Regenerate locally with:

```bash
MPLCONFIGDIR=/private/tmp/archcomp26_sp_matplotlib_cache /opt/anaconda3/bin/python3 tools/archcomp26_sp_fourway_saved_figure_nohash.py
```
