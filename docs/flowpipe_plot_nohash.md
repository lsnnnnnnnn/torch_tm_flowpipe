# No-hash native flowpipe plotting

`python -m torch_tm_flowpipe.flowpipe_plot_nohash` reads a saved native
`ranges.bin` or an existing geometry JSON. It writes `.geometry.json` for
native input, `.m`, `.png`, `.pdf`, and `.render.json`. This entry never runs a
solver or computes a content digest. The receipt records file paths, byte
sizes, the plot configuration, and an optional JSON run configuration.
When an adjacent `RESULT.json` exists, its saved status is shown in the figure
footer as **unbound** adjacent evidence.

For the continuous ARCH-COMP 2026 Double Pendulum less-robust contract, use the
full 100 local ODE steps (`20` control periods × `5` substeps), even when the
saved binary contains only a prefix:

```console
PYTHONPATH=src python -m torch_tm_flowpipe.flowpipe_plot_nohash \
  --ranges /path/to/new-run/ranges.bin \
  --origin-path /original/host/path/to/new-run/ranges.bin \
  --label 'new native DP less-robust' \
  --benchmark 'Double Pendulum' \
  --instance-id double-pendulum-less-robust \
  --coordinate-names theta1,theta2,theta1_dot,theta2_dot \
  --projection t,theta1 --view tube \
  --step-size 0.01 --expected-steps 100 --expected-lanes 225 \
  --spec benchmarks/plot_specs/double_pendulum_less_robust_2026_nohash.json \
  --run-config /path/to/new-run/START.json \
  --output /path/to/plots/dp_t_theta1_tube
```

Use `--projection theta1,theta2` for a state-state plot. It keeps one box per
recorded lane at each displayed step. The spec draws the full initial box and
the Safe band on every physical coordinate for **all continuous time
`t∈[0,1]`**. `--display-steps 1,5,25,50,75,100` changes only display density;
the geometry still scans the entire binary and reports every missing or
partial step. `--partial-policy mark` draws incomplete record steps with
dashed boxes. It does not imply acceptance.

`ranges.bin` uses little-endian records
`<lane:uint64,step:uint64,h:float64,4×state_count float64>` with each state in
`tube_lo,tube_hi,endpoint_lo,endpoint_hi` order. The parser checks record
size, lane/step uniqueness, finite ordered intervals, `h`, and explicit lane
and horizon limits. **The binary has no accepted/status field.** A frame with
`complete=true` means its expected number of range records is present; it
does not establish solver acceptance, safety, or a certificate. The declared
spec, `--run-config` file, and binary remain unbound to one another in this
plot receipt. Record the actual source path and run identity separately if
files have been copied from another host; `--origin-path` records that
declared path without claiming byte identity.

To redraw existing geometry without recalculating source or output hashes:

```console
PYTHONPATH=src python -m torch_tm_flowpipe.flowpipe_plot_nohash \
  --geometry /path/to/saved.geometry.json --output /path/to/plots/redraw
```

Archived QUAD root1 B2 data is a parser/render regression fixture only. Its
`x3` band is a **target at endpoint `t=5`**, not a Safe region; neither the
old data nor a redraw becomes a 2026 result by changing a plot title. The
new Double Pendulum plots use the separate all-time Safe spec above.

The 2026 TORA remain native full run supplies another real example: its
[run summary](evidence/results/archcomp26_20261001/native_tora_remain_full20_001/SUMMARY.md)
links a complete `t,x1` tube figure, an `x1,x2` endpoint figure, both MATLAB
scripts, and the source `ranges.bin`. Use
`benchmarks/plot_specs/tora_remain_2026_nohash.json`, `--coordinate-names
x1,x2,x3,x4`, `--step-size 0.1`, `--expected-steps 200`, and
`--expected-lanes 12`. The figures show the full official initial box and
the all-time `[-2,2]^4` Safe region. The native run has a separate completed
process receipt and independent saved-range scan; the plotting entry itself
continues to label acceptance as unbound to the binary records.

The completed 2026 paper-equation QUAD comparison has a separate
[four-method `t,x3` figure and source guide](evidence/results/archcomp26_20261001/quad_paper_fourway_saved_20261002/SUMMARY.md).
Its full-time curves come from all 1,024 boxes at each of 1,000 saved steps
for native Flow* and P3. Huan and Xiangru saved only terminal bounds, so their
two methods appear as `T=5` endpoint intervals; no intermediate curve is
interpolated. The `[0.94,1.06]` band applies to the terminal target only.
The native whole-step bounds are axis-aligned box projections, not octagon
support values. The included MATLAB script is generated but has not been run
in MATLAB or Octave.
