# Python rendering of saved native flowpipes

`python -m torch_tm_flowpipe.flowpipe_plot_nohash` reads a saved native
`ranges.bin` or one or more existing geometry JSON files. It writes
`.geometry.json` for native input or a multi-geometry overlay, plus Python
Matplotlib-rendered `.png` and `.pdf` files and a `.render.json` receipt.
The CLI produces no MATLAB `.m` file. This entry never runs a solver or
computes a content digest. The receipt records file paths, byte sizes, plot
configuration, and separate source/geometry and Python-render wall times.
Solver time is not measured by this entry. An
optional JSON run configuration and adjacent `RESULT.json` remain unbound
declarations; the receipt performs no model or run identity check.
When an adjacent `RESULT.json` exists, its saved status is shown in the figure
footer as **unbound** adjacent evidence.

The `flowpipe_plot` and `tm_octagon_nohash` CLIs likewise render Python PNG/PDF
without automatic `.m` output. `flowpipe_plot --geometry` records artifact
paths and byte sizes without computing digests; its legacy source-export
identity checks remain unchanged and are outside this no-digest workflow.
The octagon geometry/stream redraw paths do not import the numerical solver.
The older `tools/verify_flowpipe_plot_artifacts.py` requires digest-bearing
receipts and is not an acceptance tool for the new path/byte-size receipts.
The [2026-10-04 saved-data acceptance](evidence/results/python_plot_cli_cleanup_20261004_001/README.md)
checks all three CLI paths with digest computation and MATLAB export blocked.

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

To redraw existing v1 informational geometry without rereading the range file:

```console
PYTHONPATH=src python -m torch_tm_flowpipe.flowpipe_plot_nohash \
  --geometry /path/to/saved.geometry.json --output /path/to/plots/redraw
```

Repeat `--geometry` to overlay methods already exported under the same v1
informational plot contract, without reading range files again:

```console
PYTHONPATH=src python -m torch_tm_flowpipe.flowpipe_plot_nohash \
  --geometry /path/to/method_a.geometry.json \
  --geometry /path/to/method_b.geometry.json \
  --output /path/to/plots/same_axis_comparison
```

The inputs must agree exactly on benchmark and instance, coordinate order,
projection, tube/endpoint view, step size and horizon, initial/property plot
spec, and partial-record policy. Series labels must be unique. The combined
`.geometry.json` retains each method's source path, recorded-step coverage,
and unbound run evidence; a missing step remains missing for that method.
This entry accepts only v1 informational geometry on redraw, so it does not
validate declared source digests from other plot-spec versions. The
[small synthetic overlay fixture](evidence/results/flowpipe_plot_nohash_overlay_20261003_001/README.md)
shows two visibly overlaid saved-range series, one with an unobserved second
step, together with initial/Safe/endpoint Target layers and the generated
Python PNG/PDF artifacts. The fixture is a renderer check, not a reachable-set
or solver result.

Archived QUAD root1 B2 data is a parser/render regression fixture only. Its
`x3` band is a **target at endpoint `t=5`**, not a Safe region; neither the
old data nor a redraw becomes a 2026 result by changing a plot title. The
new Double Pendulum plots use the separate all-time Safe spec above.

The 2026 TORA remain native full run supplies another real example: its
[run summary](evidence/results/archcomp26_20261001/native_tora_remain_full20_001/SUMMARY.md)
links a complete Python-rendered `t,x1` tube figure, an `x1,x2` endpoint
figure, and the source `ranges.bin`. Use
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
support values. The PNG/PDF were rendered by Python Matplotlib. A legacy
`.m` file was also written automatically but was not used.

## Actual Taylor-model eight-direction projection

`torch_tm_flowpipe.tm_octagon_nohash.export_segments` now accepts **validated
segments while their correlated Taylor models still exist**. It ranges `x`,
`y`, `x+y`, and `x-y` as Taylor models, retaining shared dependency symbols,
and saves all eight lower/upper half-plane bounds for both tube and the
accepted `final_tm` endpoint used for propagation. The latter includes
symbolic output remainders added after raw endpoint substitution in applicable
normalized-insertion modes. Polygon vertices are computed later in ordinary floating point for
drawing only; the saved directional bounds are the numerical data. The entry
rejects missing or unvalidated segments. It never derives diagonal bounds
from an axis-aligned observer or computes a content digest.

The isolated [three-step harmonic-oscillator smoke](evidence/results/archcomp26_20261001/tm_octagon_harmonic_smoke_20261002_001/SUMMARY.md)
uses the existing CPU Taylor-model solver, `x1'=x2`, `x2'=-x1`,
`x1(0)∈[1,1.2]`, `x2(0)=0`, `h=.05`, order 4. All three plant-only steps
validated to `T=.15`. The saved third endpoint octagon's display area is
`0.848866` of its axis-aligned box area; 108 sampled exact-solution cases fell
inside every saved directional interval. The check is a finite sample and
does not replace solver validation or establish an NNCS certificate.
The saved smoke is an immutable snapshot of the initial exporter, which used
`endpoint_raw_tm`; the current exporter uses `final_tm` for subsequent runs.
That change affects endpoint semantics where fixed-time tightening or symbolic
output materialization is active. The saved directional JSON remains labeled
as the original raw-endpoint smoke and is not silently rewritten.

```console
PYTHONPATH=src python -m torch_tm_flowpipe.tm_octagon_nohash \
  --smoke-harmonic --output out/harmonic_3step
PYTHONPATH=src python -m torch_tm_flowpipe.tm_octagon_nohash \
  --geometry out/harmonic_3step.geometry.json --view endpoint \
  --output out/harmonic_3step_endpoint
```

For an actual application, call `export_segments(result.segments,
initial_box, output, x=..., y=..., names=(..., ...))` at the accepted solver
boundary. The second command redraws the saved geometry without any numerical
step. The demonstration produced `.geometry.json` and Python-rendered PDF/PNG
on the saved research server; the CLI also emitted an unused legacy `.m` file.
The existing 2026 QUAD/native/P3 saved ranges contain no correlated TM,
so their figures remain explicitly box projections until a new accepted-step
directional observer is attached to those engines; the old long runs are not
repeated just for plotting.

The CPU multi-step solver now has an optional `accepted_segment_observer`
callback. `OctagonJSONLObserver` writes the actual directional supports after
each validated step, while `segment.tm` and the propagated `segment.final_tm`
still exist; it stops on an observation error. The solver still retains its
segment list in memory. Replaying the JSONL through
`geometry_from_stream` produces an **accepted-prefix** plot. The separate
[three-step streaming smoke](evidence/results/archcomp26_20261001/tm_octagon_stream_harmonic_smoke_20261002_001/SUMMARY.md)
and an unobserved same-contract control both validated 3/3 steps and had
identical tube and endpoint directional supports. This path is CPU scalar TM
only. Native Flow* would need a C++ observer over each accepted `Flowpipe`'s
correlated Taylor models before `arch_ranges::record` reduces them to boxes;
the P3 GPU driver would need a per-lane directional range before its current
`hull_ranges_s`/endpoint range union. Neither adapter is present, and no
saved native/P3 `ranges.bin` or JSONL row can be relabeled as an octagon.

An isolated [native Flow* eight-direction production gate](ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)
located the C++ accepted-step interface and matched the original MATLAB
plotter's tube support values, but its order-4 harmonic probe failed analytic
containment: 28 violations among 1,512 finite exact-sample/direction checks.
An additional actual paper-equation QUAD symbolic-remainder first-box,
first-`0.005 s`-step gate found `x7/x8` terminal bounds missing numerical
trajectories from the CROWN affine-plus-residual relaxed control set by up to
`1.98259e-6`; its sampled `x1/x2` directions passed. An isolated copied
library's refinement-skip mechanism check removed finite-sample violations
but widened bounds substantially and is not a production fix. This blocks
native octagon promotion pending a sound repaired propagation path. The
sampled control has not been established as the actual NN output, and the
existing QUAD box plots and original run receipts remain unchanged.
