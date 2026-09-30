# Flowpipe projection export and plotting

`python -m torch_tm_flowpipe.flowpipe_plot` reads saved `observer_<step>.pt`
files, exports standalone projection geometry, and writes a MATLAB script plus
PNG/PDF renderings. It never reruns the solver.

The current observer contract is `bounds[lanes, states, 4]` with the exact
column order `tube_lo,tube_hi,endpoint_lo,endpoint_hi`. Tube bounds cover one
local integration step; endpoint bounds are evaluated at the end of that step.
The exporter verifies shape, accepted-lane finiteness and interval ordering.

Example for archived QUAD observer samples:

```console
python -m torch_tm_flowpipe.flowpipe_plot \
  --benchmark QUAD \
  --series 'PyTorch P3=/path/to/full1024_p3_gpu14_1000_v1' \
  --projection t,x3 \
  --view tube \
  --step-size 0.005 \
  --expected-steps 1000 \
  --spec benchmarks/plot_specs/quad_legacy_author_contract.json \
  --output out/quad_t_x3_tube
```

The outputs are:

- `quad_t_x3_tube.geometry.json`: geometry, source hashes/identity when
  available, accepted counts, status counts, and separate numerical-horizon,
  saved-projection, and display coverage;
- `quad_t_x3_tube.m`: self-contained MATLAB plotting script;
- `quad_t_x3_tube.png` and `.pdf`: noninteractive Matplotlib renderings.
- `quad_t_x3_tube.render.json`: per-stage export/script/render wall times,
  output hashes and sizes, and the available run-timing evidence.

Use `--geometry saved.geometry.json --output new/path` to redraw without the
observer files. Multiple `--series LABEL=DIR` arguments overlay methods with
the same projection and view. A state-state projection such as `x1,x2` keeps
one rectangle per accepted lane instead of constructing a false cross-lane
Cartesian product.

This is deliberately an **axis-aligned box projection**, not Flow*'s octagon.
`--expected-steps` and `--step-size` are mandatory for source export: there is
no QUAD-specific default that can silently fabricate another benchmark's time
axis. The exporter never treats the largest saved observer number as proof
that the numerical horizon ended there. `observed_steps` means projection files/records exist,
`displayed_steps` means they were drawn, and `solver_run_evidence` separately
hashes and summarizes adjacent `RESULT.json`/`steps.jsonl` when present. An
unobserved projection step is never labelled a solver rejection or timeout by
inference. Nothing is interpolated.

Adjacent `INPUT.json` and `RESULT.json` are hash-linked only when the run
recorded a matching `RESULT.input_sha256`. Step size, horizon, and lane count
retain per-field declaration sources and are labelled identity-bound only
when the declaring INPUT/RESULT artifact itself binds to every observer
sidecar. An identity-bound RESULT cannot promote a field from an unrelated
INPUT. Run status has its own RESULT binding and is shown separately in the
figure footer. `steps.jsonl` timing has another independent binding:
`advance_s` is bound to the plotted series only when an identity-bound RESULT
explicitly hashes that file. Adjacency alone is recorded as unbound. Observer
lane count must remain constant across every saved file, so
a later pruned survivor set cannot silently become a complete frame.

A partially accepted frame is rejected by default; `--partial-policy mark`
exports accepted-only geometry with `complete=false`, retained lane IDs on
state-state plots, and dashed partial styling. For large state-state outputs,
select saved display steps explicitly, for example
`--display-steps 20,100,580,800,1000`. This changes only visualization and is
recorded in the geometry; it does not change numerical verification.
Known partial steps omitted only from display remain counted in the figure
footer. Native `ranges.bin` rows are always labelled as record coverage with
acceptance unknown; they are never presented as accepted lanes.

The bundled QUAD spec is deliberately named `quad_legacy_author_contract`.
It encodes the archived author-reproduction initial set and
`0.94 <= x3 <= 1.06` as a target only at `t=5 s`, not as a safety band over the
take-off trajectory. It is explicitly unbound and must not be presented as
the unresolved ARCH-COMP 2026 QUAD contract.
An explicitly supplied spec must decode to a JSON object; `null`, arrays, and
other JSON types fail closed rather than being treated as no spec.
Safe, Target, Unsafe, and informational regions have distinct stable styles.
A region that cannot be represented in the selected projection is listed in
the footer instead of disappearing silently. Every timed region must lie
inside the full declared numerical horizon, even when it is not drawable in
the selected projection.

MATLAB execution is not implied by script generation. Record the MATLAB or
Octave version and result separately when that environment is available.
Neither MATLAB nor Octave was installed in the 2026-10-01 local validation
environment, so the current evidence covers script generation/static checks
and the Matplotlib render, not a claimed MATLAB runtime pass.

The frozen native `ranges.bin` v1 format is also supported when the series
value names that file and `--expected-lanes` is explicit:

```console
python -m torch_tm_flowpipe.flowpipe_plot \
  --benchmark QUAD \
  --series 'native root1 B2=/path/to/ranges.bin' \
  --expected-lanes 2 --expected-steps 1000 \
  --projection t,x3 --view tube --step-size 0.005 \
  --spec benchmarks/plot_specs/quad_legacy_author_contract.json \
  --output out/native_root1_b2_t_x3
```

That binary contains record presence but no acceptance/status field, so the
geometry records `accepted_lanes=null` and never manufactures acceptance from
the range rows. Adjacent `INPUT.json` and `RESULT.json` hashes and selected
scope flags are copied into provenance when present. The archived root1 B2
run is not the original full-1024 experiment and must be labelled accordingly.

The 2026-10-01 archived-data validation, exact artifact hashes, and remaining
limitations are recorded in `docs/FLOWPIPE_PLOT_VALIDATION_20261001.md`.
