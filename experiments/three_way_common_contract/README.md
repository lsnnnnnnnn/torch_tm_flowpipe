# Three-way low-order reachability comparison under common external contracts

> **Superseded:** the committed `results/20260724T132534Z` directory is a
> preserved preliminary artifact and must not be used for winner claims. See
> `HISTORICAL_RESULTS_NOTICE.md` and the audited replacement in
> `../three_way_comparison_repair/`.

This experiment compares the real `torch_tm_flowpipe`, DiffReach, and Flow*
implementations without claiming a common internal order. The canonical ODE,
state order, initial box, step sizes, horizons, and point checks live in
`benchmark_spec.yaml` and are consumed by all three adapters.

The primary protocols are:

- `one_step_common_input`: exactly one segment from an identical input box.
- `multi_step_common_box_carry`: every accepted endpoint is reduced to a
  componentwise box and used as the next segment's input.
- `native_low_order`: each tool carries its natural low-order representation;
  this is supplementary.

Torch uses complete total-degree order 1 on float64 CPU. DiffReach primary rows
set `TRUNCATE_TO_AFFINE=True` and call the upstream
`src.reachability.CT_Dyn_Reach.step_once` method. Flow* uses its minimum legal
fixed order, order 2, through the existing static library. Flow* output uses
the previously audited workaround that restores the configured Picard
candidate remainder whose self-map was validated before stock refinement.

## Reproduce

From the required worktree:

```bash
cd /srv/local/shengenli/torch_tm_flowpipe_three_way_comparison
experiments/three_way_common_contract/run_smoke.sh
experiments/three_way_common_contract/launch_background.sh
tmux attach -t tm_three_way_common_contract
```

To run the full sweep in the foreground:

```bash
experiments/three_way_common_contract/run_all.sh
```

Set `THREE_WAY_OUTPUT_DIR=/absolute/path` to select an output directory. A full
run produces the required raw, one-step, common-time, failure-horizon, runtime,
correctness, environment, plot, and report artifacts.

DiffReach's analytic plant module imports optional neural-bound dependencies at
module import time. The dedicated environment does not contain `jax_verify`,
so the adapter installs a fail-fast import shim for only those unused symbols.
The reachability class, Picard operator, Taylor-model arithmetic, and symbolic
remainder operations are imported and executed from the unchanged upstream
DiffReach checkout. `diffreach_upstream_provenance.json` records source paths,
line numbers, hashes, and trace invocation counts.

Deterministic high-accuracy trajectories are a sanity check only. They do not
prove soundness; native validation and analytic Riccati/harmonic containment
are enforced independently.
