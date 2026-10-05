# Saved physical-state widths for all 16 report instances

These tables derive only from existing accepted saved ranges, named pooled CSVs,
and original saved reductions. They contain 168,888 long rows and 364 summary
rows covering 16 instances × four methods. No solver, historical numerical
checker, GPU experiment, MATLAB program, or content-digest operation was run.

- [Summary JSON](summary.json) / [CSV](summary.csv): one row per physical
  state and method, including explicit missing states.
- [Long CSV](widths_long.csv) / [compact JSON](widths_long.json): saved
  endpoint and tube `lo`, `hi`, and absolute width at each recorded step.
  The compact JSON has `fields` and row arrays in that field order.
- [Source map](sources.json) and [state/contract map](instances.json).
- [Coverage and reasons by instance](coverage_by_instance.json) and
  [missing fields](missing_fields.json).
- [Alternative terminal objects](alternate_endpoint_objects.json): preserve
  the separate P3 QUAD driver final hull and native terminal objects.
- [New P3 equivalence receipts](equivalent_p3_runs.json): nine completed
  acceleration runs across five instances inherit saved widths within each
  receipt's actual comparison scope. They are not new tightness improvements.
- [Extraction metadata](AUDIT.json), [30 source spot checks and full-table
  arithmetic QA](EXTRACTION_QA.json), and [native QUAD source receipt](native_quad_allstates.json).

## Definitions

`absolute_width = hi - lo`. Bounds pool accepted lanes **at one saved step**.
`common_time` is the latest endpoint time present for all four methods and the
complete initial set, separately for each state. `common_endpoint_*` uses that
endpoint. `common_max_tube_width` is the maximum of the per-step pooled tube
widths from step 1 through this time, and remains null if any required step is
missing. It is never the width of a hull pooled across all times.

`own_last_*` uses each method's last saved accepted output; its lane count must
be consulted. For TORA remain, late Huan/Xiangru outputs cover survivors only.
`own_last_complete_*` describes the last endpoint covering the full initial
set. `own_max_tube_width` includes saved partial rows;
`own_complete_prefix_max_tube_width` uses complete-lane rows.
`full_*` is populated only when the requested full endpoint or complete tube
sequence exists. A `terminal_only` endpoint can support a common final endpoint
comparison while still providing no common-window tube maximum.

`common_safe_*` is separate metadata for two **previously recorded** safe
prefixes: DP more step 60 (0.30 s), versus numerical common step 64 (0.32 s);
TORA remain step 184 (18.4 s), versus numerical common step 189 (18.9 s).
Null safe-prefix fields elsewhere mean this particular derivative field was
not populated; they do not mean unsafe. These fields do not rerun a property
checker or establish an independent NNCS certificate.

## Coverage boundaries

ACC, Attitude, Docking, DP less, NAV standard/robust, both TORA reach instances,
and Unicycle have saved full-horizon ranges for every physical state and
method. Native and P3 QUAD have all 12 states at 1,000 steps; Huan/Xiangru QUAD
have only saved driver terminal hulls. Their missing tube series stay null.
The primary P3 QUAD endpoint is the observer object. Its driver final hull is
retained separately because these saved objects are not numerically identical.

Balancing is the supplemental repository raw-four-input contract, not a
resolution of the paper five-input controller conflict. Its common numerical
prefix is 0.415 s, short of 10 s. DP more is partial to 0.32 s commonly, short
of 0.4 s. TORA remain Huan/Xiangru lose complete-lane numerical coverage after
18.9 s; surviving-lane values do not establish a full 20 s result.

Single Pendulum has the named two-state profile only. The official third-state
mapping is unresolved and all third-state cells remain missing. Both Airplane
main-contract instances have no accepted full-initial-set flowpipe data.
Rejected proposals, old singleton runs, and native split first-step diagnostic
cells are not substituted into the main-contract width table.

NAV state order follows the executable contract `x,y,speed,heading`; the
conflicting paper order is not used. Airplane uses contract names
`sx,sy,sz,vx,vy,vz,phi,theta,psi,r,p,q`, corresponding positionally to raw aliases
`x,y,z,u,v,w,phi,theta,psi,r,p,q`. Clocks, held controls and Unicycle's constant
disturbance are excluded from physical-state widths.

## Rebuild

From the repository root, run Python only:

```text
python3 -B tools/build_archcomp26_widths_20261005.py
python3 -B docs/evidence/results/archcomp26_report_20261005/widths/check_extraction.py
```

The native QUAD all-state CSV is derived by the adjacent
`derive_native_quad_allstates.py` from the already saved `SCAN.json`; the
original scanner is not executed. NAV correspondence checks reach the saved
independent reduction JSON because the large raw binaries remain remote.
QA here establishes table correspondence and formatting, not mathematical
enclosure or an independent end-to-end floating-point NNCS proof.
