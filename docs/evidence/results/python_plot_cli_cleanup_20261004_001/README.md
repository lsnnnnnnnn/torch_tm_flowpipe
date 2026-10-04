# Python-only plotting CLI acceptance — 2026-10-04

The three maintained CLI paths no longer automatically write MATLAB files:
`flowpipe_plot`, `flowpipe_plot_nohash`, and `tm_octagon_nohash`. The ordinary
plot CLI now records artifact paths and byte sizes without content digests.
Legacy source-export identity checks remain in place and were not executed.
The directional geometry/stream redraw imports no numerical solver.

This directory records saved-data rendering only. No old experiment was rerun,
no numerical source or historical evidence was modified, and these checks do
not establish a mathematical or independent NNCS certificate.

- [Machine-readable acceptance](artifacts/ACCEPTANCE.json): five CLI invocations
  cover TORA state/state endpoints, DP time/state tubes, saved directional
  geometry endpoints and JSONL tubes, and native DP range re-export.
- The DP re-export preserves all 100 frames with 225 lanes and its plot spec;
  the directional JSONL replay preserves all three saved frames and initial box.
- Seven historical JSON, JSONL, CSV and binary inputs compare byte-for-byte
  equal before and after rendering. This is direct byte comparison, no digest.
- [QUAD time/height PNG](artifacts/quad_saved_fourway_t_x3.png) and
  [PDF](artifacts/quad_saved_fourway_t_x3.pdf) use the existing saved-data helper
  with a new output prefix. Its historical `main()` was not called. The saved
  JSON was copied unchanged and the regenerated 2,002-row CSV is byte-identical.
  Native/P3 have 1,000 saved tubes; Huan/Xiangru remain endpoint-only at T=5.
- All six PNG/PDF pairs were produced by Python Matplotlib. Four representative
  PNGs were visually inspected: QUAD time/height, TORA state/state, DP time/state,
  and the CPU plant-only directional endpoint. Initial/Safe/Target layers and
  source-boundary captions remain visible; no `.m` file exists in the output.
- Four focused unit checks passed: DP native input/redraw, both other CLI
  redraws, archived QUAD B2 saved-range rendering, and overlay/refusal. The last
  two results are retained in [the unit log](additional_unit_checks.log).

All `hashlib` constructors, `new`, and `file_digest` are replaced with failing
guards during the acceptance run. The MATLAB helper functions also fail if
called. Neither was called. The check explicitly confirms `torch` was not
imported. No whole test suite or digest-based legacy verifier was run.

From the repository root, replay into a **new** output directory:

```console
MPLCONFIGDIR=/private/tmp/flowpipe-python-only-20261004-mpl \
  /opt/anaconda3/bin/python \
  docs/evidence/results/python_plot_cli_cleanup_20261004_001/verify_saved_redraws.py \
  /path/to/new/acceptance-output
```

The replay script requires the existing saved inputs, including the original
local DP `ranges.bin` path recorded in its geometry. The three general CLI
formats use JSON; this acceptance preserves CSV through the established QUAD
saved-data exporter. Native octagon production remains outside this check.
`tools/verify_flowpipe_plot_artifacts.py` expects legacy digest-bearing receipts
and does not apply to these new path/byte-size render receipts.
