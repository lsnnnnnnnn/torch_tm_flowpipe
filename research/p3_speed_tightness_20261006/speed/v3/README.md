# NAV standard fused512 — v3

The module is copied from v2, with the (B=640, rows=512) configuration added and
the source-path attributes required by the original NAV standard wrapper.
No executed v1/v2 file is changed. Two full512 scratch buffers preserve the
original batch padding: 512 real rows, then128 real+384 repeated last rows.
Both rounds run in each graph, with the same all-success-or-original fallback.
The first original-reference call covers all640 rows using original512 chunks.

`run_nav640_fused_candidate.py` loads the frozen
`expansion/run_nav_standard_candidate.py`, replaces its explicit weighted
installer, and retains its full52,224,000-byte/384,000-record comparison.
The write hook explicitly replaces inherited256 and non-fused metadata:
START and WEIGHTED_BINDING say fused512/B640; no field mislabels it256.

Required arguments: `--base-wrapper`, `--adapters` (the frozen October5 helper
folder), `--gate`, `--output`; `--source-snapshot` and `--reference` pass through
unchanged. Use a new unique output, Python `-B`, GPU1/CPUs6–9, and the parent
supervisor. This preparation has not launched a numerical run.

Qualification requires600 refine calls, fast>0, fast+fallback=600, no candidate
exception, and one full640 reference check (1280 map-row evaluations). Legal
original fallbacks are preserved and are not numerical failures. Released
original pools must total1+fallback. If every step is fast, the receipt
separately verifies1200 fused replays,2400 replay map calls,768000 real and
460800 padded map rows. A mixed path can qualify by full saved equality;
measured speed decides promotion. The original source/config/property/result/
range comparisons and reverse restoration remain unchanged.

The local CPU control check passed21 cases including the new640/512 tail chunk,
second-chunk failures, output ownership and preserved metadata-plan admission.
This is a preparation check, not a CUDA or speed qualification.
