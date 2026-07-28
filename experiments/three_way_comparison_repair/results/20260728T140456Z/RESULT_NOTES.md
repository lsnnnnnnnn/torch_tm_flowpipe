# Result notes

This is the complete 2026-07-28 correctness-repair run.

The first collector invocation incorrectly selected Outcome C because its
partial-comparison decision included known Flow* stock trajectory violations in
the Torch/DiffReach gate. The solver outputs were not rerun or altered. Commit
`bd88988` split the aggregate trajectory gate from the Torch/DiffReach partial
gate, and collection, plotting, and report generation were resumed in the same
required tmux session. The final `correctness_checks.json` selects Outcome B
while retaining all 316 Flow* exact-reference violations and 308 Flow* sampled
trajectory violations.

`flowstar_refinement_trace.csv` is present in the local result directory and
contains 4,109,335 trace records (775,553,043 bytes; SHA-256
`2e8f4df1da0d0a23528d93b8ad76d106f42d786de10d8d42bbf139ed461be630`).
To avoid committing a 740 MiB generated CSV, the git result commit contains its
lossless `flowstar_refinement_trace.csv.gz` form (SHA-256
`dceb4e9b06841dbc032740c73808e444dfe67ec6cd4fe6a591fcb075ff3e8f85`).
Use `gzip -dc flowstar_refinement_trace.csv.gz` to reproduce the exact CSV
bytes.
