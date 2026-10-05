# Fused batch v2 — existing QUAD metadata-plan admission

The first QUAD fused40 attempt stopped during installation, before numerical
execution. The v1 admission check required all four weighted functions to come
from `weighted_validation.py`. Canonical paper QUAD prepare already installs
the qualified hybrid metadata backend, whose `install` assigns
`wv._plan = weighted_plan`. This is an expected metadata factory; `_map`,
`_evaluate_map`, and `refine_accepted` remain the original numerical functions.

The actual source chain is:

1. `tools/archcomp26_quad_paper_p3_nohash.py:175` calls `hybrid.install`.
2. `tools/archcomp26_quad_p3_nohash/hybrid_metadata.py:27` loads the backend
   as `fullbatch_p3_metadata_backend` and calls `backend.install`.
3. `tools/archcomp26_quad_p3_nohash/metadata_backend.py:236` installs its
   `weighted_plan`, retaining the original in `_original_weighted`.

Only admission changes in this revision. For B1024 it also accepts the exact
function object of that registered, installed module, provided its `wv`/`se`
owners match and its `_original_weighted` is the original weighted module's
function. The existing metadata function is kept and called as installed.
Other numerical function checks remain strict. The receipt records both
metadata and original source paths. No trigonometric, strict arithmetic, or
metadata wrapper is removed or replaced.

The runner is a direct byte-for-byte copy of the v1 runner and imports the
candidate beside it. Use both Python files from this v2 directory for the
separate new QUAD attempt. The already executed v1 files and results remain
unchanged. First-live-input tensor/statistics equality, complete saved-output
comparison, counters, memory reclamation, and restore checks are unchanged.

`CPU_CONTROL_CHECK.json` includes the 14 fused control cases plus a new metadata
ownership admission check. The new check verifies retention of the installed
metadata function and rejection of an uninstalled backend, wrong owners, and
a foreign original function. It does not run a metadata numerical factory,
old checker, GPU job, or content digest operation.
