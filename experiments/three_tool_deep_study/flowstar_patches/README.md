# Flow* patch export

`fa39f7a_series/` is the authoritative five-commit series from upstream
`b85a321` through local audit commit `2310c1a`.  Apply it with `git am` in
order.  Commit 5 extends the already-audited full-Picard fallback to the
additional interval/adaptive-symbolic overloads used by the Van der Pol
comparison.  It is diagnostic: the adaptive endpoint audit demonstrates that
the fallback does not change the observed collapsed-endpoint failures.

The Flow* upstream remote rejects writes to the audit branch.  Consequently,
the formatted five-patch series in this directory, rather than an upstream
branch name, is the portable source of the final audit checkout.

The older top-level patch files were generated before the final explicit
leaf-cache trace checkpoint.  They are retained as recovery evidence and are
superseded by `fa39f7a_series/`; do not combine the two sets.
