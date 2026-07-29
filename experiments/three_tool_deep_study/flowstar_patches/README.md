# Flow* patch export

`fa39f7a_series/` is the authoritative four-commit series from upstream
`b85a321` through audit commit `fa39f7a`.  Apply it with `git am` in order.

The older top-level patch files were generated before the final explicit
leaf-cache trace checkpoint.  They are retained as recovery evidence and are
superseded by `fa39f7a_series/`; do not combine the two sets.
