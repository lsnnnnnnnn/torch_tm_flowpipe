# Airplane continuous native — isolated one-step trace alignment control

This is a new diagnostic run, separate from the three original full-box
smokes. It used the saved order-3 Airplane source, the single complete 2026
initial box, one CROWN call, and only the first `h=0.01` Flow* step. The
copied library was instrumented in the `Expression<Interval>` symbolic-
remainder overload; the actual `ODE<Real>` caller uses the `Expression<Real>`
overload, so this run produced no Picard trace rows. The mismatch was noticed
from the raw log and retained rather than relabeling it as a successful trace.

The original [RESULT](run/RESULT.json) is `failed/exit 2`, not timed out,
with 3.674898 s outer wall. The raw [native log](run/native.log) says
`STATUS 4 = UNCOMPLETED_SAFE`, `0` accepted segments, and `UNKNOWN` at this
short horizon. The [one RPC](run/controller_rpc.jsonl), [full-box ledger](run/initial_boxes.json),
empty [range file](run/ranges.bin), [build receipt](BUILD.json), exact
[copied-source patch](Continuous.patch), [one-step entry](airplane_first_reject_trace.cpp),
and [driver](driver.py) are preserved. `BOX_SAFE 1` is vacuous for zero saved
segments. This has no `T=2` or property result.

The next isolated [run `_005`](../native_airplane_first_reject_trace_20261002_005/README.md)
places the trace in the actual Real symbolic-remainder overload. `_001`,
`_002`, and `_003` were preparation-only directories stopped by source-marker
or file-writing errors; none compiled a binary or launched the solver.
