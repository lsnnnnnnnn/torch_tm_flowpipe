# Stock Flowstar observation patch

This observation-only source targets Flowstar commit
`b85a3211748cb77b736fe4ad42ee02d8d2b81148`. Copy `AuditTrace.h` into
`flowstar-toolbox/`, then apply `observation_hooks.patch` at the Flowstar root.

Build on GCC 15 with `make CXX='g++ -fpermissive'`; the flag is required by an
unrelated stock const-correctness error in `TaylorModel.h` and is not a source
fix. Set `FLOWSTAR_AUDIT_TRACE`, `FLOWSTAR_AUDIT_RUN_ID`, and
`FLOWSTAR_AUDIT_SOURCE_COMMIT` when running the benchmark.

The hook is after the stock adaptive advance returns. It reads stored objects
and emits JSONL only. Fields not available at that observation point are null
with a reason; it does not relabel a right-map or aggregate box as another
lifecycle object. The stock/instrumented schedule and both plot files are
byte-compared by `process_flowstar_observation_trace.py`.
