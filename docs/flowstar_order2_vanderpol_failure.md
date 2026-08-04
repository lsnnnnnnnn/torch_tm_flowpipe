status: diagnostic
valid_for_commit: 0cdc47038cfed7b42a392785fa144123d3737724
superseded_by: none
allowed_use: diagnostic only

# Flowstar order-2 Van der Pol validation rejection

The supported single-step probe was run with effective degree 2, horizon 0.1,
maximum one accepted segment, adaptive range 0.002–0.1, cutoff
`[-1e-10,1e-10]`, and target remainder `[-1e-4,1e-4]` per state dimension.
The backend was factually classified `stock-plus-gcc15-compat` at Flowstar SHA
`b85a3211748cb77b736fe4ad42ee02d8d2b81148`. Its sole tracked change is the
recorded GCC15 derivative compatibility fix; this is not labeled
`unmodified-stock` or `official-stock`.

The probe process compiled and returned normally. The solver attempted step
sizes 0.1, 0.05, 0.025, 0.0125, 0.00625, and 0.003125, accepted no segment,
and would next halve below the 0.002 floor. The outcome is therefore:

```text
failure_category = validation_rejected
failure_reason = remainder_self_map_failed
last_accepted_time = 0.0
completed_requested_scope = false
completed_full_horizon = false
```

At the final candidate, the x remainder was approximately
`[-1.10751e-5, 1.08411e-5]`; the y remainder was approximately
`[-2.779354e-4, 1.062582e-4]`. The failing dimension was y, with a lower-side
self-map defect of approximately `1.779354e-4`. The y multiplication
remainder contribution was approximately
`[-2.634298e-4, 9.120954e-5]`; the y cutoff polynomial difference was around
`[-2.10e-18, 6.94e-19]`. Symbolic propagated width was zero on this first-step
attempt.

This is not a crash, not `unsupported_order`, and not a completed run. It is a
single-step diagnostic, not evidence of a full-horizon result or a cross-tool
comparison. Exact command, backend state, compile/run return codes,
stdout/stderr, trace, and numerical manifest are under the cleanup audit's
`order2_smoke/` directory.
