# Lane-0 native control-remainder gate — one completed isolated diagnostic

The one-shot isolated diagnostic completed with build/native/check exit codes
**0/0/0**. The original [`START.json`](START.json), full-precision
[`LANE0_TM_TRACE.json`](LANE0_TM_TRACE.json), [`RESULT.json`](RESULT.json), and
build/run/check logs are preserved. It checked one saved source box and three
control outputs; no benchmark cell or production qualification is claimed.

The [source](lane0_control_trace.cpp) reuses the archived lane-0 native control
construction and its saved first-batch RPC/corrected bias inputs. On each of
three outputs it records the original final Taylor-model remainder, adds the
exact binary64 interval `[-2^-50, 0]` **to that remainder only**, then records
the new remainder and full-precision Taylor model. It does not change the
input TMs, slopes, corrected biases, center, radius, or output polynomial; it
does not call CROWN, ONNX, or the plant ODE.

The [checker](check_replay.py) refuses at the first baseline mismatch, confirms
all original remainders and polynomials against the frozen prior lane-0 trace,
checks that each native lower endpoint moved outward by at least `2^-50` while
each upper endpoint stayed fixed, then reuses the archived exact-rational
same-symbol gate. The first conditional inclusion refusal ends the check. The
fixed pad is a diagnostic chosen to exceed the three recorded lower deficits;
it is **not** a derived all-box rounding budget or production repair. All three
outputs passed this same-symbol sufficient condition. The saved exact lower
margins are `+5.551331889×10⁻¹⁶`, `+8.876625130×10⁻¹⁶`, and
`+8.881253105×10⁻¹⁶`; all upper margins are positive. Each actual native
lower remainder endpoint moved outward by exactly `2⁻⁵⁰`, and each upper
endpoint stayed fixed. The result remains conditional on the original
real-affine CROWN inequalities on this one saved RPC box and does not certify
the network, plant, later controls, or reach-and-remain property. The native
octagon production gate stays **CLOSED**.

The unique server run is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_quad_lane0_correlated_remainder_gate_20261003_001/`.
The [runner](run_first_box.sh) refused to overwrite a prior start/trace/result,
limited build and native/check stages to 300/120/120 seconds, and recorded
each exit code. The original binary and large build products remain in that
server directory; source, trace, receipts and logs are mirrored here. No hash
or digest verification was performed.
