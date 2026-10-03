# All-box native control-construction gate — first conditional refusal

The one-shot isolated gate completed with build/native/check exit codes
**0/0/2**. The native process wrote [1,024 full-precision box traces](ALLBOX_TM_TRACE.jsonl)
and checked all 12,288 constructed input intervals against the saved first
RPC batch. The exact-rational [checker result](RESULT.json) stopped at
**lane 1, output 1**: its conditional same-symbol lower margin is
`−1.18847023973899759798774615363135126466787028×10⁻¹⁵`; the upper margin
is `+1.91961853843890312509623155112336440053510955×10⁻¹⁵`.
The checker completed all three outputs for lane 0 and then checked only
lane 1's first output: **1 full box plus 1 additional output row**, not
1,024 conditionally qualified boxes. It retained first-refusal semantics;
the remaining 1,023 boxes are not claimed to pass this exact gate.

[`allbox_control_trace.cpp`](allbox_control_trace.cpp) reconstructed the
frozen `8×8×8×2×1×1` initial partition in original loop order. It read the
saved first-call RPC and separate conditional corrected-bias ledger, followed
the frozen `.asFloat()` slope/bias → double center/radius → native TM
construction order, and added the exactly representable `[-2^-50, 0]` only
to each **final control-TM remainder**. It did not change input TMs, slopes,
biases, centers/radii, or output polynomials. Each line of the native trace
contains hex coefficients and endpoints, original and expanded remainders,
and the source/normalized domains. There was no CROWN, ONNX, reachability or
ODE call, and no old experiment was restarted.

[`check_allbox.py`](check_allbox.py) reused the archived exact-rational
same-symbol test. It first matched lane 0's new native trace exactly to the
prior [one-box padded trace](../native_quad_lane0_correlated_remainder_gate_20261003_001/README.md),
then checked source/input/RPC inclusion, saved same-slope replay, conditional
float32 affine transfer, native center/radius construction, and actual final
remainder expansion before each output inclusion test. Lane 0's three lower
and upper margins passed, as in that prior one-box receipt. The next lower
margin was negative, so the checker stopped and recorded
`UNDECIDED_FIRST_REFUSAL`. The upper side of that row passed.

The original [START](START.json), [RESULT](RESULT.json), native trace, and
build/run/check logs and exit-code files are retained. The unique server run
directory is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_quad_allbox_correlated_remainder_gate_20261003_001/`.
The [runner](run_once.sh) refused to overwrite a previous start/trace/result
and capped each stage. The fixed pad was selected after observing lane 0's
deficits; the lane-1 refusal shows that it **does not clear this all-box
same-symbol sufficient gate**. The negative sufficient margin is
not a demonstrated true-network output escape or a CROWN counterexample.

Even a passing conditional construction row assumes the original real-affine
CROWN inequalities over that saved RPC box. This run independently proves
neither those inequalities nor the selected network's behavior, Flow* plant
containment under widened controls, later control periods, or the `T=5`
reach-and-remain property. Native octagon production remains **CLOSED**. No
hash or digest verification was performed.
