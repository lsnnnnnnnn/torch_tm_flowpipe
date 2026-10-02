# Original Flow* symbolic-remainder QUAD short gate — failed

This **new isolated** diagnostic copied the frozen 2026 paper-equation native
QUAD source and original Flow* archive. The [source diff](quad_gate.patch)
limits the first of 1,024 initial boxes to one CROWN control call and one
`h=0.005`, order-2 symbolic-remainder ODE step. The original full-50 job,
source and library were not changed or restarted. Observer off/on each ran in
a new native process with the same fresh CROWN server, GPU 2, CPU 10, and
120 s per-process limit.

Both modes: status `2 = COMPLETED_SAFE`, one accepted step, identical RPC
input/output JSON, identical 12-state terminal axes and identical saved
`ranges.bin`. The [observer](on/octagon.csv) saved x1, x2, x1+x2 and x1−x2
for the accepted tube and propagated endpoint. The printed `UNKNOWN` concerns
the **full T=5 target at T=0.005**, not failure to accept the one small step.

The [independent finite-sample audit](AUDIT.json) exited nonzero: 513 sampled
initial/control choices, 20,520 direction and axis checks, 2,048 terminal
`x7`/`x8` checks outside the frozen Flow* endpoint bounds, maximum overshoot
`1.9825917926144243e-6`. All sampled x1/x2 octagon checks passed. The first
point is `x1…x3=-0.39999999999999997`, `x4…x6=-0.4`, other states zero, with
constant `u=(12.730082304606912,-0.001486676491913386,
-0.006470973906107247)`. This u is inside the **affine-plus-residual control
relaxation injected by the C++ source**: `T·x + center ± radius`, after the
source's float32 JSON conversions. RPC `u_min`/`u_max` are intercept bounds,
not full control bounds without `T·x`. The sample is not claimed to be the
actual neural network output.

SciPy DOP853 integrated the paper ODE with `rtol=1e-13`, `atol=1e-15`,
`max_step=0.00125`; Radau used `rtol=1e-12`, `atol=1e-14`, same max step.
Their first-witness full-state maximum difference was `2.054e-15`. At
`t=0.005`, DOP853 gave `x7=-3.441380768320594e-7` against saved
`[1.6384537157764776e-6,1.6545006846119772e-6]`, and
`x8=-1.4979106264136853e-6` against
`[3.9749375661109013e-7,4.0249593908225256e-7]`.

Files: `quad_gate.cpp` and `quad_gate.patch` identify the short-run source;
`off/` and `on/` hold unmodified raw status, axes, RPC, range and stdout/stderr
receipts; `controller_rpc.jsonl`, `server.log` and `listeners_at_start.txt`
show the server calls and ownership; [AUDIT.json](AUDIT.json) stores the
concrete witness and exact comparison outcomes. The generator, observer,
runner and checker are in `tools/` with names beginning `build_native_quad_sr`,
`quad_gate_observer`, `run_native_quad_sr`, and `check_native_quad_sr`.

This finite one-box/one-step result blocks promotion of the original library
as a validated relaxed-input enclosure for a production octagon observer.
It does not prove a true NN closed-loop counterexample, invalidate every
range in the old long job, or establish a full-time property conclusion.
