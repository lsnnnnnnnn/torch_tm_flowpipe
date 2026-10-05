# Opposite-op trig power sharing candidate

Current paper QUAD already uses the existing direct-variable same-op adapter.
The October5 full-run receipt reports46 trig calls,6 unique same-op plans and
40 potential repeated-op hits per weighted executor invocation. Its metadata
key begins with `sin` or `cos`, so it does not share powers between opposite
operations. This candidate addresses that remaining work rather than repeating
the existing optimization.

`trig_power_reuse.py` installs over the existing binding, retaining it. The
new outer executor creates a fresh memo for each call. Its key removes only the
operation from the existing conservative key; variable id, source and output
supports, gather-table identity, order, cache length, pair identities, full
power chain and accumulator bindings must still match. Non-direct expressions
remain ineligible.

The first operation executes the original `SparseValidCtx.mul_valid` for every
power and stores owned snapshots of its polynomial, remainder, and three
multiplication-cache entries. The opposite operation still executes the
original trig function, with a context proxy that supplies those owned powers
and cache entries. The proxy advances exactly the original `_i` cursor. Every
derivative-wheel coefficient, `acc_add`, remainder accumulation and strict
Lagrange tail is executed separately in its original order. `_j` advances only
through the untouched accumulator. Original cache offsets and final output
layout are retained. Returned reused powers are clones, so consumers cannot
modify the memo. The memo never crosses a map invocation, weighted round, step,
support change, or graph warmup/capture invocation.

Admission requires the exact existing same-op binding and original sparse
context/trig functions. The adapter checks call order, original cache owner,
complete chain length, table identity and context ownership. It does not
remove strict validation or replace the trigonometric mathematical body.

## New gates and runner

`trig_power_gate.py` is callable on CPU or the caller's already-prepared CUDA
runtime. It creates small new two-state direct and non-direct inputs at orders3
and4, tests point and interval coefficients, includes strided input storage,
then changes values in a second call. All returned coefficients, remainders,
range caches, strict tails and bad masks must match the existing same-op
implementation byte-for-byte. CUDA additionally captures/replays four fresh
graphs, including changing input values at the same signature. No old checker
or old saved input is executed. No extension is compiled or loaded by this gate.

`TRIG_POWER_CPU_CHECK.json` records8 passing new input cases using the real
frozen engine and CPU Torch2.2.2. There were44 computed and44 reused powers,
12 opposite-op chain reuses,8 deliberately ineligible non-direct calls, and
zero failed executions. Both adapters restored successfully. Only the install
version string was mocked for this CPU admission test; the production adapter
requires the existing Torch2.5.1 runtime. CPU equality is not a CUDA or full
trajectory qualification.

`run_trig_power_candidate.py` runs the original October5 QUAD wrapper with
private outputs, original weighted256, same-slope and equivalent comparison.
It does **not** use the slower QUAD fused-weighted candidate. It installs the
new trig adapter after those preparations and before the production engine's
first graph, runs the new small GPU gate, then executes the original driver.
The gate uses a separate small metadata engine; production initial boxes,
orders, strict injection, SR history and observers stay unchanged. The gate's
elapsed time is reported separately and remains included in outer/payload
wall time; the original driver timer starts after the gate.

Arguments: `--base-wrapper <frozen run_quad_candidate.py>`, `--gate <existing
private-output gate>`, `--output <new directory>`, optional `--source-runner`,
and `--mode batch2|full50` (default batch2). The parent schedules the run under
the original GPU3/CPU14–17 and resource/time limits. This preparation did not
launch any remote work. A full run requires a separate decision after the new
short result is read.

The original saved-output comparison executes before the optimization gate;
its result is also saved independently. Runtime qualification requires the
new eager/graph gate, completed executor calls without failures, nonzero
opposite-op reuse during the numerical run (excluding gate counters), complete
saved equality and restoration. Counters record Python eager/warmup/capture
construction only; they are never presented as CUDA replay or substep counts.
Timing determines whether to promote the candidate. No independent NN/CROWN
or end-to-end floating-point certificate follows from saved equivalence.
