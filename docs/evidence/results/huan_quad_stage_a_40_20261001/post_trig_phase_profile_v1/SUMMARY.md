# Old-author QUAD P3 post-trig bounded phase profile (no digests)

Date: 2026-10-02. This is a **new isolated 40-step diagnostic**, not a restart
or continuation of the archived 1,000-step experiment. It uses the saved
old-author QUAD `quad_40.yaml`, all 1,024 boxes, h=0.005, two controller periods,
working P3 / ODE point P2 / validation P4, strict boundary and host K20 full SR,
and direct sin/cos reuse. The source and actual command are in
[`phase_probe.py`](phase_probe.py) and the supervisor [`START.json`](completed_run_001/START.json).
The probe loads the existing [no-digest, no-JIT observer runner](../observer_pair_v1/runner.py),
which preloads eight existing CUDA libraries. It checks that the runner contains
its digest/JIT/build guards and no old snapshot restore call before execution.
No content digest was calculated or checked in these attempts.

| New attempt | Result | Original evidence |
| --- | --- | --- |
| `v1/run_001` | 0/40; Python's `cProfile` imported the local script named `profile.py` instead of the standard-library module. Numerical solver did not start. | [failure receipt](failed_run_001/PHASE_PROFILE.json), [supervisor result](failed_run_001/RESULT.json), [exact failed source](failed_run_001_source/profile.py) |
| `v2/run_001` | Completed 40/40, all 40,960 lane-steps accepted, 2 NN calls, no broken boxes. | [phase record](completed_run_001/PHASE_PROFILE.json), [solver result](completed_run_001/data/RESULT.json), [supervisor result](completed_run_001/RESULT.json) |

The successful attempt's supervisor wall was 60.134056 s and driver's internal
time was 53.221452 s. Its phase probe inserted one CUDA synchronization after
each numerical advance so event spans could be read; this changes timing
boundaries. The 40 advance calls occupied 52.675512 s on the probe's wall
clock and 52.674081 s in aggregate CUDA stream spans. Within that stream span:

| Outer phase | Calls | Stream span (s) | Share of measured advance span |
| --- | ---: | ---: | ---: |
| Weighted accepted refinement | 40 | 30.503348 | 57.91% |
| Ordinary validation | 40 | 17.249883 | 32.75% |
| Compose | 40 | 1.639102 | 3.11% |
| Structural Picard | 40 | 0.612256 | 1.16% |
| Host SR propagation | 40 | 0.610925 | 1.16% |
| Ordinary refinement | 40 | 0.115454 | 0.22% |
| Self-map retry wrapper | 40 | 0.004094 | 0.01% |

Weighted accepted refinement plus ordinary validation account for **47.753231 s
of the 52.674081 s measured advance span (90.66%)**. For steps 21–40 alone,
their combined share is 22.746878/24.264662 s (93.74%). The separate working
graph eviction hook ran after each prune and measured 0.090426 s over 40 steps;
9 steps had one evicted entry each. These are time spans around outer calls,
not sums of GPU kernel times or a complete decomposition of every advance
operation. Warmup/capture, host scheduling, the added synchronizations and
uncovered operations affect the numbers. They do not establish any 1,000-step
runtime, stable speed ranking, or safe removal of validation work. Both this
run and its direct comparison arm already omit the frozen full-run diagnostic
state signature, so this is not a signed-versus-clean ablation of that path.

The probe compared every saved 40-step observer JSON row and selected driver
metrics to the earlier `observer_on_002` run. It also compared all values in
the final per-box `tube_12x2` and `endpoint_12x2` arrays and the final status
array; each is directly equal. The copied local `.npy` files and 40-row JSONL
are also byte-for-byte equal to the archived local reference by direct file
comparison, without digests. See the [comparison receipt](completed_run_001/PHASE_PROFILE.json).
Neither experiment compares all intermediate plant/SR tensors, and this
non-JIT path is not byte-qualified against the frozen full-run JIT path.

The shortened unchanged target check prints `FALSIFIED` at t=0.2; it is **not**
a T=5 property result. No production optimization was enabled by this profile.
The next justified work is a narrowly scoped study of the two validation
calls on saved matching inputs, with a direct numerical gate before adopting
any change. Existing 1,000-step results remain immutable.
