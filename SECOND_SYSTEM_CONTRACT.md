# Pre-registered second-system contract

This contract was committed before any run of the experiment defined below.
It freezes one benchmark and three lanes. No pilot, capacity sweep, horizon
ladder, parameter change, or replacement benchmark is permitted after this
commit.

## Source identity and plant

- Benchmark: the mature two-dimensional Flow* `brusselator` benchmark.
- Flow* source commit: `b85a3211748cb77b736fe4ad42ee02d8d2b81148`.
- Pinned benchmark path: `benchmarks/continuous/brusselator/brusselator.cpp`.
- Pinned benchmark file SHA-256:
  `b982f7c6f737e4b5e070942dc5fe01fa9d60e17a419a146d42444c71b5bf4f3b`.
- Generic accepted-boundary core commit:
  `b88888691eaeefac1fb2e48d5ab0f82ad50c58ac`.
- Pre-registration parent commit:
  `a33c402cf1a07ec92e741ed253a1c30184ea5ef5`.

The mathematical plant and its fixed expression trees are

```text
x' = 1 + x * (x * y - 4)
y' = x * (3 - x * y)
```

The Torch implementation must preserve those multiplication and subtraction
orders. It must not rewrite the right-hand side into an algebraically equal
graph after results are known.

## Frozen reachability request

```text
initial x = exact decimal interval [1.48, 1.52]
initial y = exact decimal interval [2.98, 3.02]
partition = B1
dtype/device = CPU float64
Taylor-model order = 6
fixed step = exact decimal 0.02, represented once as binary64
requested steps = 1000
requested horizon = 20
remainder estimate = [-1e-4, 1e-4] in each plant dimension
cutoff threshold = 1e-10
endpoint tightening/repair = disabled
```

Flow* must parse the initial decimal endpoints outward. The Torch affine
initialization must use `FlowstarNormalFlowpipeState.from_exact_decimal_box`.
The source-level Flow* time variable is not a third plant dimension in the
Torch lanes.

The Torch validator is the generic `flowstar_raw_remainder_compat` operator
with at most two Picard validation attempts, `validation_eps=1e-12`, standard
right-map range evaluation, and constant-centered normalization. The backend
is the existing CPU dense backend with the already published VDP production
range policy: adaptive subdivision, proactive depth one only in
`polynomial_truncation`, at most four leaves, and split variables `0,1`.
This inherited policy is frozen here and is not tuned on Brusselator results.

## Exactly three lanes

1. `flowstar`: the pinned Flow* polynomial reachability implementation with
   its benchmark-native symbolic-remainder queue capacity 1000.
2. `torch_generic_no_queue`: dependency-preserving normal insertion with no
   accepted-boundary symbolic-remainder queue.
3. `torch_generic_sr100`:
   `normalized_insertion_dependency_preserving_generic_sr` with capacity 100.

The lanes run sequentially without CPU contention. No other queue capacity,
reset mode, order, step, remainder, cutoff, horizon, range policy, or plant is
run in this phase. A terminal fixed-step rejection ends that lane; no adaptive
retry or clipped step is allowed.

## Endpoint and tube semantics

For accepted step `k` with local time `tau` and `h=0.02`:

- `endpoint(k)` is the outward range of the accepted raw segment Taylor model
  after substituting `tau=h`. It is not an endpoint-tightened or repaired box.
- `tube(k)` is the outward range of that same accepted segment over
  `tau in [0,h]`.
- `prefix_tube(k)` is the componentwise hull of the exact initial box and all
  `tube(1)..tube(k)`.
- Width is always `hi-lo`; aggregate width is `x_width+y_width`.

The Flow* extraction uses `Flowpipe::intEvalNormal` with the pinned
`step_end_exp_table` for endpoints and `step_exp_table` for tubes. Torch uses
`endpoint_raw_tm.range_box()` and `segment.tm.range_box()` respectively.

## Pre-registered checks

### Exact and local soundness

- The committed exact `Fraction` 2D generic-operator test must pass at the run
  source commit.
- Every published interval must be finite, ordered, and internally consistent
  with its recorded width within `1e-12`.
- Every accepted Torch step must have a successful finite Picard/subset
  validation record. A rejected step must not publish an accepted endpoint.
- The first ten accepted steps in each Torch lane must contain the same fixed
  deterministic sample set (four corners, four edge midpoints, and the center)
  at the endpoint and at local fractions `0,1/4,1/2,3/4,1`. Samples are only a
  local falsification check; the outward residual/subset check is the
  certificate-bearing condition.
- Flow* must report its native completion status and accepted fixed-step count.

Failure of any certificate-bearing soundness or finite-accounting check gives
`C3_GENERICITY_SOUNDNESS_GATE_FAILED_STOP` and stops the phase.

### Owner accounting

For accepted Torch SR100 boundary `k`, generation and accepted-boundary index
must both equal `k`. After the commit, the queue must have

```text
size = 0 and reset_count = k/100                  when k mod 100 = 0
size = k mod 100 and reset_count = floor(k/100)  otherwise.
```

The live owner generations and boundary indices must be the strictly
increasing boundaries after the most recent reset. All propagated, current,
roundoff, and cutoff owner widths must be finite and nonnegative. The no-queue
lane must never create a queue. A failed attempt must leave the last accepted
queue hash, generation, boundary, and reset count unchanged.

### Divergence, late prefix, runtime, and native horizon

- `first_live_divergence` is the first common accepted boundary where any
  endpoint or per-step tube binary64 bound differs between the two Torch lanes.
  It is reported, not selected.
- The late common prefix is the final 20% of the common validated horizon. Its
  median endpoint-width and tube-width ratios are
  `no_queue_width / sr100_width` at matched fixed-step boundaries.
- Solver wall time and, when available, Flow* reported core time are recorded;
  runtime is not a soundness or usefulness gate.
- Native horizon is `accepted_steps * 0.02` for each fixed-step lane and must
  equal the published completed horizon within `1e-12`.

## Status decision fixed before results

The generic core is soundness-validated only if the exact/local checks and
owner accounting pass. A material gain is established by either pre-registered
criterion:

1. Horizon gain: SR100 reaches at least `18.0`, never trails no-queue, and
   exceeds the no-queue native horizon by at least `2.0`.
2. Late-prefix tightness: both Torch lanes reach `20.0`; over `[16,20]`, the
   median no-queue/SR100 ratio is at least `1.10` for both endpoint and tube
   aggregate widths, and the 95th-percentile SR100/no-queue ratio is at most
   `1.05` for both metrics.

Flow* must complete the pinned T20 request for the final production-useful
status. The only possible conclusions are:

```text
C3_GENERICITY_SOUNDNESS_GATE_FAILED_STOP
C3_GENERIC_POLYNOMIAL_PLANT_CORE_VALIDATED
C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_NO_MATERIAL_GAIN
C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_PRODUCTION_USEFUL
```

`C3_GENERIC_POLYNOMIAL_PLANT_CORE_VALIDATED` is used when soundness is proved
but the common validated horizon is below 2.0 or Flow* does not complete, so a
material-gain decision is not eligible. Otherwise a sound run receives exactly
one of the two material-gain statuses. Only
`C3_GENERIC_CORE_VALIDATED__SECOND_SYSTEM_PRODUCTION_USEFUL` authorizes a later
batched/GPU phase.
