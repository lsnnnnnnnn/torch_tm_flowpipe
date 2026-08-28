# Brusselator terminal replay and SR1000 parity contract

This follow-up contract is frozen before either numerical execution described
below.  It follows the completed, pre-registered three-lane run in
`SECOND_SYSTEM_CONTRACT.md`; it does not reopen that phase or authorize a new
benchmark, capacity sweep, horizon ladder, or performance study.

The machine-readable authority for frozen identities, numeric fields, and
decision thresholds is
`benchmarks/brusselator_terminal_sr1000_contract.json`.  This document defines
their scientific interpretation.

## Scope and ordering

Exactly two baseline numerical executions are authorized, in this order:

1. one reconstruction of the published Torch SR100 accepted prefix followed
   by exactly one replay of its terminal fixed-step rejection at boundary 356;
2. one Torch SR1000 run from the original Brusselator initial set until the
   first fixed-step rejection or the requested boundary 1000.

The committed stock Flow* T20 trace is reused read-only for parity.  No new
Flow* numerical lane is needed.  Huan, CROWN, NN controllers, GPU throughput,
and every additional benchmark are outside this contract.

If SR1000 proves the non-capacity verdict, one paired baseline/candidate local
operator evaluation on the mechanically selected byte-identical Torch
checkpoint is allowed by the C4 gate below.  It is not a native horizon lane.

## Frozen identities

- Plant: `x' = 1 + x * (x * y - 4)`, `y' = x * (3 - x * y)`, preserving these
  expression trees.
- Stock Flow* source commit:
  `b85a3211748cb77b736fe4ad42ee02d8d2b81148`.
- Stock Flow* benchmark SHA-256:
  `b982f7c6f737e4b5e070942dc5fe01fa9d60e17a419a146d42444c71b5bf4f3b`.
- Reused stock segment trace SHA-256:
  `08e184e2b0a99be48417be8971ed6632eccec0630849787a1048b9962d15f567`.
- Generic accepted-boundary core commit:
  `b88888691eaeefac1fb2e48d5ab0f82ad50c58ac`.
- Published SR100 run commit:
  `33ea600d01143177d02784b204cafabb4343711d`.
- Published SR100 accepted prefix: 355 boundaries, horizon 7.10; terminal
  rejection: attempted boundary 356.

All Torch executions require CPU float64, a clean tracked worktree, and no
change to the generic core paths relative to the generic-core commit except a
later, separately authorized C4 fix under the gate below.

## Execution 1: one terminal replay

The replay reconstructs boundaries 1 through 355 with the exact SR100 settings
and compares every accepted endpoint/tube binary64 bound plus every accepted
queue hash to the committed trace.  Any mismatch stops before boundary 356 and
does not consume the one terminal replay.

After exact reconstruction, canonical v5 checkpoints of the complete current
TM, normal state, and accepted-boundary queue are written immediately before
and immediately after one call attempting boundary 356.  The call is required
to reject without publishing an endpoint or poststate.  Rollback closes only
if both checkpoint files are byte-identical, their full checkpoint hashes are
equal, and queue hash, generation, accepted boundary, size, reset count, and
owner indices are unchanged.  There is no retry.

This supplemental evidence may satisfy the missing SR100 rollback check in the
original evidence package.  It must not rewrite the original segment CSV or
summary declaration.

## Execution 2: completely matched SR1000 lane

The Torch lane matches the stock Flow* request field for field:

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
symbolic-remainder queue capacity = 1000
endpoint tightening/repair = disabled
```

The Torch validator remains `flowstar_raw_remainder_compat`, with at most two
Picard attempts, `validation_eps=1e-12`, standard right-map range evaluation,
constant-centered normalization, and the already frozen dense range policy:
adaptive subdivision, proactive depth one only for
`polynomial_truncation`, at most four leaves, split variables `0,1`.

The reset mode is
`normalized_insertion_dependency_preserving_generic_sr`.  Before boundary
1000, generation and accepted-boundary index equal the accepted step, queue
size equals that step, reset count is zero, and owners are `1..step`.  If
boundary 1000 is accepted, its post-commit queue is empty with reset count one.
A failed attempt leaves all accepted-state fields unchanged.

Endpoint, tube, prefix-tube, validation, completion, and width semantics are
exactly those in `SECOND_SYSTEM_CONTRACT.md`.  The long prefix is every common
accepted fixed-step boundary from 1 through the last Torch accepted boundary;
the stock side must have the same step number and `h=0.02`.

## Capacity/reset decision

- `QUEUE_RESET_CAPACITY_SUFFICIENT` requires the sound SR1000 lane to accept all
  1000 boundaries and complete T20.  This supports the capacity/reset bundle as
  sufficient to explain the prior Torch horizon stop; it does not distinguish
  capacity from the reset transition without an additional, separately
  authorized intervention.
- If the sound SR1000 lane rejects before boundary 1000, the remaining
  stock-vs-Torch gap is `NOT_SOLELY_QUEUE_RESET_CAPACITY`.  The first terminal
  rejection and the full common prefix are retained; no capacity variant may
  be run.

## Material divergence and C4 authorization

If and only if the decision is `NOT_SOLELY_QUEUE_RESET_CAPACITY`, localize the
first stock/Torch operator-stage divergence that is both persistent and
material.  `material` retains the previously published absolute threshold:
an endpoint/tube bound or an outward stage interval differs by more than
`1e-12`; an isolated smaller binary64 difference is recorded but is not the
C4 target.  The search order is fixed:

1. accepted input/right-map and queue state;
2. polynomial Picard iterate before cutoff/truncation;
3. truncation and cutoff owner intervals;
4. raw RHS remainder image and Picard residual;
5. validated remainder/subset decision;
6. endpoint substitution and tube range;
7. accepted-boundary normalization, insertion, and symbolic history update.

Cross-tool states must not be reboxed, projected, or stripped of time/queue
state and called same-input.  A Torch C4 numeric change is authorized only when
one hypothesized operator replacement is run against a byte-identical Torch
checkpoint and all non-intervened inputs are hash-identical.  The replacement
must preserve outward containment, change the predicted material field in the
stock direction, and pass its exact/local oracle.  Baseline and intervention
must be evaluated on the same input; sequential native reruns are insufficient.

At most one Torch C4 numeric fix may be implemented.  Instrumentation,
fail-closed verification, evidence packaging, and tests do not count as a
numeric fix.  If no candidate passes the same-input gate, the required result
is `NO_C4_FIX_AUTHORIZED`.
