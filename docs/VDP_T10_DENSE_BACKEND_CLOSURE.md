# Van der Pol T=10 dense-backend closure

Highest achieved state: **S3 — `dense_multistep_integrated`**.

The authoritative order-4 request did not complete T=10. The unmodified
`hybrid_dense_core` validates 308 steps through
`t=6.3172908799330765`; the next adaptive retry would be
`h=0.0019929997162210157`, below the fixed `h_min=0.002`. The status is
`minimum_step_reached`, not timeout. No endpoint repair, tightening, violating
sample deletion, inner-loop sparse fallback, nonfinite value, or device transfer
was used.

## What changed

The original dense prototype exposed tensor polynomial operations but its
VDP-named step was Euler (`x+h*f(x)`), ignored requested order, had no explicit
local-time integration or remainder self-map validation, and could not perform
adaptive rejection/retry or dependency-preserving multi-step carry. This work
extended that one canonical module with deterministic basis/route caches,
grouped truncation bounds, local-time integration, a named remainder ledger,
generic ordered polynomial ODE evaluation, true polynomial Picard, fail-closed
self-map validation, and the production scheduler/status/output integration.

## Contract and backend identity

The runner reads `benchmarks/canonical.yaml` and
`benchmarks/three_tool_matched_contract.yaml`: VDP
`x'=y`, `y'=y-x-x²y`, initial box `[1.1,1.4]×[2.35,2.45]`, order 4,
candidate remainder `[-1e-4,1e-4]²`, cutoff `1e-10`, adaptive
`h∈[0.002,0.1]`, physical local time, raw endpoint plus last segment plus full
tube. The lane is hybrid: the numerical Picard/remainder core is dense; the
normalized-insertion scheduler/right-map carry is the existing sparse outer
path. Each accepted segment reports two boundary conversions and zero fallback.
The coefficient layout is `[batch, state_output, monomial_slot]`; the order-4
VDP step has two state outputs, three polynomial variables (two normalized
uncertainty generators and physical local time), and 35 complete-total-degree
slots. Remainders are `[batch, state_output]` and domains are
`[batch, polynomial_variable]`.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| G0 provenance/baseline | PASS | baseline 299 passed/2 skipped; final 343 passed/2 skipped |
| G1 basis | PASS | three variables, order 4, 35 slots, stable fingerprint |
| G2 operator parity | PASS | retained/dropped multiply, range, integration, conversion and adversarial tests |
| G3 analytic one-step | PASS | constant, affine and quadratic true-Picard tests |
| G4 VDP one-step | PASS | h=0.005/0.01 dense/sparse status and coefficients/remainders align |
| G5 short multi-step | PASS | T=0.1/0.5/1.0 exact schedules and reported ranges; 7/34/52 steps |
| G6 T=10 | FAIL | natural minimum-step rejection at 6.3172908799330765 |
| G7 internal performance | PASS (diagnostic) | synchronized CPU/CUDA production-operator timings |

T=4 completes in 142 steps and T=6 in 246. Fresh T=7.5 and T=10 requests both
reproduce the same 308-step natural boundary. The clean-SHA T=1 timing is
17.97 s dense versus 68.03 s sparse, but this is only an internal same-repository
comparison and is not a Flow*/DiffReach/CROWN speed claim.

## First blocker and ledger

At the first terminal attempt, x retains margin `9.9601e-5`; y has margin
`-5.111670937766742e-6`, with image upper remainder
`1.0511167093776675e-4`. The integrated ledger width sums are led by
integration overflow (`9.1683e-5` total, `8.0906e-5` in y), then polynomial
truncation (`4.8027e-5` in y), then remainder×polynomial (`7.7717e-6` in y).
The raw RHS ledger is dominated by polynomial truncation. This is remainder
growth under the unchanged subset test, not a tensor execution failure.

The single allowed factor changed only right-map centering from constant to
range midpoint. It reached `6.390931109681597` (+0.0736402297485208) but still
failed at h_min, with zero repair/fallback. It is explicitly tagged
`single_factor_diagnostic=true` and is excluded from authoritative completion.

## Claims and next priority

The repository may claim a generic validated dense operator/Picard core, exact
short-prefix equivalence to its sparse oracle, hybrid adaptive multi-step
integration, and an exercised CUDA correctness path. It may not claim T=10,
full-dense carry, S4/S5/S6, formal soundness on all hardware, or a cross-tool
speedup.

The one next priority is a validated tighter range for the dropped high-degree
polynomial contribution at the saved terminal pre-state (for example, a tested
blocked/Horner/subdivision enclosure). The ledger shows that further tensor
kernel tuning cannot resolve the mathematical rejection.

Machine-readable evidence and checksums are in
`evidence/generic_batched_tm_backend_vdp_t10/20260804T152536Z`.
