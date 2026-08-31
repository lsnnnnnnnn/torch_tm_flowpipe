# C4 polynomial-plant reference configuration

The formal reference lane is named
`flowstar_like_polynomial_plant_reference`. It is an explicit frozen
configuration, not a portfolio and not an automatic mode selector.

`FlowstarLikePolynomialPlantConfig.van_der_pol()` binds the accepted native C3
contract: the canonical order-4 Van der Pol plant, CPU binary64, Flow*-pinned
post-accept refinement, accepted-boundary queue capacity 100, constant-centered
normal insertion, the accepted proactive depth-1 subdivision policy on
polynomial truncation, and the frozen native step policy through T=10.

`FlowstarLikePolynomialPlantConfig.brusselator()` binds the accepted generic C4
contract: the canonical ordered-term Brusselator expression, exact-decimal
initial box `[1.48,1.52] x [2.98,3.02]`, order 6, fixed `h=0.02`, 1000 steps,
remainder `[-1e-4,1e-4]`, cutoff `1e-10`, validation epsilon `1e-12`, accepted-
boundary queue capacity 1000, and generic post-accept raw-remainder refinement.

Both contracts freeze the Flow* replay ceiling at 491 evaluations,
`STOP_RATIO=0.99`, whole-vector atomic subset commits, CPU float64 outward
rounding, accepted-only queue mutation, normal insertion/reset, and v5 full-
queue checkpoint/rollback semantics. Endpoint repair is disabled.

Legacy validation/reset strings remain unchanged. The reference object only
names the accepted combination; callers must explicitly request it.
