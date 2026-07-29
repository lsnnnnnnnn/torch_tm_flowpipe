# Controlled protocols

## A — identical one-step input

Every solver receives the same polynomial ODE, state order, componentwise
initial box, step `h`, and one-step horizon. Whole validated tubes and raw
`tau=h` endpoints are separate rows. Torch fixed-time tightening is excluded
from primary rows. Each row records polynomial, remainder, and total width,
monomial-family support, validation, analytic containment where available, and
runtime components.

The local bases remain native and are labeled; this protocol compares local
enclosure construction, not a nominal integer order.

## B — common affine carry

Between steps every endpoint must have exactly

`x = c + A xi + I`.

Constants and affine generator coefficients are retained. Every higher-degree
term is interval-ranged over its current generator domain and added to a fresh
independent remainder, and the previous remainder is preserved. The projection
never embeds discarded radius in an existing generator. Torch uses its natural
order-one local segment, DiffReach uses its restricted quasi-quadratic local
construction followed by the audited strict-affine projection, and Flow* uses
corrected order-two local construction followed by native `ctrunc(...,1)`.

This is the closest valid interpretation of the requested “first-order”
comparison. It controls carried representation, not local construction basis.

## C — common box carry

The previous raw endpoint is evaluated to a componentwise box and the next step
starts from a fresh box. No affine dependency is retained. This is a
reset/wrapping control and is never described as native solver performance.

## Correctness admission

Primary rows require native validation, finite values, raw endpoint contained
in the tube, analytic containment for Riccati and harmonic, explicit endpoint
semantics, and no post-validation mutation. Flow* generated sources are scanned
for the historical post-`advance` remainder assignment; its presence is a hard
failure.
