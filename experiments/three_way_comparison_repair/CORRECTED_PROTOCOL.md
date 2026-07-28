# Corrected comparison protocols

All primary comparisons use the same ODE, initial componentwise box, requested
step, state component, absolute time, and interval meaning. Internal polynomial
bases and rounding backends remain tool-specific and are always reported.

## A — `one_step_tube`

Compare the whole validated segment over `[0,h]`: Torch `tube_tm`, DiffReach
composed TM on the full time domain, and the stock composed Flowpipe range.

## B — `one_step_raw_endpoint`

Compare direct `t=h` substitution in each validated segment. Torch uses
`endpoint_raw_tm`; DiffReach fixes the composed segment time coordinate; Flow*
evaluates the stock composed Flowpipe endpoint. No endpoint tightening or
post-advance remainder overwrite is permitted.

## C — `common_box_raw_endpoint_carry`

After each native segment, extract the raw endpoint componentwise box and use
exactly that box as the next initial set. This controls the external carried
representation but intentionally destroys all native cross-step dependency.
H4 is therefore confirmed: the protocol hides native dependency behavior.

## D — `native_representation`

Flow* carries the completely stock returned Flowpipe. DiffReach carries its
upstream affine or restricted quasi-quadratic symbolic representation as
separate variants. Torch reports dependency-preserving raw carry and legacy
tightened carry as separate variants.

## E — `deliberate_low_order_stress`

Torch uses order 1, DiffReach uses its affine flag, and Flow* uses its minimum
legal fixed order 2. This deliberately unmatched diagnostic may expose failure
modes but may not support same-order or general performance claims.

## F — `known_working_tool_sanity`

Run the original Flow* Van der Pol configuration and two identical-settings
harnesses to `T=10`. This establishes installation/harness function and parity;
it is not ranked against Protocol E.

Primary tables exclude diagnostic candidate reinjection, no-refinement, and
full-revalidation variants. A width comparison is emitted only at equal
absolute times and only through the requested `interval_kind`.
