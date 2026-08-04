status: historical
valid_for_commit: unknown
superseded_by: docs/EXPERIMENT_PROTOCOL.md
allowed_use: provenance only

# Protocols and finite bases

## Protocol A: `native_semantics`

The frozen artifact is the historical Protocol-A record:

- Torch complete total degree one;
- DiffReach's affine flag with transient restricted `Lt`; and
- Flow* fixed order one unsupported.

It is not rerun.

## Protocol B: `matched_affine_carry`

Every carried segment state has only a constant, affine state-generator terms,
and an independent interval remainder.

- Torch uses the B1 finite dictionary and affine box recenter/rescale.
- DiffReach uses the experiment-local sound `Lt` projection.
- Flow* constructs and validates locally at order two, lowers the endpoint to
  degree one with `ctrunc`, and starts the next public step from that affine
  endpoint.

Metadata separates local construction, carried basis/degree, projection,
reset, validator, and backend.

## Protocol C: `complete_degree_two_reference`

- Torch carries its complete total-degree-two Taylor model.
- Flow* carries its fixed complete order-two flowpipe.
- DiffReach carries its native restricted `{1,z,t^2,t*z}` form and is explicitly
  labeled `restricted_quasiquadratic_not_complete_degree_2`.

## Torch basis ablation

The experiment-local projection supports:

```text
B1   = {1, tau, xi}
B_DR = {1, tau, xi, tau^2, tau*xi}
B2   = every monomial with complete total degree <= 2
```

Every removed monomial is ranged independently on its actual domain and added
to the independent remainder.  Exponent, coefficient, interval, and width are
logged.  Picard iterations, step, initial set, box-reset policy, validation,
float64 dtype, and CPU device are fixed across B1/B_DR/B2.

An important literal-basis consequence is that `tau*xi^2` has total degree
three and is absent from both B_DR and B2.  After substituting local time at the
endpoint and applying the same affine reset, B_DR and B2 can therefore coincide
on these systems.  The report treats the requested hypotheses as tests, not
assumptions: unsupported hypotheses remain labeled as such.
