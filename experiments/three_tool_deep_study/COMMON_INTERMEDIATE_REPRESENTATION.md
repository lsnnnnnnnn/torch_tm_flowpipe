# Common segment representation

The JSON record is read-only evidence. It never replaces native solver
arithmetic.

Every record contains:

- state dimension, local-time domain, generator domains, variable names, roles,
  and the tool's local-time index;
- sparse exponent tuples and coefficients for every physical-state polynomial;
- derived constant, affine-state, time-only, time-state, state-state, and
  higher-order coefficient groups;
- one independent interval remainder and any native structured/symbolic
  remainder metadata;
- a separately exported raw endpoint and whole-segment tube;
- native validation trace and reset/preconditioning metadata.

Mappings:

- Torch serializes each `Polynomial.terms` dictionary directly.
- DiffReach maps `c`, `L`, and `Lt` into exponents and also stores the upstream
  local model and composed parameterization arrays.
- Flow* calls `Flowpipe::compose`, then traverses the resulting native
  `Polynomial<Real>` terms through read-only coefficient/degree accessors.

Round-trip gates evaluate exported polynomials at deterministic points and
compare them with values emitted by the native implementation. Every
polynomial-plus-remainder point interval must lie in the exported tube; raw
endpoint evaluations must lie in the exported raw endpoint; and every raw
endpoint box must lie in the whole tube.

The affine projection retains only constants and affine generator monomials.
Every degree-two-or-higher term is ranged independently over its generator
domain and added to a fresh interval remainder. Existing interval remainder is
preserved. No discarded radius is folded into an existing generator.
