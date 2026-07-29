# Common intermediate representation (CIR v2)

The JSON record is read-only evidence. It never replaces native solver
arithmetic.  `cir_schema.json` is the machine-readable structural contract;
`common.validate_record` enforces the semantic and enclosure gates that JSON
Schema alone cannot express.

Every record contains:

- the system name, state names, polynomial equations, initial domain, requested
  horizon, segment start/end, and requested/accepted step;
- state, local-time, dependency-generator, and noise-generator names, domains,
  and roles (an empty noise list means no noise variable, not missing data);
- sparse exponent tuples and coefficients for every physical-state polynomial;
- the requested/native basis and degree/order, with a capability gap explicitly
  marked `{"availability": "unavailable", "reason": ...}`;
- derived constant, affine-state, time-only, time-state, state-state, and
  higher-order coefficient groups;
- one independent interval remainder and any native structured/symbolic
  remainder metadata, or an explicit unavailable marker;
- separately exported whole-segment tube, raw endpoint, and tightened endpoint;
  tools without a distinct tightening operation use an explicit unavailable
  marker rather than copying the raw endpoint;
- reset/carry policy, success/rejection/failure category and reason;
- backend, dtype, device, repository commit, and setup/propagation/export
  runtime fields (unmeasured components are explicit, never zero-filled); and
- native validation trace and reset/preconditioning metadata.

Mappings:

- Torch serializes each `Polynomial.terms` dictionary directly.
- DiffReach maps `c`, `L`, and `Lt` into exponents and also stores the upstream
  local model and composed parameterization arrays.
- Flow* calls `Flowpipe::compose`, then traverses the resulting native
  `Polynomial<Real>` terms through read-only coefficient/degree accessors.

Schema/round-trip gates evaluate exported polynomials at deterministic points and
compare them with values emitted by the native implementation. Every
polynomial-plus-remainder point interval must lie in the exported tube; raw
endpoint evaluations must lie in the exported raw endpoint; and every raw
endpoint box must lie in the whole tube.

The affine projection retains only constants and affine generator monomials.
Every degree-two-or-higher term is ranged independently over its generator
domain and added to a fresh interval remainder. Existing interval remainder is
preserved. No discarded radius is folded into an existing generator.
