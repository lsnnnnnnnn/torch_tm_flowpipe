# Algorithm contracts

An interval stores lower and upper tensors. Arithmetic expands finite
float64 results with `torch.nextafter` where implemented, but the project does
not claim universal hardware-independent directed rounding.

A sparse polynomial maps deterministic exponent tuples to coefficients and
applies an explicit total-degree cutoff. A Taylor model is a polynomial plus
an independent interval remainder over a declared domain.

One propagation step constructs local time, performs Taylor/Picard expansion,
classifies truncated terms, validates a remainder candidate, and emits
separate segment/tube and endpoint objects. A rejected step is incomplete and
must not expose a partial result as a completed horizon.

Multi-step modes are distinct:

- `range_only` evaluates the endpoint to a box and reinitializes the next
  step, losing symbolic dependency by design.
- `dependency_preserving` carries a Taylor model into the next step.
- Flowstar-compatibility variants are diagnostic contracts with explicit
  basis, remainder, and step-policy identities.

Raw endpoint, tightened endpoint, segment box, and tube box are different
bound semantics. Only raw endpoints marked primary-comparable enter the
cross-tool primary table.
