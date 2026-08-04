# Algorithm contracts

An interval stores lower and upper tensors. Arithmetic expands finite float64
results with `torch.nextafter` where implemented. The repository does not
claim universal hardware-independent directed rounding or proof-grade real
arithmetic.

A sparse polynomial maps deterministic exponent tuples to coefficients and
applies an explicit requested total-degree cutoff. `requested_order`,
`effective_order`, and `effective_degree` are recorded separately; equal
nominal order across tools does not imply equal basis, retained monomials, or
work. A Taylor model is a polynomial plus an interval remainder over a
declared domain.

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

Bound location and refinement are orthogonal dimensions:

| Bound semantics | Location | Refinement |
|---|---|---|
| `raw_endpoint` | endpoint at requested evaluation time | raw |
| `tightened_endpoint` | endpoint at requested evaluation time | tightened |
| `segment_box` | one accepted segment | raw |
| `tube_box` | full tube | raw |

Only a completed raw endpoint whose exporter semantics and backend identity
are explicit can become primary-comparable. Sampling containment is regression
evidence, not a formal enclosure proof.
