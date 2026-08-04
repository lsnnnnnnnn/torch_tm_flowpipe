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

The dense representation uses the same semantics over a deterministic complete
total-degree basis. Cached routes aggregate retained products by slot and
dropped products by exponent before intervalization. Integration routes divide
by the new local-time exponent; degree overflow and cutoff are added to named
remainder-ledger categories. VDP order 4 has two output states but three
polynomial variables and 35 slots.

One propagation step constructs local time, performs Taylor/Picard expansion,
classifies truncated terms, validates a remainder candidate, and emits
separate segment/tube and endpoint objects. A rejected step is incomplete and
must not expose a partial result as a completed horizon.

The dense step performs `order` polynomial Picard iterations on physical
`tau∈[0,h]`, then evaluates the candidate remainder self-map. The Flowstar raw
compatibility lane replays the ordered RHS expression at effective degree
`order-1`; expression order is therefore part of the contract. Batch acceptance
is an all-leaf reduction, and rejection publishes no endpoint.

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
