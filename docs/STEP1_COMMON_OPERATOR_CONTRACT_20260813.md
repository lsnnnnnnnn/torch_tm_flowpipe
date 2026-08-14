# Common Flow*–Torch step-1 operator contract — 2026-08-13

Status: `COMMON_STEP1_MATHEMATICAL_INPUT_CLOSED`.

The mathematical state is `(x,y)`, local time is `tau in [0,1/100]`, and the
uncertainty generators are `ux,uy in [-1,1]`:

```text
x(0) = 5/4  + (3/20) ux
y(0) = 12/5 + (1/20) uy
x' = y
y' = y - x - x^2 y
```

The target remainder is `[-1/10000,1/10000]`; the cutoff radius is
`1/10000000000`; the polynomial space is the complete total-degree-O4 support
in canonical variable order `(tau,ux,uy)`, containing 35 monomials.

Flow* uses actual variables `(tau,ux,uy,ut)` and states `(x,y,t)`. For physical
`x,y`, every recorded `ut` exponent is zero. The deterministic clock state is
`t=t_pre+tau`, so eliminating `t` and `ut` from the two-state Torch payload is
algebraic, not a silent projection. Torch uses actual variable order
`(ux,uy,tau)`; the ledger permutes exponents into the canonical order.

Polynomial coefficients, ordinary remainder, degree-truncation remainder,
cutoff remainder, and symbolic-source ledger are separate owners. Segment
evaluation ranges `tau` over `[0,h]`. Endpoint evaluation first substitutes
`tau=h`, merges equal monomials, and then ranges the remaining uncertainty.
Reset input/output and the step-2 prestate are different ledger objects.

The machine contract is emitted by
`experiments/build_step1_common_contract.py` as `common_contract.json`,
`basis_mapping.json`, `support_complete_o4.json`, both canonical initial-state
files, and `initial_state_equivalence.json`.

This status is mathematical. It does not assert that a binary runtime encoding
contains the exact set. Gate D performed that distinct check and found the
fail-closed witness documented in the independent-oracle report.
