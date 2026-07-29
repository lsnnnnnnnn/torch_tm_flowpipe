# Component attribution protocol

The attribution tables are controls, not cross-tool rankings.  They use the
same raw endpoint rows as the controlled and native experiments, then retain
the named configuration dimensions needed to compare changes within one tool.

- Torch: raw versus legacy-tightened endpoint, dependency carry versus affine,
  QR, and box/range resets, and orders 1, 2, 4, and 6.
- DiffReach: affine versus restricted quasi-quadratic support, symbolic
  remainder windows 1/10/100, and refinement rounds 1/3/5.
- Flow*: fixed orders 2/3/4/6, the original adaptive symbolic-remainder
  configuration, refinement disabled, stock cached refinement, full-Picard
  revalidation, the variable-leaf cache patch, and candidate remainder
  sensitivity.

Widths are split into polynomial interval range, exposed independent interval
remainder, exposed structured remainder, and a residual dependency/reset
field.  The residual is a bookkeeping diagnostic; it is not treated as a
fourth native remainder object.  DiffReach's structured symbolic state is
reported as unavailable when the upstream public output does not expose a
separate interval width for it.

## Matched basis

The Torch-engine attribution uses one float64 backend, order-3 arithmetic
ceiling, two Picard constructions, growth validator, interval range backend,
step size, initial box, and no reset.  Only the retained dictionary changes:

- B1: constant, local time, and affine state generators.
- B_DR: B1 plus local-time squared and local-time/state terms.
- B2: all monomials of complete total degree at most two.
- B3: complete quadratic dependency/state terms plus their one-local-time
  integration lift (`tau*xi_i*xi_j`); general cubic state terms and higher
  time powers are excluded.

Every projected term is logged with exponent, coefficient, outward interval
contribution, and destination (`fresh_independent_interval_remainder`).
