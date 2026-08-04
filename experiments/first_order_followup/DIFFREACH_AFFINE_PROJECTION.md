status: diagnostic
valid_for_commit: unknown
superseded_by: docs/RESULTS_STATUS.md
allowed_use: diagnostic only

# Experimental strict-affine DiffReach projection

DiffReach's plant polynomial has the restricted form

```text
P(t,z) = c + L[t,z] + t Lt[t,z].
```

The last part contains `t^2` and `t*z_j`.  Simply clearing `Lt` is unsound.

## Projection

`strict_affine_projection` processes every point-valued `Lt` coefficient over
the actual local box:

```text
I_k = range(a_k m_k(t,z))
I = outward_sum_k I_k
m = midpoint(I)
c' = c + m
R' = R + (I - m)
L' = L
Lt' = 0
```

All products, sums, and residual endpoints receive `jax.numpy.nextafter`
outward expansion.  An asymmetric `[0,h]` time range is used directly.
The pre-existing remainder is added, not replaced.  Overflow is a fresh
independent interval and is never embedded in an existing generator.

Consequently, for every point in the local box,

```text
P(t,z) + R ⊆ c' + L'[t,z] + R'.
```

The adapter applies this projection only after the stock restricted
quasi-quadratic construction and native Picard validation, then performs the
same affine composition/normalization used by the plant-only baseline kernel.
The external DiffReach repository remains unchanged.

## Tests and parity

`test_diffreach_projection.py` covers positive and negative `t^2`, positive and
negative `t*z`, `[0,h]`, multiple generators, nonzero pre-existing remainder,
pre/post polynomial containment, zero final `Lt`, and identity when `Lt` is
already zero.  DiffReach does not support interval-valued polynomial
coefficients, so that conditional case is documented rather than fabricated.

`test_diffreach_parity.py` compares the baseline custom plant core against a
literal transcription of the corresponding stock `CT_Dyn_Reach.step_once`
plant statements, for `TRUNCATE_TO_AFFINE=False` and `True`.  Direct import of
the public stock class is blocked in this checkout because its module requires
the absent optional `jax_verify` package.  Picard rounds, initial seed,
symbolic window, normalization, and configuration are held equal.

Rows are labeled `diffreach_experimental_strict_affine`; they are not claims
about stock DiffReach.
