# Numerical soundness and common defect diagnostic

Native validation and numerical arithmetic are not interchangeable across the
three tools.

- Flow* uses MPFR-backed interval arithmetic and is the strictest numerical
  enclosure path in this study.  Primary Flow* rows use the audited cache fix.
- Torch uses float64 tensors with explicit outward `nextafter` inflation in its
  interval operations.  These rows are labelled floating-point enclosure
  candidates because this is not an MPFR proof path.
- DiffReach runs the upstream JAX propagation in float64, but does not use a
  directed-rounding interval backend.  Its rows are also floating-point
  enclosure candidates.

The common CPU diagnostic exports the local polynomial `p` and computes

`d = dp/dtau - f(p)`

without altering native solver arithmetic.  Sparse polynomial differentiation
and composition are shared across all tools.  Tiny Riccati and coupled-system
identities are unit-tested with exact `Fraction` coefficients.  General ranges
use conservative float64 interval operations with outward `nextafter`
inflation.

The reported Jacobian comparison constant is an infinity-norm row-sum bound
over the native whole-tube box.  The common radius uses a Gronwall comparison,
including the largest magnitude of the exposed independent interval remainder
as a conservative initial mismatch.  It is a diagnostic certificate for the
exported polynomial core, not a replacement for a tool's native validator.
Native and common radii therefore remain separate columns.

Analytic containment is mandatory for Riccati and harmonic rows.  For the
coupled quadratic and Van der Pol systems, the collector integrates every
corner/midpoint tensor-product sample with SciPy DOP853 at `rtol=1e-12` and
`atol=1e-14`, then checks raw endpoints and five within-step tube times.
Those deterministic trajectories are bug-catching checks only.  Their pass
status is stored separately from native validation and analytic containment,
and no report text treats a floating-point trajectory as a proof of enclosure.
