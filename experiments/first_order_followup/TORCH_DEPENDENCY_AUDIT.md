status: diagnostic
valid_for_commit: unknown
superseded_by: docs/ALGORITHM.md
allowed_use: diagnostic only

# Torch dependency-preserving audit

## Exact diagnostic

The audit uses the frozen harmonic oscillator box, `h=0.01`, float64 CPU, and
records all segments through `T=10`.  Every segment records:

- center and affine coefficients;
- active polynomial variables and domains;
- polynomial, remainder, and total widths for endpoints and tubes;
- coefficient L1 and range width by monomial group;
- candidate and validated remainder traces;
- validation attempts and local-time add/drop semantics;
- discarded-term ranges for finite-basis runs; and
- an affine-generator condition surrogate plus reset statistics.

The old local-time variable is substituted at `tau=h` and dropped before the
next segment.  A fresh local-time variable is introduced by each step.
Endpoint widths equal polynomial interval width plus independent-remainder
width.

## Isolated mechanism

At total degree one, the essential harmonic term `tau*xi` has degree two and
is ranged into an independent interval remainder.  Dependency-preserving carry
keeps the original affine generator coefficients while that loss accumulates.
For the first five steps of the focused forensic trace, the symmetric
remainder radii grow approximately:

```text
0.0010125, 0.00203778, 0.00307601, 0.00412734, 0.00519195
```

The range-only path instead turns each endpoint box into a fresh identity
affine parameterization.  At step 5 its remainder radius is approximately
`0.00105413`; previous width is now represented by new affine generators
rather than left in the independent remainder.  The clean affine-box reset
reproduces this effect.  The QR policy is also recorded; for this symmetric
two-dimensional diagnostic it does not materially beat box reset.

Thus range-only's advantage is an implementation/reparameterization effect.
It is not evidence that dependency erasure is fundamentally tighter.

## Comparisons

`run_torch_followup.py` emits five diagnostic paths:

1. current dependency-preserving;
2. current range-only;
3. clean affine box recenter/rescale;
4. QR-preconditioned affine reset; and
5. an exact rotation oracle.

The oracle is a diagnostic only and is never mixed into benchmark timing.
Plots 2 and 3 in each result directory show the component decomposition and
policy comparison.
