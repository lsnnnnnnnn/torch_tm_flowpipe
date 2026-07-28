# Flow* remainder and Riccati under-enclosure audit

## Adapter mutation

The historical generated C++ performed, after a successful `advance`, the
equivalent of:

```cpp
next.tmvPre.tms[state].remainder =
    setting.tm_setting.remainder_estimation[state];
```

The repaired `flowstar_stock` path never performs this assignment.
`flowstar_candidate_reinjection_diagnostic` preserves it solely to reproduce
the old output.

For Riccati `[0,0.1]`, order 2, `h=0.01`, cutoff `1e-15`, and candidate radius
`1e-4`:

| Variant | Endpoint lower | Endpoint upper | Width | Remainder width | Analytic containment |
|---|---:|---:|---:|---:|---|
| stock native refinement | -2.501255132095153e-05 | 0.10010008767642772 | 0.10012510022774868 | 2.5100227748658935e-05 | Fail: upper miss 1.2423672382522177e-08 |
| candidate reinjection diagnostic | -0.00012498750000000384 | 0.10017501250000001 | 0.10030000000000001 | 0.0002 | Pass |
| refinement disabled diagnostic | -2.531280001875384e-05 | 0.10010033783129377 | 0.10012565063131253 | accepted initial remainder | Pass |
| full-Picard revalidation diagnostic | same as refinement disabled | same | same | accepted initial remainder | Pass |

Thus H1 is confirmed: reinjection causes the old shift/widening and conceals
the native under-enclosure. It is not a correctness repair.

## First invalid operation

Instrumentation of the exact fixed-step/fixed-order `Flowpipe::advance`
overload records:

- accepted initial remainder:
  `[-3.2530001875001326e-07, 2.5325331293751586e-05]`;
- final remainder-only refinement:
  `[-2.5051320947700027e-08, 2.5075176427711236e-05]`;
- regenerated full Picard image plus polynomial difference:
  `[-1.2511267781623767e-07, 2.517527552828517e-05]`;
- full regenerated subset result: false.

The first source-level operation that loses the inclusion guarantee is the
acceptance of the remainder-only refined image using cached intermediate data.
When `Picard_ctrunc_normal` and the polynomial-difference interval are
regenerated, that box is not a self-map. Polynomial construction, initial
inclusion, composition, `intEval`, direct time evaluation, endpoint
substitution, and wrapper extraction precede/follow this point without being
the first violation in the minimal case.

The audit-only revalidation restores the already accepted initial remainder
when the regenerated check fails. This directly removes the analytic violation,
but it is a conservative fallback—not a proven general upstream correction and
not merged into the user's original Flow* checkout. The remaining algorithmic
reason why the cached remainder-only update disagrees with the regenerated full
Picard image is unresolved.

The full one-factor sweep over order, step, candidate radius, cutoff, and
zero/native refinement is saved in `flowstar_parameter_sensitivity.csv`. It
also repeats the base configuration with `intervalNumPrecision=256` to
distinguish algorithmic refinement behavior from the default 53-bit MPFR
precision. The 256-bit run returns the same first-step upper bound
`0.10010008767642772`, so increased precision does not remove the violation.
Every instrumented intermediate is in
`flowstar_refinement_trace.csv`.
