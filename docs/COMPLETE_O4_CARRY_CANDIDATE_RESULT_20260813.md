# Complete-O4 carry candidate result — 2026-08-13

Status: `NO_FIX_AUTHORIZED`.

No carry candidate was implemented, selected, or benchmarked. Consequently no
L1/L2/L3 success level is claimed.

## Why implementation is not authorized

The source delta is localized, but the Goal requires every soundness gate to
close before a narrowing implementation:

1. The complete post-step Flow* coefficients, domains, and `Phi_L/J` source
   queue cannot be losslessly exported and imported, so the required full 2×2
   same-prestate attribution is unavailable.
2. The pinned stock Flow* source has an independent analytic and directed-MPFR
   scalar-affine under-enclosure witness. Its correctness gate remains open, so
   copying its narrower behavior is not a soundness argument.
3. Exact affine/quadratic/cubic fixtures establish the dependency mechanism and
   containment of simple independently intervalized ranges, but there is no
   independently proved outward primitive covering nonlinear multiplication,
   complete-O4 truncation, cutoff, renormalization, queue/source merging, and
   the full one-step transition together.

Narrowness, a T10 outcome, or absence of sampled witnesses would not substitute
for those obligations. Existing optional Horner/symbolic-queue experiments are
therefore diagnostics, not a promoted repair.

## Evidence retained despite no candidate

The audit still closes useful facts:

- baseline results reproduce exactly at `.01`, `1`, `3`, and `6.32`;
- all four Flow* minima are ordinary positive widths above `1e-9`;
- the benchmark-to-output runtime path and active feature set are source-mapped;
- the first decision-relevant carry divergence is step 1 → step-2 scale;
- exact affine, quadratic, and cubic dependency fixtures pass containment;
- high-precision point replay finds no Van der Pol containment witness, while
  remaining explicitly non-probative for the continuum.

## Single next question

Can a lossless binary state-and-queue fixture be paired with an independently
outward-rounded source-ledger carry primitive that proves the complete one-step
O4 containment contract, including nonlinear multiply, truncation, cutoff, and
renormalization?

Until that question is answered, `NO_FIX_AUTHORIZED` is the only sound result.
