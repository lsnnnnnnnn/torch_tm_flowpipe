# Flowstar Next Mechanism Decision

Decision: real Flow*-style symbolic queue propagation through Phi_L/J/scalars plus Horner insertion.

## Evidence

- `normal_eval` did not improve the o4 or o6 normalized insertion baselines: best remained t≈7.496, and old vs normal right-map range maxima remained equal at about 21.8845.
- Scalar recentering did not improve the branch enough to change the horizon conclusion.
- Split symbolic queue variants did not produce a horizon improvement.
- The bounded Horner diagnostic produced real stage outputs but showed no material direct-vs-Horner range reduction: deltas were roundoff-sized at the early validated reset.
- Full high-degree failure-neighborhood Horner diagnostics were attempted and did not return in practical time, so the clean-room Horner implementation is diagnostic-useful but not yet a production replacement.
- Flow* one-step oracle evidence is mixed: the width-control local oracle validates a PyTorch-rejected local box, but the later symqueue-split failure box is already too wide for Flow* to validate.

## Rationale

The evidence rules out another normal-domain range evaluation tweak, scalar-only correction, or split symqueue parameter sweep. It also does not justify running `normalized_insertion_horner` to h10. The next source-guided mechanism should carry Flow*-style symbolic queue information through the actual linear/preconditioning maps (`Phi_L`, `J`, and scalar channels) and combine that with Horner insertion, rather than treating Horner insertion alone as the missing mechanism.

## Not Chosen

- `preconditioning/scaling matrix detail`: still plausible, but current diagnostics point more directly at missing propagation/accounting semantics than at only scale construction.
- `better polynomial/range representation`: useful later, but it does not explain the source-map gap by itself.
- `stop branch as not merge-ready`: the branch is not merge-ready, but the next mechanism is specific enough to continue rather than stop.
