# Branch Decision

Decision: NEEDS_MORE_WORK

## Evidence

- Previous best `flowstar_style_o6_candidate8_output6_cutoff` reached t=`2.4007376673997931`.
- New width-control `flowstar_style_o6_candidate8_output6_cutoff_symqueue` reached t=`0.094531250000000011`.
- Horizon 5 reached: no.
- Width ratio improved over the validated same-run horizon: yes.
- Reset boxes shrank against the previous best: yes.
- After-width-control oracle: Flow* validated the same rejected local box at t=`0.094531250000000011` with h_try=`0.00263671875`.
- Failure mode: `Picard residual not subset of target remainder`.

## Recommendation

The queue is tighter over its short horizon but fails much earlier; implement normalized insertion/composition next.
