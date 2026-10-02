# QUAD all-box first-step ledger audit

This [independent saved-ledger audit](AUDIT.json) ran **after** the adaptive
plant gate wrote its terminal result. Its [scanner](audit_saved.py) reads
the original 1,024-box source/RPC/observer receipts and four completed
ledger segments: fixed-bootstrap lanes `0` and `1–6`, adaptive-bootstrap
lane `7`, and adaptive-bootstrap lanes `8–1023`. It does not run Flow*,
CROWN, a neural network, or the earlier reachability experiments, and it
does not edit their files.

The scanner found every lane `0–1023` exactly once, with no missing or
extra ledger row. It checked each segment's result and available checkpoint.
It reconstructed **3,072 exact-rational control hulls** from the saved
float32 RPC coefficients, verified each source box lies inside its RPC box,
and independently recomputed **seven strict first-exit bootstrap
inequalities per lane**. The first seven historical records store only the
five state inequalities; the two rate inequalities were independently
reconstructed from their control hulls and fixed rates. The remaining
1,017 records store all seven and their exact adaptive rate choices.

Every saved lane reports 1,000 strict Picard substeps. The scanner checked
the exact identity, order, numerical bounds, and Boolean inclusion for all
**20,480/20,480 composed physical-state comparisons** against the original
saved observer CSV. All also fit the parsed native binary64 bounds without
the extra outward one-ULP allowance. The composed-state set is eight
`x1/x2` tube/endpoint directions and twelve endpoint axes per lane.
The three `x6/x10/x11` endpoint bounds use the recorded exact-rational
algebraic argument; the other seventeen use the directed Decimal Picard
bound. The scanner validates the saved arithmetic and completion records;
it does not independently execute the 1,024,000 Picard substeps again.

This result covers **only the first `h=0.005 s` plant step** for each of
the 1,024 source-defined initial boxes under its saved first-call
affine-plus-residual control hull. It does not cover the other nineteen
small steps of the first control period, later controls, `T=5`, or the
reach-and-remain property. It does not independently validate CROWN/NN or
Flow* parsing and internal floating-point soundness. The native octagon
production gate remains **closed**.
