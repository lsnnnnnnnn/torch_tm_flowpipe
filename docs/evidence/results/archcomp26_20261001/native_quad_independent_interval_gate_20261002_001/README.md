# Independent whole-set interval check for the short native QUAD gate

This is a **new local, read-only check of saved evidence**. It did not run
Flow*, contact CROWN, change either library, or restart the 50-period QUAD
job. The checker is [`tools/check_native_quad_fullset_interval_gate_nohash.py`](../../../../../tools/check_native_quad_fullset_interval_gate_nohash.py).
It reads the original-library `_005` and copied VAR-tail repair `_007` RPC,
octagon, and terminal-axis receipts. No content digest was calculated.

The first RPC input bounds enclose the **whole first initial box**. The checker
reconstructs the C++ float32 `T`, `u_min`, and `u_max` conversion and encloses
every `u=T·x(0)+center+r`, with each residual `r` in its saved interval, by an
interval control hull. This hull loses affine correlation, so it is broader
than the injected control set. The paper-equation 12-state ODE is then
propagated for **every initial state and every control in that hull**, over
the complete `t∈[0,0.005]` step. It uses 1,000 substeps. For each substep,
the saved `picard_inclusion_passed` means the interval Picard image lies
strictly inside a candidate tube. Sine and cosine use interval Taylor
polynomials plus explicit Lagrange remainders for `|angle|<0.1`; all Decimal
arithmetic is directed outward at 60 digits. The union of those substep
tubes covers every time in the small step. This is a mathematical ODE check,
not a finite trajectory sample.

The resulting independent tube directions (`x1`, `x2`, `x1+x2`, `x1−x2`),
propagated endpoint directions, and all 12 endpoint axes are compared with
the saved native intervals, enlarged by one binary64 ULP to cover text
serialization. There are 32 comparisons per library:

| Saved library | Independent intervals contained by saved intervals | Comparisons not established |
|---|---:|---|
| [Original `_005`](original_AUDIT.json) | 23/32 | x6 composed; x7/x8 pre and composed; x10/x11 pre and composed |
| [VAR-tail copy `_007`](repaired_AUDIT.json) | 27/32 | x6 composed; x10/x11 pre and composed |

For the repaired copy, **all eight x1/x2 tube and endpoint direction
comparisons pass**, as do x7 and x8 endpoint pre/composed comparisons. The
independent whole-set x7 endpoint enclosure is about
`[-3.445e-7,3.641e-6]`, inside the repaired saved
`[-2.383e-6,5.675e-6]`; x8 is about `[-1.585e-6,2.386e-6]`, inside
`[-3.570e-6,4.370e-6]`. The original saved x7/x8 endpoints exclude those
independent enclosures. That failed **interval containment** is consistent
with the earlier concrete finite witness; failure of this particular broad
enclosure alone is not a new point counterexample.

The repaired gate remains **inconclusive as a full 12-axis certificate**.
The x6 composed interval is much narrower than the plain interval hull
because Flow* retains initial-state/control correlation; this checker does
not. The independent x10/x11 hulls extend beyond the saved endpoints by only
about `2–3e-15`; this also fails strict containment but does not by itself
identify a reachable counterexample. Those five comparisons remain open.
The check does not validate CROWN bounds, the neural network, Flow* parsing
or rounding, the other 1,023 initial boxes, later control calls, or `T=5`.
**Native production promotion remains blocked.**
