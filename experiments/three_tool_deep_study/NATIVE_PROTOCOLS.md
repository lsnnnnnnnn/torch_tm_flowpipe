# Native protocols

The native regime deliberately does **not** force a common representation.
It asks what each implementation can do with its own propagation and reset
machinery.  The raw endpoint enclosure remains the primary observable.

## Torch TM flowpipe

- Order 1: dependency-preserving raw carry, legacy tightened carry, and
  range-only box restart.
- Orders 2, 4, and 6: dependency-preserving carry where tractable, plus box
  affine reset.
- Order 4 also uses the repository's QR reset path.
- Raw and tightened endpoints are recorded separately.  Tightened endpoints
  are supplemental and are never substituted for raw primary results.

## DiffReach

- The upstream affine flag and restricted quasi-quadratic flag are separate.
- Symbolic remainder windows 1, 10, and 100 and finite remainder-refinement
  rounds 1, 3, and 5 are varied where applicable.
- The scan is the upstream `CT_Dyn_Reach.step_once` pipeline in float64.
  DiffReach's convenience `verify` wrapper hard-codes float32 constructors, so
  the runner creates the identical scan carry with the upstream constructors'
  public dtype arguments set to float64.
- DiffReach output is a floating-point enclosure candidate, not a directed-
  rounding proof.

## Flow*

- Fixed orders 2, 3, 4, and 6 use native `Flowpipe` carry and composition.
- The audited variable-leaf cache correction is enabled.  No candidate
  remainder is written into an accepted flowpipe.
- The original Van der Pol configuration supplies the adaptive-step,
  order-4, symbolic-remainder-window-100 row.
- This checkout does not expose a stable public QR-off/QR-on switch.
  That requested comparison is therefore reported as unavailable, not
  simulated by editing unrelated internals.

All configurations record validation status, completed horizon, raw width,
polynomial and independent-remainder width where exposed, and first/steady
runtime categories where the runtime supports that distinction.
