# Taylor order and Van der Pol diagnostics

## What order means in this prototype

`torch_tm_flowpipe` uses the `order` argument as a **total polynomial degree**
cutoff.  During Taylor-model multiplication and after each Picard iterate, the
polynomial part is truncated so that only monomials with

```text
a_1 + ... + a_n + a_tau <= order
```

are kept.  In dependency-preserving mode, each segment temporarily adds one
local time variable `tau in [0,h]`.  After building the segment, the endpoint is
formed by substituting `tau=h` and dropping `tau`, so the final Taylor model for
the next segment depends only on the original initial-state variables.

This is close in spirit to fixed Taylor order in Flow*, but it is not a claim
that the two implementations keep identical coefficient sets or use identical
internal normalization/preconditioning.  For fair comparison, the numeric `order`
should match, but the comparison should be described as **same requested fixed
Taylor order**, not identical internal Taylor-model representation.

Use

```bash
python experiments/tm_order_audit.py --all --csv outputs/tm_order_audit.csv
```

to record the actual final polynomial degree, term counts, active variables, and
whether a local `tau` remains after endpoint substitution.

## Why Van der Pol can be wider in dependency-preserving mode

The Van der Pol benchmark is

```text
dx/dt = y
dy/dt = y - x - x^2*y
```

The cubic term `x^2*y` amplifies Taylor-model interval remainders.  In
range-only mode, every step compresses the endpoint to a box and restarts from
fresh identity Taylor models with zero remainder.  This loses dependency, but it
also resets accumulated interval remainders into the domain box.

In dependency-preserving mode, the endpoint Taylor model itself is propagated.
This preserves symbolic dependency, but the previous step's interval remainders
remain inside the Taylor models.  When those Taylor models are multiplied in the
nonlinear term `x^2*y`, the arithmetic creates terms such as polynomial-range
`*` remainder and remainder `*` remainder.  Without symbolic remainder or a more
advanced range bounder, these interval remainders accumulate and can dominate
the final width.

Therefore the current result is expected: dependency-preserving is beneficial
when repeated box wrapping is the main error source, but it can regress when
nonlinear remainder multiplication dominates.  This motivates symbolic
remainder, better truncation/range bounding, and eventually adaptive order/step.

Run

```bash
python experiments/diagnose_van_der_pol.py --csv outputs/van_der_pol_diagnostics.csv
```

to decompose Van der Pol final widths into polynomial range width and interval
remainder width.  The RK4 sampling width printed by this script is only a sanity
estimate, not a proof.
