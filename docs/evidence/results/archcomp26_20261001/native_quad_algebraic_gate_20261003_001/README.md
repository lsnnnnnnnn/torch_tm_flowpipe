# Saved QUAD first-step algebraic gate

This read-only check uses the saved repaired-library first-call/one-step RPC,
directional CSV, and terminal-axis CSV. It does not start Flow*, CROWN, a
numerical ODE solver, or the original long run. Reproduce from the repository
root with:

```bash
python3 tools/check_native_quad_algebraic_gate_nohash.py > /tmp/quad_algebraic_audit.json
```

The script reads the prior 1,000-substep directed-rounding Picard audit and
rechecks its 27 contained numerical comparisons against the saved CSV. It
adds algebraic physical-state bounds for the five prior inconclusive columns,
with exact `Fraction` comparison against saved binary64 intervals. The
output is [AUDIT.json](AUDIT.json).

## First-step contract and proof

The proof domain is the **saved first-call RPC/recenter TM input box**.
It does **not** cover the C++ source's entire first physical initial box:
for `x1`–`x3`, the source box starts at binary64 `-0.4`, while the saved RPC
lower bounds are `-0.39999999999999997`, one binary64 ULP higher. The
[initial-box audit](../native_quad_initial_recenter_gate_20261003_001/README.md)
records this split/recenter boundary separately. The script
rounds each saved `T`, `u_min`, and `u_max` to binary32 as the frozen C++
`asFloat()` does, then forms `u_j = T_j x(0) + center_j + r_j`,
`r_j ∈ [-radius_j,radius_j]`. For these saved values it verifies that
`center_j` and `radius_j` are exactly representable in binary64, the type
used by the C++ midpoint calculation. Input bounds are the saved binary64
RPC values. The mathematical paper ODE constants `h=0.005`, `9.81`, `1.4`,
and `0.054` are treated as exact decimal rationals. These choices and the
arithmetic are recorded in the audit. It does not independently validate
the CROWN response or native parsing and rounding.

Here `x12(0)=0`, `x12'=0`, and the three injected controls are constant over
this first step. Therefore the paper ODE gives exact RPC-domain endpoint
extrema `x10(h)=h u2/0.054` and `x11(h)=h u3/0.054`. These **physical-state**
bounds lie inside the saved composed intervals and numerically inside the
pre columns. The latter comparison does not establish inclusion of the
`tmvPre` symbolic set. Unlike the previous plain interval propagation, this
calculation does not accumulate substep dependency. All four numerical
x10/x11 comparisons pass even against the *unexpanded* saved binary64
bounds. The narrowest direct margin, about `3.30e-19`, is for the paper ODE
with exact decimal `h=0.005` and `0.054`. Replacing only those two constants
by their nearest binary64 values leaves the four numerical comparisons
inside the saved columns, as a sensitivity check. It does not validate
Flow*'s decimal parser or its internal arithmetic.

For `x6`, the script retains the shared initial-state symbols in
`x6(0) - h u1/1.4`. Its exact affine RPC-domain range is approximately
`[-0.445705256279, 0.349566624959]`. A strict first-exit bound proves,
for all `t∈[0,h]`, `|x4|,|x5|,|x6|<1`, `|x7|,|x8|<0.01`,
`|x10|<0.0015`, and `|x11|<0.001`. The audit records each strict closure
inequality. Since `|1-cos(x7)cos(x8)|≤0.01²`, the remaining integral
`∫[x11 x4 − x10 x5 + 9.81(cos(x8)cos(x7)−1)]dt` has magnitude at most
`0.000017405`. The resulting RPC-domain endpoint range is about
`[-0.445722661279, 0.349584029959]`, inside the saved **composed** `x6`
range `[-0.446199541413, 0.349816005255]`. This new comparison also
passes against the unexpanded binary64 bounds.

## Result and meaning

| Saved observer view | Comparisons contained | Interpretation |
|---|---:|---|
| Composed x1/x2 whole-step tube and propagated endpoint | 8/8 | Physical-state directional projection, from prior Picard audit |
| Composed terminal axes | 12/12 | Physical-state endpoint; three newly resolved by algebra |
| Pre-composition terminal-axis columns | 12 physical bounds numerically inside 12 columns | Column comparison only; no independent `tmvPre` set inclusion claim |

The **20/20 composed physical-state comparisons** form a first-step plant
gate only for the saved RPC/recenter domain under the exact-decimal paper
ODE contract. `Flowpipe`
represents `tmvPre ∘ tmv`; the 12 `tmvPre.intEval` axis columns are
separate numeric diagnostics. A physical trajectory bound landing in a pre
column does **not** prove that the pre symbolic set itself is contained, and
pre columns are not part of the physical octagon observer.

This gate applies only to the copied VAR-tail repair candidate, the saved
first-call RPC/recenter domain, its affine-plus-residual control, and
`t∈[0,0.005]`.
The original frozen library still has the saved relaxed-control
under-enclosure witness. The excluded strip of the first physical box,
the other 1,023 boxes, future control calls,
controller/CROWN soundness, true network trajectories, and the full
50-period property are unverified. **Native production promotion remains
closed.**
