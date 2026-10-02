# Frozen QUAD first-box recentering audit

This is a read-only source and saved-RPC audit. It starts no Flow*, CROWN, or
ODE run and calculates no content digest. Reproduce from the repository root:

```bash
python3 docs/evidence/results/archcomp26_20261001/native_quad_initial_recenter_gate_20261003_001/check_initial_recenter_nohash.py
```

The output should match [AUDIT.json](AUDIT.json). The calculation uses exact
`Fraction` values and explicitly rounds each relevant operation to binary64,
which is the frozen Flow* library's 53-bit MPFR precision. It checks the
saved repaired gate's [RPC](../native_quad_var_tail_repair_gate_20261002_007/quad/on/rpc.json);
the earlier [unrepaired gate RPC](../native_quad_sr_octagon_gate_20261002_005/on/rpc.json)
has the same first three lower bounds.

## Source and endpoint trace

The original frozen QUAD source is
`docs/evidence/results/archcomp26_20261001/native_quad_paper_full50_001/build/archcomp/Quadrotor/quad_paper_full50.cpp`.
The small [source excerpt](FROZEN_QUAD_EXCERPTS.txt) shows
`Interval init_x1(-0.4,0.4)` through `init_x3`, each split into eight parts,
the first `Flowpipe(X0)`, and the `tmvPre.intEval(...).inf()` value sent as
RPC `input_lb`. The isolated repair gate retains this initialization.

The exact server library inspected through read-only SSH is at
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_quad_paper_build_001/flowstar/flowstar-toolbox`.
The saved [library excerpt](FROZEN_LIBRARY_EXCERPTS.txt) records these
implementation lines: `include.h:36` and `Interval.cpp:15` set 53-bit MPFR;
`Interval.cpp:683–689` constructs an interval from the binary64 C++
literals; `Interval.cpp:824–843` copies the original lower bound into the
first split; `TaylorModel.h:2838–2875` converts each split interval to
`center + radius·ξ`, `ξ∈[-1,1]`; `Interval.cpp:927–938` computes the radius
only as `upper − center`; and `Continuous.cpp:260–268` uses that Taylor model
as the initial `tmvPre`.

For each of `x1`, `x2`, and `x3` in the first split:

| Quantity | Exact binary64 hex |
|---|---|
| Original C++ interval lower bound | `-0x1.999999999999ap-2` |
| First split upper bound | `-0x1.3333333333333p-2` |
| Recentered center | `-0x1.6666666666666p-2` |
| Recentered radius | `0x1.9999999999998p-5` |
| Recentered Taylor-model lower bound | `-0x1.9999999999999p-2` |
| Saved RPC `input_lb` | `-0x1.9999999999999p-2` |

The recentered lower bound is **one binary64 ULP** above the original
split lower bound: `2⁻⁵⁴ ≈ 5.551115123125783×10⁻¹⁷`. It is also above
the exact decimal number `−0.4`. Thus the saved RPC bounds cover the
recentered Taylor-model input set, **not the entire first physical initial
box declared in the C++ source**. The difference is a small but real missing
lower strip along each of these three coordinates. This finding concerns
only the first split of `x1`–`x3`; the other split boxes have not been
audited here.

The earlier [independent interval gate](../native_quad_independent_interval_gate_20261002_001/README.md)
and [algebraic gate](../native_quad_algebraic_gate_20261003_001/README.md)
use saved RPC bounds as their initial domain. Their containment comparisons
remain comparisons for that **RPC-defined Taylor-model domain**. Their
"whole first initial box" wording must not be read as coverage of the
source-declared `X0`. Widening only the checker domain would not establish a
CROWN guarantee outside the saved RPC input bounds. Production promotion
remains closed; no conclusion here covers the other boxes, later control
calls, or the full horizon.
