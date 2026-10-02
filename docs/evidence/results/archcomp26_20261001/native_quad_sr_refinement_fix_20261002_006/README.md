# Isolated copied-library SR refinement mechanism probe

This is a **diagnostic**, not a revised benchmark result. It copied the
original Flow* `Continuous.cpp` and `libflowstar.a` to a new server directory,
then changed only `bool bfinished = false` to `true` in the Real
symbolic-remainder `Flowpipe::advance` loop at frozen `Continuous.cpp:2382`.
The [exact source diff](Continuous.patch) shows the one-line change. The
original source, library and full QUAD job were untouched.

The same first-box, one-control-call, one-ODE-step 2026 paper QUAD diagnostic
ran observer off/on against the copied archive. Both accepted one step,
produced equal status/RPC/terminal axes/range files, and the identical
[finite-sample checker](AUDIT.json) found **0 of 20,520 sampled checks outside**
the saved bounds. For comparison, the unpatched copied archive's
[original gate](../native_quad_sr_octagon_gate_20261002_005/README.md) had
2,048 terminal x7/x8 sample failures.

The copied-library change keeps the first validated Picard remainder instead
of refining it. Bounds become much wider: first endpoint x7 changes from
about `[1.638e-6,1.655e-6]` to `[-5.576e-4,5.609e-4]`. This confirms a
local refinement mechanism consistent with the under-enclosure, but the
one-line skip is not a production-quality fix, a speed claim, or a soundness
proof. The original-library failing receipt remains the production gate.
