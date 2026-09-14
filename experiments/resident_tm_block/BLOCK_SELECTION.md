# Resident Taylor-model block selection

The implementation target is frozen as the complete call to
`insert_ctrunc_normal_dependency_preserving` made by accepted-boundary symbolic
remainder preparation.  Its input boundary is the endpoint Taylor map without
constants plus the previous accepted right map.  Its output boundary is the
composed Taylor map, including every multiplication-stage truncation, cutoff,
polynomial/remainder cross term, remainder/remainder term, and the outer
remainder.  Intermediate Horner polynomials and interval bounds are inside the
selected block.

This choice was made before implementation from fresh online `Gp` solver
windows.  The profiler wrapped existing arithmetic without changing it and
reported the union of concurrent wall intervals, so overlapping worker waits
were not added as if they were serial wall time.  Across all eight prescribed
selection windows, the measured solver wall time was 189.158071279 s and the
normal-composition interval union was 68.742341 s, or 36.3412%.  The two B32
windows measured 41.5907% for Van der Pol and 32.9197% for Brusselator.  The
late-step windows still executed the same two-variable block at orders 4 and 6.

For an end-to-end wall reduction of 20%, Amdahl's law requires the chosen block
to become at least

`0.363412 / (0.363412 - 0.20) = 2.224x`

faster, equivalently a 55.0% or greater reduction in this block.  Its ideal
end-to-end reduction ceiling is 36.34%, so the target is possible but not
automatic.

The other bounded candidate was the dense Picard/validation call.  It occupied
85.3136% of the weighted wall time, but it includes the whole per-step fixed
point, validation policy, remainder ledger, and boundary transition.  Moving
that envelope would be a second solver architecture rather than one auditable
continuous Taylor-model block.  It was therefore rejected for this round,
despite its larger ceiling.  The accepted-boundary transition envelope was
measured as context (52.8595%) but was not a third candidate: queue propagation
and commit are separate ownership/state operations, while normal composition is
the reusable numerical block within it.

The authoritative raw measurements are
`artifacts/runs/resident_tm_block_20260914T032650Z/BLOCK_SELECTION_PROFILE.json`.
No saved request or saved answer was used to advance a solver state.
