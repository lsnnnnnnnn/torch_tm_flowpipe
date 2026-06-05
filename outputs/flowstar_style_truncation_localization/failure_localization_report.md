# Flowstar-Style Failure Localization Report

Last validated t: `2.1095541733932355`.
Failure reason: `Picard residual not subset of target remainder`.
Which state dimension fails target containment first? `y`.
Is residual only slightly above target or orders of magnitude above? slight containment miss; final width-sum ratio=`0.49383386558341774`.
Width ratios are diagnostic only: containment can fail when a residual interval is shifted outside the symmetric target even if its width is below the target width.
Is the dominant term still polynomial_range * remainder? no.
Is failure triggered by truncation, cutoff uncertainty, interval polynomial range, or RHS aggregation? Dominant recorded component: `truncation`.
What h_try fails at the end? `0.003821380143545313`.
Would h below 0.002 likely help? `likely yes numerically, but only as a diagnostic because it goes below Flow* min step`.
Should the next fix be adaptive order, remainder-only Picard refinement, tighter range bounding, or real symbolic remainder queue? `adaptive order fallback first, then tighter polynomial range bounding`.

## Output Files

- `failure_step_attempts.csv` records the focused accepted/rejected attempts near failure.
- `failure_residual_breakdown.csv` records the Van der Pol RHS multiplication breakdown.
- `residual_components_near_failure.png` and `step_size_near_failure.png` visualize the failure neighborhood.
