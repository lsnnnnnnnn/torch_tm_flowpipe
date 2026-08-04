# Flowstar order-2 Van der Pol failure closure

## Accurate status

Order 2 is supported and does not crash. Two different scopes must remain separate:

- a fixed generated-stock step with `h=0.001`, order 2, candidate `1e-4`, cutoff `1e-10` completed successfully;
- the adaptive single-segment probe starting at `h=0.1` rejected `0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125`; the next halving is below `h_min=0.002`, so it completed zero accepted segments and is classified `validation_rejected`.

Neither result says that order 2 completed T=10.

## Mechanism

At the terminal `h=0.003125` attempt the candidate target is `[-1e-4,1e-4]` in each state. The generated-stock trace reproduced:

- x candidate remainder `[-1.107509613e-5, 1.084106445e-5]`, contained;
- y candidate remainder `[-2.779354115e-4, 1.062582242e-4]`;
- y self-map defect `1.779354115e-4`;
- nonlinear multiplication contribution `[-2.634297524e-4, 9.120953758e-5]`;
- y cutoff polynomial difference `[-2.0990e-18, 6.9389e-19]`;
- symbolic propagated width zero.

Thus the dominant mechanism is the cubic VDP RHS under retained Picard degree `order-1=1`: nonlinear multiplication/truncation remainder dominates, while cutoff is about fourteen orders smaller. The process itself completed normally; the mathematical self-map failed.

The exact current-run command and all six attempts are in `one_step_trace/flowstar_vdp_o2_rejection/order2_failure_manifest.json`. The source Flowstar check builds a no-remainder Picard polynomial, seeds `remainder_estimation`, computes `Picard_ctrunc_normal`, adds polynomial differences, and requires subset inclusion (`flowstar-toolbox/Continuous.cpp:956-1004`); refinement repeats the inclusion test (`:1014-1028`).

## Route parity limitation and first trace difference

The official program route does not expose its internal order-2 candidate fields. Therefore an exact official-program versus generated-stock candidate-defect equality is not claimed. This is one reason the official/generated field-parity gate remains false.

The supplied attempt-aligned Flowstar/Torch diagnostic finds the first matched numeric divergence at the Picard residual for `h=0.025`: Flowstar residual-width sum `0.01043966`, Torch `0.00454062`; both reject. Downstream acceptance or endpoint differences are not treated as independent root causes.

After aligning initial coordinates to Flowstar-normalized `[-1,1]`, both generated-stock and Torch accept the order-2 `h=0.001` sensitivity step, and their observed supports match after permuting local time. Their retained coefficients differ by at most `2.29e-16`; their y remainders are respectively approximately `[-7.70486e-5,2.19787e-5]` and `[-8.16088e-5,2.65024e-5]`. Full Picard field parity is still unavailable on stock.

## GCC15 compatibility edit

The sole tracked Flowstar change corrects the assignment target in `TaylorModel::derivative` (`flowstar-toolbox/TaylorModel.h:897-900`). Repository search found no VDP-specific caller outside the generic derivative methods. Official and generated order-4 plot segments match exactly under the same compiled archive. An unpatched GCC15 binary cannot be built in this environment, so a direct patched/unpatched numerical equivalence experiment is unavailable; the audit records it as a scoped compilation/assignment compatibility change rather than silently calling the checkout clean.
