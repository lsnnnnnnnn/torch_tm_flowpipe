# Flow* Accepted-Step Trace Plan

This diagnostic uses a repo-local C++ probe in `experiments/flowstar_probe/` and does not commit Flow* source changes.

The probe links against `/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a` and mirrors the local Van der Pol adaptive symbolic-remainder step path with original Flow* settings: adaptive step `0.002..0.1`, order `4`, cutoff `1e-10`, target remainder `[-1e-4, 1e-4]`, initial box `x=[1.1,1.4]`, `y=[2.35,2.45]`.

Trace rows are emitted per adaptive attempt, including rejected shrink attempts and accepted steps. The aligned comparator keeps Flow* as the diagnostic reference and converts existing PyTorch normalized insertion diagnostics for `normalized_insertion` no_queue and `normalized_insertion_symqueue_v2`/`flowstar_linear_v2` into the same columns.

Required channels:

- accepted-step timing: `t_before`, `h`, accepted/rejected status
- pre/right map range widths and normal range widths
- endpoint range before center extraction, center, scale, inverse scale
- normalized `new_x0` box/range and target remainder
- Picard no-remainder and ctrunc residual widths
- cutoff/polynomial-difference contribution when available
- symbolic queue sizes, scalar values, and symbolic width contributions
- final flowpipe segment width

This is not a Flow* parity proof. It is a short-horizon probe for localizing the first material divergence channel.
