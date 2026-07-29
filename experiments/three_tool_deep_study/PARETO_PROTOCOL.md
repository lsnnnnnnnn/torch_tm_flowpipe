# Native practical Pareto and runtime protocol

Pareto rows use three objectives: width at one explicitly recorded absolute
time, successful horizon, and steady full-configuration CPU runtime.  Because
native practical configurations do not share exact bases or arithmetic
backends, dominance is computed only **within one tool**, system, and absolute
evaluation time.  The cross-tool plot is a labelled tradeoff display, not a
relative ranking.  Adaptive Flow* rows at `T=10` are therefore never ranked
against fixed rows at `T=1`.

The complete native sweep contributes every documented candidate.  Selected
practical configurations additionally receive ten post-warmup, batch-size-one
full-horizon repetitions:

- Torch: order 2 and order 4 with a normalized affine box reset;
- DiffReach: restricted quasi-quadratic mode, symbolic window 100, five
  refinement rounds;
- Flow*: corrected fixed order 4 with native `Flowpipe` carry.

The median is primary; minimum and maximum are retained.  Torch reports Python
orchestration plus polynomial arithmetic and validation.  DiffReach separates
JIT/first execution from ten after-JIT executions.  Flow* separates build time,
per-process startup/full execution, and emitted per-step timing.  These
categories are not collapsed into a claim of backend fairness.

CPU float64 (or Flow*'s MPFR interval mode) is primary.  A secondary
implementation/hardware study uses the first coupled-quadratic configuration,
whose cross terms are known to activate, and repeats the same selected native
full configuration on:

- Torch CPU and CUDA when `torch.cuda` exposes a device;
- DiffReach JAX CPU and JAX CUDA when the installed JAXlib exposes a GPU
  device;
- Flow* CPU.

The secondary rows are explicitly not an algorithmically hardware-fair
cross-tool comparison.  An unavailable backend produces a capability row, not
a fabricated timing.  Process peak RSS or device peak allocation is recorded
where the runtime exposes a useful measurement, and otherwise marked
unavailable.
