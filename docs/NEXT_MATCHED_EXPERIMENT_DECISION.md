# Next matched experiment decision

Decision: the TORA B48 matched-input study described below is designed but **not
authorized** while the Flow* scalar-affine and Xiangru complete-Q3 interval
soundness gates remain open.  The next separate task after this scalar-affine
diagnosis is the Xiangru complete-Q3 interval-soundness audit; only after both
soundness gates close may the TORA study be authorized.

The Flow* diagnosis now selects Outcome F: the unmodified stock remainder
refinement first loses scalar-affine containment at
`Continuous.cpp:1013-1029`, and the gate remains open. This completed diagnosis
does not authorize TORA. It does leave the previously declared next separate task
unchanged: audit the complete-Q3 dynamics interval add/mul/sin/cos path for an
outward-rounding guarantee, without changing its arithmetic in that audit.

The deferred TORA B48 matched-input design is:
the stock/author-native Flow* plant lane and Xiangru's complete-Q3 PyTorch dynamics
lane would receive the same 48 initial leaves, held controller affine bounds,
controller periods, plant constants, 0.1-second segment boundaries, T=20 target and
full-tube `[-2,2]^4` property.

It is valuable because the fresh B48 complete-Q3 run reaches T=20 while coarser
B12/B24 runs fail at 15.1/18.1 seconds, and Xiangru's native CROWN-Reach/Flow* lane
also reaches T=20.  A matched B48 workload would isolate representation and
remainder propagation more meaningfully than the currently incomparable B12
Flow* and B48 Q3 timings.

## What must match

- byte-identical ONNX controller and per-period input boxes;
- the same 48 `(8,6,1,1)` initial leaves and leaf ordering;
- identical TORA ODE constants, control hold, 0.1 segment boundaries and T=20;
- identical full-tube property and fail-closed completion/certificate rule;
- separately reported cold process, controller, dynamics/core, validation and
  compile/warm boundaries with explicit CUDA synchronization.

Native differences must remain explicit rather than be adapted away: Flow*'s
Taylor-model basis, interval arithmetic and reset; complete-Q3's 84-slot support,
DR-RP seed/refinement and symbolic carry; C++ CPU versus PyTorch CUDA execution;
and each tool's native certificate fields.

## Gates and success condition

Both lanes must reach T=20, expose every requested full-tube property field, retain
unmodified source identities, and pass their own native validation.  Before any
cross-tool claim, the Flow* scalar-affine under-enclosure must be independently
resolved and the complete-Q3 ordinary float64 interval path must gain a justified
soundness classification.  Controller affine bands must also be shown identical
at every period.

This experiment is designed but not run here.  The current Flow* lane is B12, not
B48; changing its partition workload would be a new experiment, while the two
soundness gates above are still open.  Running an easier substitute would not
answer the matched question.
