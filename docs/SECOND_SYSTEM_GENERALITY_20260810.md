# Second-system generality (2026-08-10)

The pinned DiffReach revision contains no NAV/DR15 or navigation configuration
or source, so no private NN/controller asset was assumed. The prescribed
fallback was used instead.

The generic fixed-support object pipeline ran both a harmonic oscillator and
scalar quadratic/Riccati system for 100 steps at h=.01 on CPU and V100, with
B1 and real B64 partitions. All eight rows completed T1 and their aggregate
endpoint hulls contained the analytic endpoint hull. Harmonic uses DR7; the
one-state Riccati system uses DR5. Endpoint and full-step tube are available;
no navigation safety property is available.

The result label is `GENERALITY_GATE_PASSED`, scoped strictly to the
fixed-support plant-only fallback. It is not a native NAV result, not a
controller result, and not evidence that VDP-specific compiled preparation is
generic. S1 itself did not pass its local prefix gate, so its primitive
harmonic/quadratic tests are not presented as an integrated second-system S1
run.
