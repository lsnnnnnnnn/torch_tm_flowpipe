# Local operator causal closure — 2026-08-13

Status: `LOCAL_OPERATOR_SOURCE_DELTA_OPEN`.

The intended eight P/R/X swap cells were not executed. This is not a schema
workaround: Gate D produced an `UNDER_ENCLOSURE_WITNESS` before P, R, or X.
Feeding that normalized prestate into any swap would propagate a set that does
not contain the declared exact input, which the goal's fail-closed rule
forbids.

The matrix records every cell as `executed=false` with reason
`GATE_D_UNDER_ENCLOSURE_STOP`:

```text
P_F + R_F + X_F
P_T + R_T + X_T
P_F + R_T + X_T
P_T + R_F + X_T
P_T + R_T + X_F
P_oracle + R_T + X_T
P_oracle + R_oracle + X_T
P_oracle + R_oracle + X_oracle
```

The smallest blocker is not “near Picard or range.” It is the binary encoding
boundary that constructs a point-coefficient normalized TM from decimal
binary64 endpoints (`FlowstarNormalFlowpipeState.from_initial_box` and
`Interval.mid/radius` on Torch; the initial `Flowpipe(box)` normalization on
Flow*). A future run must first introduce and independently test an outward
encoding of the exact rational affine set on both sides. That change was not
authorized as an L1 operator candidate in this run.
