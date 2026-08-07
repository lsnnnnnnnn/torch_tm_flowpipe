# TORA-Q3 closed-loop closure report

This closure is Case C: the sound full-loop candidate improves the formal horizon, while the required 10x GPU performance gates remain unmet.

## Formal outcomes

- baseline complete-Q3 K2: certified through T=4.3; property first fails at segment 44
- candidate complete-Q3 K3: certified through T=4.4; property first fails at segment 45
- baseline segment 44 still passes finiteness and every numerical subset certificate
- T5/T10/T20 widths are N/A after the hierarchical candidate gate fails at segment 45

## Root cause

At T=1, 99.924% of the measured Torch/Xiangru difference is already present in the direct endpoint before projection. At segment 40, width is remainder-dominated; project/materialize inflation is about 1e-12.

## Performance

- compiled B48 one-step: `4.621x`
- compiled common-control T20: `4.854x`
- P0/P1/P2 pass; P3/P4 fail the required 10x thresholds

The next concrete technical problem is a sound reduction of the remainder-dominated period-5 controller input, not another affine projection reorder.
