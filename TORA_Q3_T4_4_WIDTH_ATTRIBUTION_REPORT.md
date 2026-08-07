# TORA-Q3 T4.4 width attribution report

The formal L0 verdict first fails at segment 44 because the safety property is not proved. All 48 leaves still pass finiteness, the initial subset check, and all ten remainder rounds.

Diagnostic propagation remains numerically certified through segment 47 and first loses the numerical certificate at segment 48.

## T=1 refresh attribution

- direct endpoint versus Xiangru: `0.014211021942602` max abs
- project/materialize change: `1.07787637813328e-05` max abs
- dominant source: the preceding ten plant segments, not projection

## Sound shadow/candidate results

| lane | formal horizon | first failure |
|---|---:|---|
| L0_baseline_native | 4.3 | property at segment 44 |
| L1_tight_endpoint_box_controller | 2.1 | property at segment 22 |
| L2_physical_endpoint_projection | 4.3 | property at segment 44 |
| L3_horner_registered_best | 4.3 | property at segment 44 |
| L4_k3_picard | 4.4 | property at segment 45 |

K3 is the selected sound method candidate: it moves the formal horizon from 4.3 to 4.4, then fails the property at segment 45.
