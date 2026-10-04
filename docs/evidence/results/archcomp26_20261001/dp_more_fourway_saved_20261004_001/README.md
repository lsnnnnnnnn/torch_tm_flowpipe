# Double Pendulum more robust four-method saved prefix

Date: 2026-10-04. These plots and statistics reuse the fixed more-robust
controller, full [1,1.3]^4 initial box in 225 cells, 0.02 s held-control periods,
0.005 s numerical steps and requested T=0.4. The physical state order is
theta1, theta2, theta1_dot, theta2_dot. The Safe box [-1.5,1.5]^4 applies to all
continuous time in [0,0.4], as recorded by the
[informational plot contract](../../../../../benchmarks/plot_specs/double_pendulum_more_robust_2026_nohash.json).

The [four-state same-axis PNG](dp_more_fourway_common_prefix.png) and
[vector PDF](dp_more_fourway_common_prefix.pdf) show the **common numerical
prefix of 64 steps through T=0.32**, while the **common saved Safe prefix is
60 steps through T=0.30**. The dotted boundary marks T=0.30. All four saved
methods first cross the Safe band at step 61; those intervals remain visible.
The gray remainder has no common four-method data and is not interpolated.
The lower timeline separately records P3/Huan/Xiangru 72 saved steps each and
Native 64. Native ends with its `UNKNOWN` property result, and the other three
stop after the author checker reports `Unsafe.`. Those labels and interval
crossings are not independently verified trajectory counterexamples.

The P3 data are the current affine-split4 full-horizon request, read directly
from 72 JSONL rows containing all 225 accepted per-lane tube/endpoint intervals.
The old interval-controller P3 step-9 failure is a separate source and is not
silently substituted. Huan/Xiangru and Native use their original saved binaries.
The new reduction at step 60 exactly matches all existing Huan/Xiangru/Native
endpoint unions in their common-60 scans.

[Geometry JSON](geometry.json) retains all available steps, every physical
state's tube/endpoint unions, union widths, and per-box mean/max widths.
[Saved bounds CSV](saved_bounds.csv) has all 1,280 planned method/step/state
rows (4 × 80 × 4); missing rows keep their bounds empty.
[Absolute widths CSV](absolute_widths.csv) has 64 rows: common numerical
endpoint T=0.32, common saved Safe endpoint T=0.30, each method's last saved
endpoint, and requested T=0.4 endpoint (empty for all methods). Each available
scope keeps the full-prefix tube union separate from the endpoint union.

No interval was repaired. Huan and Xiangru each contain 14,149 endpoint
components marginally outside the corresponding tube, maximum
4.6629367034256575e-15; P3/Native have zero such components. These are counted
from the original per-lane records and retained in [AUDIT](AUDIT.json).
The two author curves overlap; overlap is not independent correctness evidence.

Generation uses the sibling [saved-data Python script](../balancing_raw4_fourway_saved_20261004_001/plot_saved_prefixes.py)
under guards blocking every `hashlib` constructor, `new`, and `file_digest`.
No solver, model inference, MATLAB export, content digest, or old experiment
was executed. [Direct byte comparisons](SOURCE_BYTE_COMPARISON.json) preserve
all eleven source files read across the two new packages. PNG/PDF were
visually and structurally checked. The native octagon production gate and
independent NNCS proof remain unresolved.

Original outcomes: [P3](../dp_more_p3_affine_split4_full20_20261003_001/README.md),
[Huan/Xiangru](../author_dp_more_v1/SUMMARY.md),
[Native](../native_dp_more_full20_001/SUMMARY.md).
