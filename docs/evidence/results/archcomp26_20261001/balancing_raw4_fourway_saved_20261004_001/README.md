# Balancing raw4 four-method saved prefix

Date: 2026-10-04. This is a Python redraw and direct numerical reduction of
existing records, not a new experiment. It uses the separately named
`balancing-fixed-repo-raw4` contract, full initial box
`[-0.1,0.1] × [-0.05,0.05] × [-0.1,0.1] × [-0.05,0.05]`, raw four-state network,
0.02 s held controls, 0.005 s ODE substeps and requested T=10. The paper's
five-feature controller remains unresolved; see the
[contract gate](../../../../ARCHCOMP26_BALANCING_EXECUTION_GATE_20261002.md).

The [four-state same-axis figure](output/balancing_raw4_fourway_common_prefix.png)
and [vector PDF](output/balancing_raw4_fourway_common_prefix.pdf) draw only
the common 83-step numerical prefix, t in [0,0.415]. The full-horizon timeline
shows the distinct saved prefixes: P3 86 steps, Huan/Xiangru 98 each, Native 83.
P3/Huan/Xiangru reject the next numerical step; native reports
`UNCOMPLETED_SAFE` at step 84. These are incomplete processes, not full T=10
results. The intended target x1,x3,x4 in [-0.001,0.001] applies only to
8<t<=10; it appears on the future-window timeline and is never shown as a
Safe band on the early prefix. No method reaches that window.

[Geometry JSON](output/geometry.json) retains every available step's four-state
absolute tube and endpoint unions, union widths, and per-box mean/max widths.
The native binary stores an additional clock state; it is explicitly excluded
from the four physical-state statistics. [Saved bounds CSV](output/saved_bounds.csv)
has 32,000 planned method/step/state rows (4 × 2,000 × 4); rejected and
unobserved rows have empty bounds. [Absolute widths](output/absolute_widths.csv)
has 48 rows for the common endpoint, each method's last saved endpoint, and
the requested T=10 endpoint, whose bounds remain empty. Prefix tube unions
are separate from endpoint unions.

All finite ordered source intervals are retained without repair. Huan and
Xiangru each have 189 endpoint components marginally outside the same saved
tube, maximum 2.4868995751603507e-14; these discrepancies are disclosed in
[AUDIT](output/AUDIT.json), not clamped away. The native and P3 records have
zero such component discrepancies. Neither overlapping curves nor direct
saved-value agreement is an independent NNCS certificate.

[The shared Python script](plot_saved_prefixes.py) reads only existing JSONL
and binary data, also producing the sibling DP-more package. All `hashlib`
constructors, `new`, and `file_digest` fail if called during generation; no
content digest or MATLAB file is produced. Source bytes are compared directly
before and after; [the receipt](output/SOURCE_BYTE_COMPARISON.json) records the
eleven unchanged files read across both packages. PNG/PDF were visually and
structurally checked. Each script run requires new output directories; it does
not overwrite old evidence.

Run from the repository root with the existing Python plotting environment:

```console
MPLCONFIGDIR=/private/tmp/flowpipe-python-only-20261004-mpl /opt/anaconda3/bin/python \
  docs/evidence/results/archcomp26_20261001/balancing_raw4_fourway_saved_20261004_001/plot_saved_prefixes.py \
  --output-root /path/to/new/prefix-redraw
```

Original run summaries: [P3](../balancing_fixed_raw4_p3_full500_001/SUMMARY.md),
[Huan](../balancing_fixed_raw4_huan/AUDIT.md),
[Xiangru](../balancing_fixed_raw4_xiangru_20261002/SUMMARY.md),
[Native](../native_balancing_raw4_20261002/SUMMARY.md).
