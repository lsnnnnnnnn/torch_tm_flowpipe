# Trajectory Audit Cross-check

This file compares the generated trajectory audit CSVs against the older authoritative CSVs without rerunning the older experiments.
Runtime columns are intentionally excluded from exact matching.

## Sources

- New PyTorch TM: `outputs/trajectory_audit/torch_structured_summary.csv`
- New Flow*: `outputs/trajectory_audit/flowstar_structured_summary.csv`
- Old PyTorch diagnostics: `outputs/van_der_pol_diagnostics_by_order_v2.csv`
- Old TM order audit: `outputs/tm_order_audit_vdp_order2_8.csv`
- Old Flow* sweep: `outputs/flowstar_vdp_remainder_cutoff_sweep.csv`

## Result

- Overall: 85/85 comparisons passed; 0 failed.
- PyTorch diagnostics: 70 comparisons for h=0.01, steps=10, orders 2..8, both modes.
- Flow* sweep: 15 comparisons for the three representative fixed-step/fixed-order cases.

All required comparisons passed.
