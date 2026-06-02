# Van der Pol Order and Flow* Results

Authoritative files for the corrected report bundle:

- `order_and_vdp_flowstar_report.md`: final narrative report copied from `docs/order_and_vdp_flowstar_report.md`.
- `tm_order_audit_vdp_order2_8.csv`: torch order/degree audit; use this for requested-vs-actual degree semantics.
- `van_der_pol_diagnostics_by_order_v2.csv`: torch Van der Pol diagnostic decomposition by order.
- `flowstar_vdp_remainder_cutoff_sweep.csv`: authoritative Flow* setting sweep. Flow* endpoint boxes are unavailable here; use last-segment and tube columns.
- `flowstar_vdp_plot_input_v2.csv`: combined plot input used for corrected figures.
- `order_flowstar_status_table.md`: Flow* status table by order, h, steps, and setting.
- `flowstar_vdp_remainder_cutoff_sweep_summary_v2.md`: corrected summary with semantic torch-vs-Flow* ratios.

Corrected plots:

- `torch_over_flowstar_last_segment_width_ratio_by_order.png`
- `torch_over_flowstar_tube_width_ratio_by_order.png`
- `van_der_pol_endpoint_width_vs_order.png`
- `van_der_pol_last_segment_width_vs_order.png`
- `van_der_pol_tube_width_vs_order.png`
- `van_der_pol_runtime_internal_vs_order.png`
- `van_der_pol_runtime_wall_vs_order.png`
- `van_der_pol_remainder_frac_vs_order.png`
- `van_der_pol_poly_vs_remainder_stacked_by_order.png`
- `flowstar_status_by_order_and_setting.png`

Deprecated ambiguous outputs were removed from this bundle. Do not use any old
table or plot that compares torch endpoint/final width to Flow* GNUPLOT
last-segment width.

Width semantics: endpoint, last-segment, and tube widths are distinct. Torch-vs-Flow* ratios in this bundle use matching last-segment or tube widths unless both tools explicitly provide endpoint boxes. Runtime semantics are also distinct: Flow* internal reach time comes from `FLOWSTAR_RUNTIME_S`, Flow* wall run time includes executable and plotting overhead, compile time is separate, and torch runtime is Python algorithm wall time.
