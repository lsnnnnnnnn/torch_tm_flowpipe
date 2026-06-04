# Flow* Benchmark Parity Outputs

This directory contains the Van der Pol Flow* original benchmark parity audit.

## Scope

- Original Flow*: `/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol`
- Generated Flow*: C++ harness generated from `original_flowstar_params.json`
- PyTorch TM range-only: weak baseline over the original Flow* segment time grid
- PyTorch TM dependency-preserving: fairer TM comparison that propagates `seg.final_tm` between original Flow* segments
- Horizon: `10.0`
- Flow* patch used: no

## Runtime Semantics

Original Flow* `wall_run_s` is subprocess wall time for running the original executable. Generated Flow* records compile wall time, executable wall time, and internal `FLOWSTAR_RUNTIME_S`. PyTorch `runtime_s` measures only TM propagation and not plot writing.

## Bound Semantics

Flow* GNUPLOT boxes are flowpipe segment boxes, not final-time endpoint boxes. `endpoint_box_available=false` for Flow* rows unless a true endpoint source is extracted from the Flow* API. This audit reports last-segment and tube widths only for Flow* parity. Failed PyTorch rows distinguish the attempted failed segment from the last validated segment with `failed_segment_*`, `last_attempted_t`, `validated_segments`, and `last_validated_t`.

## Files

- `original_flowstar_params.json`
- `original_flowstar_params.md`
- `original_flowstar/`
- `generated_flowstar/`
- `torch_range_only/`
- `torch_dependency_preserving/`
- `parity_summary.csv`
- `generated_flowstar_vs_original_comparison.csv`
- `parity_report.md`
- `overlay_*png`

## Scope Guard

No CROWN, no auto_LiRPA, no Jacobian bounds, no sin/cos support, no hybrid automata, no Flow* Python binding, no NN controller workflow, and no new algorithm were added.
