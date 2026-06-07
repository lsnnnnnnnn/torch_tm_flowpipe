# Flow* Accepted-Step Trace Divergence Report

This is a diagnostic probe, not a Flow* parity claim.

- Horizon traced: T=0.5
- Flow* source: local toolbox probe linked against `/srv/local/shengenli/flowstar/flowstar-toolbox/libflowstar.a`
- PyTorch modes: existing `normalized_insertion` no_queue and `normalized_insertion_symqueue_v2` with `flowstar_linear_v2`
- Material threshold: width/residual/channel ratio outside `[0.8, 1.25]`, or center/scaling absolute delta above `1e-6`

## First Channel Divergence

- Step: 0
- Channel: center/scaling
- Flow* h: 0.012500000000000001
- no_queue width ratio: 0.9213038720324402
- v2 width ratio: 0.9213038720324402
- no_queue residual ratio: 4.9826544783281745
- v2 residual ratio: 4.9826544783281745

## First Width Or Residual Divergence

- Step: 0
- no_queue width ratio: 0.9213038720324402
- v2 width ratio: 0.9213038720324402
- no_queue residual ratio: 4.9826544783281745
- v2 residual ratio: 4.9826544783281745

## Output Files

- `outputs/flowstar_step_trace_compare/flowstar_trace.csv`
- `outputs/flowstar_step_trace_compare/torch_noqueue_trace.csv`
- `outputs/flowstar_step_trace_compare/torch_v2_trace.csv`
- `outputs/flowstar_step_trace_compare/aligned_trace_diff.csv`

## Limitations

- The Flow* C++ probe mirrors the local adaptive symbolic-remainder path for this benchmark and logs public internals; it does not patch or commit Flow* source.
- PyTorch cutoff/Picard fields use existing diagnostics. Fields absent in a mode are left blank or zeroed when the channel is not present.
