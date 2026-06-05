# Flowstar One-Step Oracle Report

Source run: `flowstar_style_o6_candidate8_output6_cutoff`.
Reset segment index: `94` at t=`2.4007376673997931`.
h_try: `0.003619759249546223`.
Does Flow* validate the same local box and h_try where PyTorch rejects? no.
Flow* does not validate the same local box and h_try; the local reset box is already too wide or the step is too hard.
Flow* local one-step width sum: ``.
PyTorch failed candidate final width sum: `26.976655524727043`.
Flow*/PyTorch width ratio: ``.

## Order Comparison

| order | status | validated | runtime_s | last_width_sum | segments | failure_reason |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 4 | skipped | false | 0 |  | 0 | Flow* toolbox root not found; set FLOWSTAR_ROOT or pass --flowstar-root |
| 6 | skipped | false | 0 |  | 0 | Flow* toolbox root not found; set FLOWSTAR_ROOT or pass --flowstar-root |
| 8 | skipped | false | 0 |  | 0 | Flow* toolbox root not found; set FLOWSTAR_ROOT or pass --flowstar-root |

## PyTorch Attempt

PyTorch validation status: `failed`; rejection reason: `Picard residual not subset of target remainder`.
This is a local diagnostic only, not full Flow* parity.
