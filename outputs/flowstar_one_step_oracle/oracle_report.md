# Flowstar One-Step Oracle Report

Source run: `flowstar_style_o6_candidate8_output6_cutoff`.
Reset segment index: `94` at t=`2.4007376673997931`.
h_try: `0.003619759249546223`.
Did Flow* actually compile and run? yes.
Does Flow* validate the same local box and h_try where PyTorch rejects? no.
Flow* does not validate the same local box and h_try; the local reset box is already too wide or the step is too hard.
Flow* local one-step width sum: ``.
PyTorch failed candidate final width sum: `26.976655524727043`.
Flow*/PyTorch width ratio: ``.

## Order Comparison

| order | status | validated | runtime_s | last_width_sum | segments | failure_reason |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 4 | not_completed | false | 0.00092599999999999996 |  | 0 | Flow* reach did not complete; no segment boxes emitted |
| 6 | not_completed | false | 0.0039500000000000004 |  | 0 | Flow* reach did not complete; no segment boxes emitted |
| 8 | not_completed | false | 0.014678999999999999 |  | 0 | Flow* reach did not complete; no segment boxes emitted |

## PyTorch Attempt

PyTorch validation status: `failed`; rejection reason: `Picard residual not subset of target remainder`.
This is a local diagnostic only, not full Flow* parity.
