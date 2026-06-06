# Flowstar One-Step Oracle Report

Source run: `flowstar_style_o6_candidate8_output6_insert_symqueue`.
Reset segment index: `37` at t=`3.3500000000000014`.
h_try: `0.0023437500000000003`.
Did Flow* actually compile and run? yes.
Does Flow* validate the same local box and h_try where PyTorch rejects? yes.
Flow* validates the same local box and h_try where PyTorch rejects; the PyTorch kernel is missing or tighter than a Flow* mechanism.
Flow* local one-step width sum: `0.4839530000000003`.
PyTorch failed candidate final width sum: `0.45836311114672357`.
Flow*/PyTorch width ratio: `1.0558288575824883`.

## Order Comparison

| order | status | validated | runtime_s | last_width_sum | segments | failure_reason |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 4 | completed | true | 0.0010189999999999999 | 0.4839530000000003 | 1 |  |
| 6 | completed | true | 0.0040150000000000003 | 0.4839530000000003 | 1 |  |
| 8 | completed | true | 0.014841999999999999 | 0.4839530000000003 | 1 |  |

## PyTorch Attempt

PyTorch validation status: `failed`; rejection reason: `initial or cutoff remainder exceeds target remainder`.
This is a local diagnostic only, not full Flow* parity.
