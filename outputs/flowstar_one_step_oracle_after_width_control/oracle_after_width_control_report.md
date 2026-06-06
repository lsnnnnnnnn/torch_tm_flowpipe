# Flowstar One-Step Oracle Report

Source run: `flowstar_style_o6_candidate8_output6_cutoff_symqueue`.
Reset segment index: `2` at t=`0.094531250000000011`.
h_try: `0.00263671875`.
Did Flow* actually compile and run? yes.
Does Flow* validate the same local box and h_try where PyTorch rejects? yes.
Flow* validates the same local box and h_try where PyTorch rejects; the PyTorch kernel is missing or tighter than a Flow* mechanism.
Flow* local one-step width sum: `0.6071100000000003`.
PyTorch failed candidate final width sum: `0.5919358932421922`.
Flow*/PyTorch width ratio: `1.0256347130340338`.

## Order Comparison

| order | status | validated | runtime_s | last_width_sum | segments | failure_reason |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 4 | completed | true | 0.00099500000000000001 | 0.60711000000000026 | 1 |  |
| 6 | completed | true | 0.0040810000000000004 | 0.60711000000000026 | 1 |  |
| 8 | completed | true | 0.014541 | 0.60711000000000026 | 1 |  |

## PyTorch Attempt

PyTorch validation status: `failed`; rejection reason: `Picard residual not subset of target remainder`.
This is a local diagnostic only, not full Flow* parity.
