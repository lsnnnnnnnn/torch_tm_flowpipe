# Official sigmoid: working 6, point 5, validation 7

This new wrapper stages the original official sigmoid `u=11f` source and changes only the active working-order literals from 3 to 6, the strict `solution_plus_one` validation declaration from 4 to 7, and the cutoff from `1e-6` to `1e-8`. It derives from the executed order4 wrapper; that wrapper and all saved source/reference files remain unchanged. The inherited final Picard point sweep uses working order minus one, hence point order 5.

`run_tanh_cutoff_candidate.py` is an exact direct-byte copy of the parent-directory geometry helper. The wrapper imports only `range_widths`; it does not execute that helper's tanh entry point. The geometry comparison preserves all 500 records, all four states, tube and endpoint bounds, exact-rational narrower/equal/wider and subset counts, and target-entry evidence without inheriting a property-verification label.

Run with `python -B run_sigmoid_order6_candidate.py --adapters <frozen-Oct5-adapters> --gate <passed-private-gate> --output <new-unique-run-directory>`, adding the original `--source-snapshot` and `--reference` paths if needed. The parent controls a single 120-second attempt on GPU2 / CPUs10–13; this directory has not launched a server job. The wrapper retains the original full500 observer and first-refusal stop, strict injection and endpoint handling, symbolic-remainder history, private/Horner bindings, and the fused single-row adapter's first live reference comparison. This is a new numerical method and makes no saved-output-equivalence claim.

## Order support

The original `archcomp26_dp_p3_nohash.py::strict_injection` admits the TORA 6-variable layout and requires the support degree to be at most `engine.tables.k`; it has no order4 ceiling. The staged TORA engine guard explicitly becomes `(n,k) == (6,6)`. The unchanged strict endpoint helper requires matching FULL support and likewise has no order4 ceiling.

The frozen `flowstar_gpu.sparse_exec` final structural Picard sweep uses `k_i=i-1`, and `solution_plus_one` selects `validation_order=order+1`. Its existing validation engine and bytecode factories accept that order and check the working/product-basis prefix. The reciprocal implementation accepts positive integer orders. No strict remainder, containment, or validation round is removed.

## Local check and limits

`PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B check_sigmoid_order6.py` passed using real CPU Torch2.2.2 and the frozen engine's actual tables, support, interval operations and extracted strict-injection function. It exercises a genuine degree6 polynomial term, checks exact-rational affine residual enclosure at five support points and both bias endpoints, verifies unchanged physical rows, and checks rejection of degree7 support without mutation. It also checks all declared source edits, rejection of repeated edits, and compatibility with the real order7 integer exponent enumeration.

The small CPU check builds the actual working-order6 table; it does not build the complete validation7 multiplication table or run an ODE, NN, GPU, historical checker, or content digest. The full GPU order6/validation7 execution, its memory cost, and whether its complete widths improve remain untested here. The CPU receipt is `SIGMOID_ORDER6_CPU_CHECK.json`. Independent end-to-end floating-point NNCS certification remains false.
