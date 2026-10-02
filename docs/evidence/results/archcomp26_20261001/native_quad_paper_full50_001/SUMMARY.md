# Native Flow* QUAD, 2026 paper equations, full initial set

This is the **existing** `native_quad_paper_full50_001` job. It was monitored and read after its natural completion; no old QUAD job was restarted. The original 417,792,000-byte `ranges.bin` remains on the server at `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_quad_paper_full50_001/ranges.bin`. It was streamed there read-only; only small receipts and pooled ranges were copied here. No content digest or checksum was computed.

## Run identity and completion

- [START.json](START.json) names the 2026 paper-equation, 1024-box, 50-period contract, the Torch ONNX controller, GPU 1 and CPU 6–9. The exact isolated [C++ source](build/archcomp/Quadrotor/quad_paper_full50.cpp), [BUILD.json](build/BUILD.json) and [launcher](run_archcomp26_quad_paper_native_pair.sh) are preserved. The physical initial set is `[-0.4,0.4]^6 × {0}^6`, partitioned `8×8×8×2×1×1`; 12 physical states and four auxiliary variables are integrated with Flow* order 2 and fixed `h=0.005`.
- [RESULT.json](RESULT.json): started `2026-10-01T11:25:56.399440+00:00`, ended `2026-10-02T00:30:15.301466+00:00`, exit code 0, status `completed`, no timeout, outer wall `47058.886571025476 s` (one sample). [native.log](native.log) has Steps 0–49, 50 `Flow* started/finished` pairs, final `VERIFIED`, and internal `time cost: 46918.890000 s`. [controller_rpc.jsonl](controller_rpc.jsonl) has 50 calls, each with 1024 input and output boxes. Original [stdout](stdout.log), [stderr](stderr.log), [server log](server.log) and [startup listeners](listeners_at_start.txt) are also preserved; stderr is empty.

## Original range scan

- [scan_ranges.py](scan_ranges.py) streamed the original server file as 408-byte little-endian records: `uint64 box`, `uint64 step`, `float64 h`, then four `float64` bounds `(tube_lo,tube_hi,endpoint_lo,endpoint_hi)` for each of 12 physical states. File order is control period (50) → box (1024) → 20 substeps. [SCAN.json](SCAN.json) records **1,024,000/1,024,000** records, exactly one of every `(box,step)` for boxes 0–1023 and steps 1–1000. All records have `h=0.005`; counts of nonfinite bounds, reversed intervals, endpoint outside the same-step tube, sequence mismatch, and invalid box or step are all zero. This verifies saved record coverage and basic range consistency, not solver acceptance; `ranges.bin` has no acceptance field.
- At `T=5`, the pooled `x3` **endpoint** is `[0.965771839016746, 1.0167484756616678]`, width `0.0509766366449218`, inside the terminal target `[0.94,1.06]`. The pooled last-step **tube** is `[0.965771088884312, 1.0167484756616678]`, width `0.05097738677735586`. Across all 1000 saved tubes, `x3` spans `[-0.40817151558335585, 1.452252535346294]`; this all-time hull must not be used as the terminal result.
- As an additional range-only diagnostic, every pooled `x3` tube from step 775 through 1000 lies inside `[0.94,1.06]`, nominally covering `t∈[3.87,5]`; step 774 is the last tube outside. The paper's reach-and-remain checker semantics have not been independently fixed by this scan.
- [terminal_12states.csv](terminal_12states.csv) gives each physical state's terminal tube and endpoint union bounds and widths, plus mean/max per-box widths. [pooled_x3_1000steps.csv](pooled_x3_1000steps.csv) gives each step's `x3` tube and endpoint union for plotting. `SCAN.json` also contains all 12 states' pooled bounds for every saved step.

The source's `VERIFIED` branch calls `fp_end_of_time.isInTarget(targetSet, setting)` for each initial box; it checks the terminal `x3` target. This is the native implementation's endpoint result under this selected contract, not an independent end-to-end floating-point neural-network certificate or a resolved temporal interpretation of the 2026 paper's reach-and-remain wording. A single outer-wall sample does not support a stable speed ranking.

**Subsequent validity gate (2026-10-02):** An [isolated first-box, one-control,
one-step test of the same frozen Flow* symbolic-remainder path](../native_quad_sr_octagon_gate_20261002_005/README.md)
returned numerical `COMPLETED_SAFE` but missed finite DOP853/Radau samples
drawn from the C++ program's CROWN affine-plus-residual relaxed control set
in the `x7/x8` terminal bounds (maximum excess `1.98259e-6`). The control
chosen for that witness was not established as the actual NN output. This
short result does not prove every saved bound in this full run incorrect, but
it blocks treating the above `VERIFIED` and accepted numerical coverage as
independent evidence of sound relaxed-set enclosure or a full NNCS proof.
The frozen full-run receipt, count and wall time remain unchanged.
