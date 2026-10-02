# Balancing fixed-repository raw-four-state / P3 first-period diagnostic

This is one new, isolated run of the explicitly named `balancing-fixed-repo-raw4` profile. It is **not** the paper's unresolved five-feature controller. The original server run is `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/balancing_fixed_raw4_p3_smoke1_001`; the preceding CPU-only run is in the sibling `balancing_fixed_raw4_p3_preflight_001`. Existing Huan runs were not restarted.

## Frozen scope and preflight

- Full physical initial box: `[-0.1,0.1] × [-0.05,0.05] × [-0.1,0.1] × [-0.05,0.05]`, one unsplit box. The auxiliary clock and held force start at zero.
- Fixed repository `model.onnx` takes raw `(x1,x2,x3,x4)` and produces one force `f`. The CPU-only [preflight](../balancing_fixed_raw4_p3_preflight_001/payload/PREFLIGHT.json) read its `4→1` interface and `Gemm,Tanh,Gemm,Tanh,Gemm,Tanh` graph, checked the saved CartPole ODE and fixed official text, and did not initialize CUDA. The model's correspondence to the fixed official file rests on the earlier direct byte comparison recorded in the [execution gate](../../../../ARCHCOMP26_BALANCING_EXECUTION_GATE_20261002.md); no new content check was run.
- Physical ODE: `x1'=x2`, `x2'=2f`, `x3'=x4`, `x4'=(0.08·0.41·(9.8 sin(x3)−2f cos(x3))−0.0021 x4)/0.0105`. The force is held for one `0.02 s` control period. P3 used four `0.005 s` ODE steps, working order 3, validation policy `solution_plus_one` (order 4), strict endpoint/control injection, box same-slope CROWN and float32 controller transport. [Generated config](payload/config.yaml) and [method record](payload/P3_METHOD.json) fix these settings.
- Physical GPU 3, CPUs 14–17, GPU memory cap 11 GiB, outer timeout 300 s. Before launch, GPU 3 reported 0 MiB allocated by listed processes and the run ID did not exist. The P3 refinement callback recorded internal steps and selected the eager refinement path; these times are diagnostic, not a production P3 timing sample.

## Observed result

The [outer RESULT](RESULT.json) reports exit 0, no timeout and process wall **5.230873 s**. The [inner RESULT](payload/RESULT.json) reports `completed_short_prefix`, **4/4 accepted ODE steps**, `broken=0` in [metrics](payload/metrics.json), no first numerical rejection, no property check, and no certificate. Driver time was 1.147668 s; peak CUDA allocation was 77,574,656 bytes and peak reservation 176,160,768 bytes. The raw [stdout](stdout.log) and [stderr](stderr.log) are preserved.

An [independent standard-library scan](INDEPENDENT_SCAN.json) of all four saved [range records](payload/ranges.jsonl) found exactly steps 1–4, finite ordered four-state tubes and endpoints, with each endpoint included in its corresponding tube. The full observed `t∈[0,0.02]` tube union is:

| State | Saved tube union | `t=0.02` endpoint |
| --- | --- | --- |
| `x1` | `[-0.10174384575441542, 0.10175115609645909]` | `[-0.1017417988586432, 0.10175115609645909]` |
| `x2` | `[-0.12429685133695681, 0.1251156096458443]` | `[-0.1241798858642586, 0.1251156096458443]` |
| `x3` | `[-0.10235840884288187, 0.10233564056280534]` | `[-0.10235840884288187, 0.10232926127969098]` |
| `x4` | `[-0.24052005108409646, 0.237970095726142]` | `[-0.24052005108409646, 0.23760427418029093]` |

The [internal refinement trace](payload/refinement_trace.jsonl) has 136 events, 34 for each accepted step; there is **no first-rejection cause to report**. The full property window is `8<t≤10 s` (equivalently closed at 8 for continuous trajectories and a closed target). It does not intersect this first-period run, so the empty [property-check ledger](payload/property_checks.jsonl) means **not applicable**, never `VERIFIED`. This result is neither a full 500-period outcome nor an independent end-to-end floating-point NNCS proof or speed ranking.

The exact [outer START](START.json), [inner START](payload/START.json), CPU [preflight run](../balancing_fixed_raw4_p3_preflight_001/START.json), [source snapshot](SOURCE_SNAPSHOT/archcomp26_balancing_raw4_p3_nohash.py), scanner and dependent entry scripts are retained without content-digest checks.
