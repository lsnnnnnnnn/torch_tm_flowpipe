# QUAD native-contract memory experiment: historical result

Final status: **plant_numerical_rejection_controller_unqualified**. Completed records: 597 / 1000; all-1024-lane accepted steps: 594; accepted lane-steps: 609861; broken lanes: 1024.

plant runtime requested strict mode; final plant qualification is pending the endpoint repair. Native-f64 CROWN affine/bias arithmetic lacks proved outward roundoff accounting, so controller_unqualified and no strict NNCS certificate. No matched end-to-end speed claim is made.

A later Airplane audit identified FULL/spatial support-ID mixing in the public endpoint helper. These frozen NNCS runs are historical observations pending that repair and rerun; they are not final strict qualification.

Frozen engine `15e527e9dd281e71897c8b23bc4a917dbbf9bb3b`; horner composition, graph glue, measured support, unchanged truncated validation. Native extra-case B=1024, n=16, order=2, h=.005, 50 control periods of .1 (T=5), SR queue=1000, cutoff=1e-6, remainder ±.1. Original boxes, splits, dynamics and controller remain fixed. The original inventory QUAD remains a separate invalid contract.

Resources: process 170.299254 s; peak owned GPU 10.716797 GiB; peak RSS 1.331783 GiB; supervisor completed. Limits remained allocator 9 GiB, observed RSS/GPU guard 11.5 GiB and timeout 600 s.

SR allocation: preallocated_full_capacity. reserve(1000) occurs at qlen=jlen=0; interval history is first allocated by strict propagation. SR initialization 0.020124 s is included in child/process wall time; the original driver loop timer begins after it. Plant advance 32.126084 s, controller setup 1.412320 s, controller bounds 2.411785 s, observer 30.532037 s, gzip I/O 96.334233 s. Success-step inactive cleanup count 7, time 0.074043 s, included in plant/driver/process totals.

Five real extension groups dispatched; SR Phi 595, J history 596. Binary hashes match the validated five-extension build. Final graph counters: {'enabled': True, 'hits': 4775, 'captures': 19, 'segments': 19}.

First numerical rejection is initial self-map failure, not nonfinite/domain or target rejection. At step 595, 64 lanes fail: x6 in 64 lanes and x5 in 22 lanes (86 violating components). Original guess is [-.1,.1]. Worst is lane 1009, physical state x6 (index 5), proposal [-0.10361963064283031,0.09383972255773969], maximum absolute proposal ratio 1.036196306428303. Step 594 worst ratio was .9079484830116021. A light diagnostic preserves the production tape dispatch; every accepted count through 595 and endpoint/tube/status at 594/595 matches the formal run. No numerical settings were changed.

Qualification tests: exact-Fraction SR matrix/J oracles, 12 reserve/reset/fresh-error regressions, actual chunk dispatch versus frozen point/interval histories, and four graph ownership/failure-recovery combinations passed. Real QUAD lazy versus preallocated checkpoints 1, 2, 16, 17 and 20 passed 50 direct uint8 tensor comparisons covering 49,730,560 elements. All state and active-history bytes are identical; only capacity differs (16/32 versus 1000).

Full trace verification: {"verifier_version": 2, "streamed_one_step_at_a_time": true, "records": 597, "all_records_finite_ordered_intervals": true, "accepted_lane_steps": 609861, "trace_gzip_bytes": 346749351, "trace_sha256": "3a90c25eca3291a66f43d76c1eb7514b66c3c623f56f2e38385cd81bb28d120a", "observer_checked_state_immutability": true}

First-512-record comparisons: {"largest_first15e_vs_preallocated": {"records": 512, "all_serialized_records_equal": true, "first_nonidentical_record": null, "first_numeric_field_differences": [], "left": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/nncs_quad_memory_20260922/quad_full_largest_first/quad_submit_native_trace.jsonl.gz", "right": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/nncs_quad_memory_20260922/quad_full_preallocated/quad_submit_native_trace.jsonl.gz"}, "5d_vs_preallocated": {"records": 512, "all_serialized_records_equal": true, "first_nonidentical_record": null, "first_numeric_field_differences": [], "left": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/nncs_quad5_20260922/quad_full/quad_submit_native_trace.jsonl.gz", "right": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/nncs_quad_memory_20260922/quad_full_preallocated/quad_submit_native_trace.jsonl.gz"}, "chunked97d_vs_preallocated": {"records": 512, "all_serialized_records_equal": true, "first_nonidentical_record": null, "first_numeric_field_differences": [], "left": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/nncs_quad_memory_20260922/quad_full_chunked/quad_submit_native_trace.jsonl.gz", "right": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/nncs_quad_memory_20260922/quad_full_preallocated/quad_submit_native_trace.jsonl.gz"}, "wall_s": 19.93462069518864}

Earlier attempts are retained: original 5d reached 512 accepted steps before a guard-triggering 12.597656 GiB transient; conservative 9 GiB lazy candidates reached the same prefix then refused the 3.91 GiB growth allocation. The optional global einsum observer caused a separate 0-step TorchScript startup failure and was removed. None of these prefixes is labeled a complete result. Largest-first allocation lowers the standalone live-copy peak, while upfront allocation avoids overlapping old/new large histories and fragmented full-run allocation.

Driver protocol messages: [] (subject to controller qualification above).

Large raw traces and state snapshots remain remote with size/SHA256 manifests. The small archive includes logs, metadata, source snapshots, scripts and validation evidence.
