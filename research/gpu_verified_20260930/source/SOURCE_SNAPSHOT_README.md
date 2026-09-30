# Latest completed QUAD source snapshot — 2026-09-30

This directory preserves the exact Python/CUDA source bytes associated with the completed QUAD P3 strict + trigonometric-reuse experiment (`full1024_p3_trig_gpu14_1000_v1`). It is a source and provenance snapshot, not a newly qualified portable package. No numerical code was rewritten during packaging. No experiment was started.

## What is included

- `engine/src/flowstar_gpu/`: all 36 Python files listed by the run INPUT for frozen base commit `a3fb2e94ba976aaf498c4a9cb3f98165cddcc272`. Files were recovered from local source snapshots using exact SHA-256 matches, rather than substituted from another revision.
- `adapters/`: the P3/SR, strict endpoint and controller transfer, reciprocal correction, memory-lifetime and trig-reuse sources, with the original development directory relationships retained. The latest complete entry is `adapters/quad_fullbatch_p3_20260928/run_fullbatch_p3_trig_continue.py`; the optimization is `trig_direct_reuse.py` in the same directory.
- Qualification/checker and historical prerequisite sources that the frozen entry imports or hashes are retained. Their inclusion does not mean each script is an approved alternative production entry. `SOURCE_MAP.json` marks the latest entry separately and records every file's inclusion reason.
- `integration/crown_reach.py`: byte-frozen shared Xiangru driver used by the actual run. Copying this driver does not relabel it as an independently authored component of our engine.
- `benchmark/quad_author_resolved.yaml` and `benchmark/quad_boxes.json`: the exact config (`940cb0…`) and 1024-box input (`8dd181…`). The YAML still contains the original server model path.
- `generated/reciprocal/`: the byte-frozen generated tape Python, replay CUDA and validation CUDA sources used by the reciprocal qualification, in addition to the generator in `adapters/reciprocal_geometric_20260928/`.

`SOURCE_MAP.json` gives SHA-256, byte count, original local source, recorded server path where available, role, and dependency-selection notes. The 36 engine hashes and all 26 direct COMMAND source hashes match the saved experiment. All copied Python files parse successfully; this is a packaging check, not a CUDA execution test.

## Why this is not a clean-machine one-command runtime

The original run is intentionally frozen to its server environment. Its bootstrap reads `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`, requires the base engine Git checkout to be clean at `a3fb2e…`, and checks SHA-256 before loading qualified precompiled CUDA `.so` files. It also verifies previous bounded-run/cold-replay receipts and state tensors. Those files are not replaced by this source snapshot. Relocating paths or bypassing gates would change the execution contract and was not done for this branch.

The byte-pinned QUAD ONNX model is included in `benchmark/quad_controller_3_64_torch.onnx`. NN Python dependencies, qualified binaries, source qualification receipts and large reference PT snapshots remain external assets in the original server workspace. Do not read an absolute path in the YAML or COMMAND as a promise that the file is in this Git repository. Do not delete frozen gate checks merely to make this snapshot start on another machine.

The actual completed run used Python 3.11, PyTorch 2.5.1+cu121, a CUDA 12.6 extension toolchain and a Tesla V100-SXM2 16GB. Historical upstream package metadata with different Python/Torch requirements is provenance, not the tested environment definition.

## Reading order

1. Read the branch README and experiment report for numerical scope, measured time and correctness limitations.
2. Read the latest INPUT/COMMAND evidence and `SOURCE_MAP.json` to identify the exact run.
3. Start at `run_fullbatch_p3_trig_continue.py`; the wrappers load and verify earlier entries rather than copying the entire algorithm into a new executable.
4. Treat portable packaging, a unified benchmark entry, and native plotting as subsequent work requiring validation. The source snapshot itself does not complete those tasks.

The historical P3 run's source retains `fullbatch_qualification=false` and `end_to_end_strict_certificate=false`. Completion of 1000 steps and equality to its P3 reference do not establish a full NNCS floating-point safety theorem. Huan parity timings, our P3 timings, and incomplete/timed-out native runs must remain distinct comparisons.
