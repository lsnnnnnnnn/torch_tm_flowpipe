# Reproducibility

## Current full-horizon closure

The final compact package target is
`outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z/`.
Build and verify it from a completed raw run at the exact tested source:

```bash
python experiments/build_full_horizon_pairwise_package.py \
  --source-run-root /path/to/raw-run \
  --h1-tests-dir /path/to/h1-test-logs \
  --repo-root "$(git rev-parse --show-toplevel)" \
  --tested-source-sha "$(git rev-parse HEAD)" \
  --output-root outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z

python experiments/verify_full_horizon_pairwise_package.py \
  --package-root outputs/three_tool_full_horizon_pairwise_carry_closure_20260811/20260811T191549Z \
  --expected-source-sha <H1> --require-tracked \
  --repo-root "$(git rev-parse --show-toplevel)"
```

Focused numerical entry points are
`compare_flowstar_torch_fixed_schedule.py`,
`run_diffreach_explicit_f64_full_trace.py`,
`run_torch_fixed_dr7_full_trace.py`, `trace_a3_a4_carry_state.py`,
`run_a3_a4_same_prestate_substitutions.py`,
`audit_r35_dense_cni_parity.py`, and
`audit_cni_composition_accounting.py`. Every output directory must be new.

The required delivery order is H1 tested source, a true remote clone numerical
rebuild at H1, H2 with only the compact tracked package added, a second true
remote clone verifying H2, and H3 containing only delivery evidence/docs.

## Superseded three-tool bridge builder

Build the current evidence package into a new ignored output directory:

```bash
conda run -n py11 python experiments/build_three_tool_evidence_package.py \
  --run-root outputs/three_tool_matched_divergence_fixed_support_20260811/<RUN_ID> \
  --flowstar-root /path/to/flowstar-b85a321 \
  --diffreach-root /path/to/DiffReach-dd628eb \
  --diffreach-python /path/to/pinned/diffreach/python \
  --flowstar-cxx g++
```

The canonical final run ID for this round is `20260811T100304Z`; the
`<RUN_ID>` form above remains the reusable command template.

The builder refuses an existing run root, runs every command through the
eight-file runner protocol, and executes G0→G3 only through closed
predecessors. A blocked G3 is an expected evidence outcome with a recorded
nonzero exit, not a silently passing command. The builder derives verification
claims from command files and hashes, rejects non-finite JSON and unclassified
private paths, and writes root-relative `SHA256SUMS`.

On the recorded GCC 15 environment, the unmodified pinned Flow* source needs
the compiler-only compatibility flag `-fpermissive` for an old C++11 template
body that newer GCC diagnoses eagerly. The clean-build runner records that
flag, compiler path/version/SHA, build logs, and a post-build check that no
tracked source changed. It does not patch the stock source.

The environment probe preserves the supplied DiffReach Python invocation path
instead of resolving its symlink before execution, because CPython uses that
path to discover the conda prefix. It records both invoked and resolved paths
and hashes the resolved executable.

Focused causal commands are:

```bash
python experiments/trace_vdp_raw_remainder.py --help
python experiments/analyze_vdp_raw_remainder_trace.py --help
python experiments/run_vdp_schedule_validator_matrix.py --help
python experiments/run_fixed_support_descriptor_bridge.py --help
```

From the repository root, install and test:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

At implementation commits `24bd652` and `0360842`, the expected
full-suite result is 400 passed and 2 skipped. Replay the committed original
terminal fixture with the promoted fail-trigger policy:

```bash
conda run -n py11 python experiments/replay_vdp_terminal_range.py \
  --checkpoint evidence/vdp_terminal_range_closure/20260805T055556Z/02_terminal_state_replay/original_terminal_checkpoint \
  --output-dir /tmp/vdp-terminal-replay-new \
  --range-method adaptive_subdivision --subdivision-depth 1 \
  --max-leaves 4 --split-vars 0,1 \
  --named-contexts polynomial_truncation \
  --trigger on_validation_failure --device cpu
```

The expected result accepts the unchanged terminal h with x/y margins
`9.96013970567558e-5` and `2.8883253329832075e-5`, zero fallback, and the same
candidate coefficient/support hashes as natural replay.

Run the final fresh policy into a new directory:

```bash
conda run -n py11 python experiments/run_vdp_dense_backend.py \
  --output-dir /tmp/vdp-terminal-range-t10-new \
  --tm-backend dense --device cpu --horizon 10 --wall-cap-s 3600 \
  --dense-range-method adaptive_subdivision \
  --dense-range-trigger proactive_depth1_on_named_contexts \
  --dense-range-max-depth 1 --dense-range-max-leaves 4 \
  --dense-range-split-vars 0,1 \
  --dense-range-contexts polynomial_truncation
```

This command is expected to exit nonzero after writing a complete fail-closed
result at `6.397083942944808`; it is not expected to complete T=10. Verify the
tracked evidence, including deterministic gzip files, with:

```bash
cd evidence/vdp_terminal_range_closure/20260805T055556Z
sha256sum -c SHA256SUMS
find . -type f -name '*.gz' -print0 | xargs -0 -n1 gzip -t
```

If the corresponding local `outputs/vdp_terminal_range_closure/20260805T055556Z`
tree is present, its tracked evidence form can be regenerated with
`experiments/package_vdp_terminal_range_evidence.py`. The packager records the
source and stored hashes for every compressed raw file.

Run the canonical dense VDP lane into a new directory:

```bash
conda run -n py11 python experiments/run_vdp_dense_backend.py \
  --output-dir /tmp/vdp-dense-t10-new \
  --tm-backend dense --device cpu --horizon 10 --wall-cap-s 480
```

The following command and expected result describe the preceding natural S3
baseline at commit `6bf0d9a6...`: it is a fail-closed
`minimum_step_reached` at `6.3172908799330765`, not T=10 completion. The sole
diagnostic factor is reproduced by adding
`--right-map-center-mode range_midpoint`; it must remain labeled diagnostic.

Run the production operator microbenchmark with synchronized CUDA timing:

```bash
conda run -n py11 python experiments/batched_tm_gpu_microbench.py \
  --output-dir /tmp/dense-tm-microbench-new --dtype float64 \
  --batches 1,8,32,48,128 --devices cpu,cuda \
  --warmup 1 --repeats 5 --max-scalar-batch 8
```

Verify the committed closure bundle:

```bash
cd evidence/generic_batched_tm_backend_vdp_t10/20260804T152536Z
sha256sum -c SHA256SUMS
```

Resolve the read-only sibling dependencies without hard-coded private paths:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
WORK_PARENT="$(dirname "$REPO_ROOT")"
export DIFFREACH_ROOT="$WORK_PARENT/DiffReach"
export FLOWSTAR_ROOT="$WORK_PARENT/flowstar"
export DIFFREACH_PYTHON="/path/to/pinned/diffreach/environment/bin/python"
```

Run smoke into a path that does not yet exist:

```bash
python experiments/consolidated_study/cli.py smoke \
  --output-dir /tmp/torch-tm-flowpipe-smoke-new
```

The smoke run is pipeline evidence only. It records backend identity and
rejects a non-empty output directory.

Do not run formal while any gate in `benchmarks/cross_tool_gates.yaml` is
pending. The command is intentionally fail-closed before output creation.
Once a later independent audit verifies every gate, a clean code freeze can
use:

```bash
python experiments/consolidated_study/cli.py formal \
  --output-dir "artifacts/runs/$(date -u +%Y%m%dT%H%M%SZ)"
```

Formal also refuses a dirty worktree, an invalid or contaminated Flowstar
backend, and a non-empty destination. It records command, environment,
external SHAs/status, patch and library hashes, configuration identity,
checksums, and the independent audit.

Frozen historical bundles may be checksum-checked, but rerunning the current
auditor against them is expected to fail the strengthened backend and bound
contracts:

```bash
cd artifacts/runs/20260730T153654Z
shasum -a 256 -c SHA256SUMS
```

Run `20260730T153654Z` is frozen but withdrawn because it used a patched audit
backend. A new run must use a new run ID, and timings from different hardware
must not be combined.
