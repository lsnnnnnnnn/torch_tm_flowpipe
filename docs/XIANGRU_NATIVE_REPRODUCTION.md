# Xiangru native reproduction

The real repository is `/srv/local/shengenli/CROWN-Reach_Development`, remote
`https://github.com/xiangruzh/CROWN-Reach_Development.git`, branch
`2026_experiment`; fresh fetch found local and origin at clean
`84184de6c2b3f1ff2da6755f732d91925037025d`.  Saved results identify earlier
generating commits, so the runs use clean detached worktrees at `27d29050...`
(CROWN-Reach/Flow* and Q3) and `9bf2ccea...` (DiffReach provenance), not the current
branch as a substitute.

## Environment actually used

- CROWN-Reach/Q3: Python 3.11.15, PyTorch 2.8.0+cu128, NumPy 2.3.3,
  onnx 1.22.0 and exact auto_LiRPA checkout `5a098e8...`; CUDA device is a Tesla
  V100-SXM2-16GB.
- X1: the author 27d Dockerfile built under rootless Docker, image ID
  `sha256:6549fefc...`, GCC 11.4.0, with all image inspection and doctor logs saved.
- DiffReach: Python 3.12.13.  The author U0 path uses its JAX 0.8.3 environment;
  the upstream official paths use the README-exact JAX/JAXlib 0.8.1 environment.

The saved machine was an RTX 5090.  Device identity is not treated as equal and no
runtime comparison is made.

## X1: native CROWN-Reach controller plus Flow* plant

The exact author command is:

```bash
scripts/reachbench_crown_docker.sh run tora_homogeneous \
  --backend crownreach_submit --profile full --timeout 180
```

The fresh native run exits 0, executes 20 controller updates for 12 initial
partitions, reaches T=20, returns `VERIFIED`, and reports no Flow* termination.
The exact comparator passes config/manifest, return code, execution contract,
certification state and verdict.  Status: `reproduced_exact`.  Runtime is excluded.

Evidence: `xiangru/x1_crownreach_flowstar_full_rootless/{command.json,
fresh_artifacts/run.json,comparison.json}` under the run root.

## X2: Xiangru's native upstream DiffReach experiment

The byte-identical U0 config fixes 12 partitions, 200 steps of 0.1 seconds and a
one-second controller period.  The unmodified GPU command enters verification but
the available V100/cuDNN stack cannot select a valid float64 convolution during
XLA autotuning.  It exits 1 before the first step and writes no flowpipe.  Status:
`native_algorithm_failed`; the failure is preserved rather than replacing the
convolution or controller.

A clearly separated CPU supplemental run uses the same config and native
`run_ctl.py`.  It completes all 200 process steps, warns that only 66.22% of Picard
shrink flags are true, and produces extremely large terminal bounds.  Its NPZ is
byte-identical (SHA `48d68f0a...`) to the author's saved CPU NPZ.  Status:
`reference_failure_reproduced`; completion does not become a certificate or
property success.

Evidence: `xiangru/x2_diffreach_u0_full_{native,cpu_native}` and
`xiangru/x2_diffreach_u0_native/fresh_workspace_cpu/flowpipe/flowpipe_ver.npz`.

## X3: native complete-Q3 + DR-RP B48

The exact config was regenerated through the author's native
`reachbench.prepare_run`, producing the saved config hash `13b28ac2...` and
controller hash `bb80479c...`.  The exact author entrypoint then ran with CUDA,
B48, complete-Q3, cap 200, dynamic engine, auto_LiRPA CUDA/eager controller and
outward controller composition.

The fresh run reaches all 200 segments/T=20 for 48 leaves, has no retry segment,
passes CPU/CUDA replay and the full-tube property.  Across 2,850 non-timing numeric
fields the maximum absolute difference is `1.4210854715202004e-13` and maximum
relative difference is `1.6653345369377348e-9`.  Both are below the author's
pre-existing `CONTROLLER_TOLERANCE=1e-6`; 154 runtime/memory fields are excluded.
Status: `reproduced_with_declared_tolerance`.

This validates the native result but is not upgraded to formal outward rounding:
the dynamics interval primitives use ordinary float64 operations.  Soundness is
therefore `empirical` for this registry.

## Historical B12/B24 sweep

Separate clean-27d native invocations avoid the runner's persistent-controller
batch-shape limitation.  B12 first fails at attempted segment 151 and certifies
through T=15.0; B24 first fails at segment 181 and certifies through T=18.0.  These
boundaries numerically agree closely with the historical sweep, but that artifact
records commit `0a2cfcf...` with `crown_reach_dirty=true` and no saved patch.
Statuses remain `source_identity_unknown`, not reproduced.

## Tests and limits

The exact-27d full suite in the task environment executes 116 tests: 111 pass, 4
skip and one repository-consistency test fails because an ignored historical
`experiments/reachability/results/.../run.json` named by
`tora_tradeoff_matrix.json` is absent from both the clean worktree and canonical
checkout.  The missing file is not fabricated.  The first py11 attempts that
failed collection for missing `onnx`, and the task-env attempt before installing
pytest, are also preserved.

The existing patched shared first-step trace was not rerun: the completed
end-to-end and code evidence already localize the material representation,
roundoff and reset differences.  It remains ineligible for native or long-horizon
claims.
