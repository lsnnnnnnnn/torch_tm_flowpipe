# Repository status closure

Run ID: `20260804T123840Z`

Scope: governance-only reconciliation from the verified native-reproduction tip
`438ee68fd71fa6182eb66cac17229e20dd3cb7d3f`.  No native matrix row was rerun,
no scientific numeric field was changed, and every row remains
`primary_comparison_eligible=false`.

## Changed status fields

| file / JSON path | old value | new value | primary evidence | reason | scientific effect |
|---|---|---|---|---|---|
| `benchmarks/native_reproduction_registry.json` / `native_reproductions[id=xiangru_diffreach_tora_u0_gpu_v100].reproduction_status` | `native_algorithm_failed` | `environment_failed` | [`command.json`](../../../outputs/native_reproduction_no_adapters/20260804T081205Z/xiangru/x2_diffreach_u0_full_native/command.json) and its hashed `stderr.log`: exit 1, `timeout_expired=false`, cuDNN f64 convolution autotuning `NOT_FOUND: No valid config found` before a step artifact | Failure is in the available GPU/cuDNN backend, before the native reachability algorithm advances. | Prevents an environment limitation from being reported as mathematical rejection; completion/certificate and comparison eligibility remain false. |
| same row / `notes` | described V100/cuDNN failure but did not disavow algorithm rejection | explicitly identifies environment/GPU-backend execution failure and says it is not native algorithm rejection | same command/traceback | Align prose with the status enum and actual failure boundary. | No numerical change. |
| `benchmarks/native_reproduction_registry.json` / `native_reproductions[id=torch_sparse_native_vanderpol_order4_t10].reproduction_status` | `native_algorithm_failed` | `runtime_timeout` | [`h10_right_map_centering_summary.csv`](../../../outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10_artifacts/h10_right_map_centering_summary.csv): both adaptive rows say `status=timeout` and “wall-time cap reached”; [`command.json`](../../../outputs/native_reproduction_no_adapters/20260804T081205Z/torch/native_order4_vdp_h10/command.json) records the declared invocation | The fresh run observed resource exhaustion at the per-lane wall cap, not the prior mathematical self-map rejection. | Preserves the validated prefixes while preventing a false causal claim; T=10 completion and comparison eligibility remain false. |
| same row / `notes` | timeout described under an algorithm-failed status | explicitly classifies runtime timeout/resource exhaustion and disavows mathematical rejection | same summary/comparison artifacts | Align prose with the observed fresh termination. | No numerical change. |
| all registry rows / `reference_evidence_location` | absent | `server_local_private_reference`, `portable_committed`, or `not_applicable` according to `reference_artifacts` paths | Registry paths plus validator filesystem/hash checks | Server-local private references must not be mistaken for portable committed evidence. | Reproduction numbers are unchanged; provenance becomes machine-auditable. |
| `scripts/native_reproduction/validate_registry.py` / `REPRODUCTION_STATUSES` | no timeout-specific status | adds `runtime_timeout` | Torch fresh summary named `timeout` | Machine enum now represents the observed termination without overloading algorithm failure. | Fail-closed schema correction only. |
| validator / reference-location invariant | absent | validates the three reference-location enums and relative/absolute/empty path consistency | Registry artifact paths | Prevents silent provenance relabeling. | No scientific result changes. |
| `docs/NATIVE_REPRODUCTION_MATRIX.md` / Xiangru GPU status cell | `native_algorithm_failed` | `environment_failed` | same cuDNN command/traceback | Human table now matches the machine registry and primary evidence. | No eligibility change. |
| same file / Torch H10 status cell | `native_algorithm_failed` | `runtime_timeout` | same fresh timeout rows | Human table no longer calls wall-cap exhaustion a solver rejection. | No eligibility change. |
| `docs/STATUS.md`, `docs/RESULTS_STATUS.md`, `docs/XIANGRU_NATIVE_REPRODUCTION.md`, `docs/XIANGRU_RESULT_TO_ENTRYPOINT_MAP.md` / corresponding status prose | GPU path described without correct enum or as `native_algorithm_failed`; Torch described as partial failure | GPU is `environment_failed`; Torch is `runtime_timeout`; both explicitly disavow mathematical rejection | same primary artifacts | Cross-document status consistency. | No numeric or gate change. |
| `docs/PROJECT_SCOPE.md` / external PyTorch identity | implementation not identified | Xiangru clean `27d29050...` complete-Q3 identified and externally audited; NNCS/controller remains out of plant-core scope | [`XIANGRU_VS_OUR_TORCH_TM_CODE_AUDIT.md`](../../../docs/XIANGRU_VS_OUR_TORCH_TM_CODE_AUDIT.md), [`XIANGRU_NATIVE_REPRODUCTION.md`](../../../docs/XIANGRU_NATIVE_REPRODUCTION.md), native manifest | Later primary source/code evidence superseded the cleanup-era unknown. | Enables a later Q3 interval-soundness audit without claiming workload or soundness equivalence. |
| `docs/EXTERNAL_PYTORCH_TM_AUDIT_PRECONDITIONS.md` / audit state | template; implementation not identified | identity/code inspection satisfied; interval-soundness still required | same clean checkout, reproduction, and field-level code audit | Separates completed identity work from the still-open numerical guarantee. | Q3 remains empirical and comparison-ineligible. |
| `docs/THREE_TOOL_FINAL_CORRECTNESS_REPORT.md` / report status | no supersession marker; stale missing-source and official-VDP-interpreter claims appeared current | prominent `SUPERSEDED STATUS REPORT` banner with current links | native matrix and official upstream VDP T=10 evidence | Preserve historical narrative without presenting disproved status as current. | No old artifact deletion or rewrite. |
| `audits/repository_cleanup/repository_cleanup_20260804T022536Z/FINAL_ACCEPTANCE.md` / report status | cleanup-era external implementation remained unidentified | prominent superseded banner pointing to identity/code evidence | same later native evidence | Preserve cleanup provenance while making supersession explicit. | No historical artifact deletion. |
| `docs/NEXT_MATCHED_EXPERIMENT_DECISION.md` / authorization | described TORA B48 as the single next experiment worth considering | TORA design retained but expressly not authorized; Q3 interval-soundness audit follows the Flow* diagnosis | open Flow* scalar-affine gate and Q3 ordinary-float64 audit | Restores dependency order between soundness and performance work. | Prevents premature matrix/TORA execution and any speedup claim. |
| `audits/repository_consolidation/20260730T083258Z/09_branch_convergence/MAIN_MERGE_PLAN.md` / selected integration lineage | old consolidation tip and direct main fast-forward recipe | current lineage through `308b735`, `438ee68`, and this closure branch; plan-only/no merge | fetched branch tips and verified ancestry | Later governance/native evidence must not be bypassed by an old merge recipe. | No main mutation; scientific gates remain independent of mergeability. |
| `README.md`, `docs/STATUS.md` / active branch | `codex/native-reproduction-no-adapters-20260804` | `codex/flowstar-scalar-affine-correctness-closure-20260804` from exact `438ee68` | clean worktree creation and `git rev-parse HEAD` | Identifies the active closure lineage without changing the frozen native evidence run. | Provenance only. |

## Machine checks

- `scripts/native_reproduction/validate_registry.py` validates 11 native rows and
  one diagnostic row.
- Targeted registry tests cover the new timeout enum, private/portable reference
  distinction, and empty-reference rule.
- The baseline repository suite before edits was `283 passed, 2 skipped` in the
  established `py11` environment.

## Closure decision

Repository status fields are reconciled without altering scientific numbers.
The Flow* correctness gate remains open and every primary comparison eligibility
flag remains false pending the scalar-affine diagnosis and all other pre-existing
soundness gates.
