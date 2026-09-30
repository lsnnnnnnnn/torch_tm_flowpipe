# ARCH-COMP26 four-way execution status

> Automatically generated status report; this is not the final experiment report.
> Historical evidence is never promoted into an unexecuted 2026 matrix cell.

## Boundary

- Experiments paused: **yes**.
- Instances: 16; methods: 4; cells: 64.
- Run states: `not_started=64`.
- Support states: `unassessed=64`.
- Resume gate: Explicit user authorization is required. Before any launch, read-only recheck the original native QUAD job's actual terminal state and do not duplicate it.

## Input identities

| Input | Repository-relative path | SHA-256 |
|---|---|---|
| `manifest` | `benchmarks/archcomp26/manifest.json` | `6396a82a282aba1ffba97aae7ebcf6af3313df5141f575363ae3efad9a330c2b` |
| `matrix` | `benchmarks/archcomp26/execution_matrix.json` | `ab61dd16f8919596d968b226831a52a5c4f5d02cd5d8a5daa89d72935cbdbce4` |
| `contract_audits` | `benchmarks/archcomp26/evidence/contract_audits_20261001.json` | `63ad3c7d6f120f799475a6fcaaec0e9a9036bcc8bf20e6c30aa70944d26be5e6` |
| `huan_parity` | `research/gpu_verified_20260930/report/evidence/huan_parity_campaign.json` | `8651b97943518f6d16e6d73787df624292b3b903ddb9001c5d578cb823144947` |
| `p3_audit` | `research/gpu_verified_20260930/report/evidence/quad_trig_audit.json` | `517bd183820d43af794d92b64434d84e05e9fb4fa120927f2e9bf3c41968add2` |
| `plot_receipt` | `docs/evidence/flowpipe_plot_validation_20261001.json` | `5ab3ba17da3965ad5c51451ae8f6110f0b87d41b954ce05a7d71b2823c19b36e` |
| `plot_contract_status` | `benchmarks/plot_specs/archcomp26_status.json` | `79e0caf03cab764062fb5c43f9af706610c3c76b083f8d001a3011950a5b9e86` |
| `native_recheck` | `docs/evidence/remote_native_quad_recheck_20261001.json` | `2655c3ec43f52e00d944810812ff02c3ed3b8187212dc98e9467120668293062` |

## Contract audit coverage

Every row remains an unresolved execution contract. Audit status records what is known and why launch is still blocked.

| Instance | Contract | Audit | Audit status | Known issues |
|---|---|---|---|---:|
| `acc-safe-distance` | `unresolved` | `acc` | state_time_and_property_contract_known; relative_velocity_input_definition_unresolved | 3 |
| `airplane-continuous` | `unresolved` | `airplane` | continuous_time_semantics_resolved; discrete_transition_unresolved | 1 |
| `airplane-discrete` | `unresolved` | `airplane` | continuous_time_semantics_resolved; discrete_transition_unresolved | 1 |
| `attitude-control-avoid` | `unresolved` | `attitude_control` | state_time_and_unsafe_set_known; controller_selection_and_property_polarity_unresolved | 3 |
| `balancing-reach` | `unresolved` | `balancing` | state_initial_set_and_horizon_known; controller_input_and_t8_boundary_unresolved | 3 |
| `docking-constraint` | `unresolved` | `docking` | state_controller_period_horizon_and_property_known; full_execution_contract_unresolved | 3 |
| `double-pendulum-less-robust` | `unresolved` | `double_pendulum` | official_variants_and_properties_known; legacy_more_robust_result_not_promotable | 1 |
| `double-pendulum-more-robust` | `unresolved` | `double_pendulum` | official_variants_and_properties_known; legacy_more_robust_result_not_promotable | 1 |
| `nav-standard` | `unresolved` | `navigation` | time_and_property_semantics_known; state_order_and_controller_input_contract_unresolved | 2 |
| `nav-robust` | `unresolved` | `navigation` | time_and_property_semantics_known; state_order_and_controller_input_contract_unresolved | 2 |
| `quad-reach` | `unresolved` | `quadrotor` | dual_dynamics_contracts_frozen; selection_unresolved | 2 |
| `single-pendulum-reach` | `unresolved` | `single_pendulum` | two_state_physical_contract_known; repository_clock_augmentation_unresolved | 1 |
| `tora-remain` | `unresolved` | `tora` | variants_identified; controller_boundary_activation_and_reach_checker_details_unresolved | 1 |
| `tora-reach-sigmoid` | `unresolved` | `tora` | variants_identified; controller_boundary_activation_and_reach_checker_details_unresolved | 1 |
| `tora-reach-tanh` | `unresolved` | `tora` | variants_identified; controller_boundary_activation_and_reach_checker_details_unresolved | 2 |
| `unicycle-reach` | `unresolved` | `unicycle` | nominal_state_controller_contract_known; disturbance_and_reach_checker_semantics_unresolved | 2 |

## 16×4 execution matrix

| Instance | PyTorch/GPU | Huan | Xiangru | Flow* native |
|---|---|---|---|---|
| `acc-safe-distance` | `not_started` | `not_started` | `not_started` | `not_started` |
| `airplane-continuous` | `not_started` | `not_started` | `not_started` | `not_started` |
| `airplane-discrete` | `not_started` | `not_started` | `not_started` | `not_started` |
| `attitude-control-avoid` | `not_started` | `not_started` | `not_started` | `not_started` |
| `balancing-reach` | `not_started` | `not_started` | `not_started` | `not_started` |
| `docking-constraint` | `not_started` | `not_started` | `not_started` | `not_started` |
| `double-pendulum-less-robust` | `not_started` | `not_started` | `not_started` | `not_started` |
| `double-pendulum-more-robust` | `not_started` | `not_started` | `not_started` | `not_started` |
| `nav-standard` | `not_started` | `not_started` | `not_started` | `not_started` |
| `nav-robust` | `not_started` | `not_started` | `not_started` | `not_started` |
| `quad-reach` | `not_started` | `not_started` | `not_started` | `not_started` |
| `single-pendulum-reach` | `not_started` | `not_started` | `not_started` | `not_started` |
| `tora-remain` | `not_started` | `not_started` | `not_started` | `not_started` |
| `tora-reach-sigmoid` | `not_started` | `not_started` | `not_started` | `not_started` |
| `tora-reach-tanh` | `not_started` | `not_started` | `not_started` | `not_started` |
| `unicycle-reach` | `not_started` | `not_started` | `not_started` | `not_started` |

## Archived reference evidence (not matrix results)

| Evidence | Coverage | Time | Guarantee boundary |
|---|---|---:|---|
| Huan QUAD parity/box | 5 complete runs; 1,024,000 accepted lane-steps each | process median 75.250099 s (driver 72.173857 s; internal 70.727803 s) | parity reproduction; not a strict NNCS certificate |
| Current PyTorch P3 + trig reuse | 1000 steps; 1024000 accepted lane-steps | single watchdog run 1533.752052 s | fullbatch qualification `false`; end-to-end strict certificate `false` |
| Native QUAD original job | 600 complete steps; 614400 accepted lane-steps | timeout after 21609.612654 s | no complete T=5 time or endpoint width |

The Huan/P3 contracts differ in order, validation, arithmetic guarantees, instrumentation, and timing boundaries; their times are not a same-contract speedup ratio.

## Plotting status

- Validation date: 2026-10-01.
- Saved artifact sets checked: 5.
- Independent geometry check: `passed`.
- Experiments started by plotting validation: `false`.
- Plot-contract accounting: materialized v2=8; materialized v3=1; axis-aligned content ready=0; fail-closed blocked=7.
- MATLAB: `not found`; Octave: `not found`. Generated `.m` files have static checks only.
- Saved-artifact evidence covers axis-aligned QUAD projections; targeted tests also cover the declared ACC affine interval image. Neither establishes native octagon/support-direction parity.
- Plot content status does not resolve any method's execution contract.

| Instance | Plot content status | Content blockers |
|---|---|---:|
| `acc-safe-distance` | `materialized_v3_content_contract` | 0 |
| `airplane-continuous` | `materialized_v2_content_contract` | 0 |
| `airplane-discrete` | `blocked_fail_closed` | 2 |
| `attitude-control-avoid` | `blocked_fail_closed` | 1 |
| `balancing-reach` | `blocked_fail_closed` | 2 |
| `docking-constraint` | `blocked_fail_closed` | 1 |
| `double-pendulum-less-robust` | `materialized_v2_content_contract` | 0 |
| `double-pendulum-more-robust` | `materialized_v2_content_contract` | 0 |
| `nav-standard` | `materialized_v2_content_contract` | 0 |
| `nav-robust` | `materialized_v2_content_contract` | 0 |
| `quad-reach` | `materialized_v2_content_contract` | 0 |
| `single-pendulum-reach` | `materialized_v2_content_contract` | 0 |
| `tora-remain` | `materialized_v2_content_contract` | 0 |
| `tora-reach-sigmoid` | `blocked_fail_closed` | 1 |
| `tora-reach-tanh` | `blocked_fail_closed` | 1 |
| `unicycle-reach` | `blocked_fail_closed` | 2 |

## Next gate

Do not launch from this report. After explicit experiment-resume authorization, first recheck the original native QUAD job identity, PIDs, output directories, and any replacement run read-only. Never overwrite or duplicate `full1000_v1`. Then resolve a cell's contract, support, command, source/binary identity, and runtime budget before its preflight can pass.
