# Torch trace semantic repair

## Outcome

The transition trace now records the actual objects created at each normalized-insertion lifecycle stage. It no longer fills a named stage with an adjacent object. The schema is `vdp_transition_trace_schema_v2`, and stable content hashes cover basis order, domain, center, normalization scale, sorted coefficients, Taylor-model order/truncation metadata, and interval remainder.

The solver's numerical behavior is unchanged. A T=0.2 instrumented run and an independent uninstrumented run both complete with 13 accepted steps and four rejected attempts; full-tube, last-segment, raw-endpoint, range-work, conversion and sample-sanity fields are exactly identical. Runtime is deliberately excluded from numerical equivalence because tracing adds observational overhead.

## Lifecycle mapping

| Trace stage | Post-fix source object |
|---|---|
| `step_pre_state` | exact current pre-state |
| `right_map_input` | previous state's actual `tmv_right` |
| `insertion_input` | actual endpoint-without-constants object passed to insertion |
| `insertion_output` | actual inserted Taylor-model vector |
| `normalized_reset_input` | actual inserted-for-reset object |
| `normalized_reset_output` | exact `reset_tm` object |
| `next_step_pre_state` | exact normalized reset output used for propagation |
| `right_map_output` | next state's actual `tmv_right` |

Identity tests hash every stage against the corresponding live object and reject a wrong-stage object and a lifecycle with a substituted reset output. The T=0.2 trace observes all required stages; all recorded objects have hashes. Its sole declared-unavailable row is the initial `right_map_input`, because no historical right-map state exists before the first step.

## Rejected-attempt and call-44 corrections

Rejected attempts now have `accepted_step_index=null` and an explicit `accepted_count_before_attempt`. The terminal call-44 replay occurs after 307 accepted states but belongs to a rejected attempt; it is not accepted step 307.

Call 44 is the y-component raw-RHS operation `-x²y`. X polynomial roots and aggregates are component 0; y roots, discarded routes and the call result are component 1. The exported DAG is explicitly scoped to the terminal local expression. It does not claim full cross-step ancestry.

The frozen replay still classifies all 1,141 discarded routes, has no missing parent chain, and reconstructs the chosen interval exactly. The repaired identity hash is `4ffc3344d2514bccae6cb27cfef8c1b6f6f52360c2532846544759a5ba3eb7bf`; the changed hash is expected because corrected component/attempt/scope metadata is part of the identity payload.

## Minimal reruns and evidence

- One step: one accepted step, exact requested horizon, no failure.
- Short T=0.2: covers the first 12 accepted states and the known `t≈0.181874336` schedule-divergence region; completes at T=0.2.
- Terminal frozen call 44: replay passes coverage, parent-chain and reconstruction checks.
- Pre/post semantics: `outputs/xiangru_q3_matched_audit_20260806/trace_repair/pre_fix_post_fix_field_map.md`.
- Identity/equivalence: `lifecycle_identity.json` and `instrumentation_equivalence.json`.
- Raw traces and captured commands/streams: `trace_repair/one_step/`, `short_horizon/`, `call44/`, and adjacent `*_capture/` directories.

Focused trace, insertion and Q3-audit tests pass (`32 passed`). Full-suite results are recorded separately under `outputs/.../tests/` after final verification.
