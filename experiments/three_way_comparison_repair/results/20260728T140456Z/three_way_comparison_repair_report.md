# Three-way comparison correctness repair

## 1. Executive summary

The repair selects **Outcome B**.

The historical three-way ranking is invalid. The generated Flow* adapter overwrote the native refined remainder after every successful `advance`, and Torch's displayed endpoint used an additional fixed-time residual tightening that Flow* and DiffReach did not use.

Stock Flow* reproduces the Riccati under-enclosure: at h=0.01 its upper endpoint misses the analytic upper bound by 1.24236723825e-08. A regenerated full-Picard inclusion test rejects the remainder-only refinement image. The original Flow* Van der Pol benchmark nonetheless reaches T=10 with 290 segments, and both identical-settings generated harnesses reproduce its schedule.

Accordingly, the report publishes the semantics-corrected Torch versus DiffReach rows, keeps Flow* stock and diagnostics visible, and issues no three-way width ranking.

## 2. Why the previous experiment was invalid

The old figures mixed endpoint semantics and changed one solver's output after validation. Candidate reinjection was presented as an extraction workaround, Torch `final_tm` was a fixed-time residual recomputation, the Flow* wrapper reduced all failures to one integer, and fixed low order was mistaken for general tool capability. Common-box resets also removed the dependencies that native carry is designed to preserve.

## 3. Exact code-level confounders

- **Flow* remainder overwrite (confirmed code fact):** the historical generated C++ assigned `setting.tm_setting.remainder_estimation[state]` to `next.tmvPre.tms[state].remainder` after `advance`.
- **Torch endpoint tightening (confirmed code fact):** `flowpipe_step_from_tm` re-evaluated the Picard residual at `tau=h` and stored that result in `final_tm`. The repaired API exposes raw and tightened endpoint TMs.
- **Generic Flow* failure reporting (confirmed code fact):** return code 0 lost the failing inclusion state and source site. The audit build emits structured return reasons and refinement rounds.
- **Low-order configuration (confirmed code fact):** the old Flow* row used fixed order 2 even though the upstream benchmark uses adaptive steps and order 4.
- **Reset semantics (inference from code paths):** common-box carry restarts every native representation from an axis-aligned box.

## 4. Historical result reproduction

Top-level historical artifact regeneration status: `exact`. Report match: `True`; plot matches: 7/7. The frozen directory itself was never used as an output directory.

## 5. Flow* original benchmark parity

The actual local upstream benchmark reached T=10: `True`. Original/generated/generic segment counts are 290/290/290. Schedule agreement is `True` and generated versus generic bound agreement is `True`.

## 6. Flow* stock refinement investigation

At Riccati h=0.01 the stock raw endpoint width is 0.100125100228; candidate reinjection produces 0.1003. The stock upper miss is 1.24236723825e-08. The diagnostic that fully revalidates the refined remainder returns width 0.100125650631 and contains the analytic endpoint.

The first source-level failing operation is the remainder-only refinement acceptance. Its final scalar remainder is not a self-map when `Picard_ctrunc_normal` and the polynomial-difference interval are regenerated. The audit trace records `subset=0` and restores the already accepted initial remainder. This is a conservative diagnostic fallback, not a merged upstream fix.

Repeating the base run with `intervalNumPrecision=256` produces the same first-step upper bound to the exported precision and therefore does not remove the violation. This is evidence against default 53-bit numeric rounding as the first cause.

## 7. Exact Flow* failure classification

| tool_variant | protocol | system | h | successful_horizon | failure_category | failure_message |
| --- | --- | --- | --- | --- | --- | --- |
| flowstar_stock | one_step_tube | riccati | 0.05 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | one_step_tube | riccati | 0.05 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_stock | one_step_raw_endpoint | riccati | 0.05 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | one_step_raw_endpoint | riccati | 0.05 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_stock | one_step_tube | harmonic | 0.05 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | one_step_tube | harmonic | 0.05 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_stock | one_step_raw_endpoint | harmonic | 0.05 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | one_step_raw_endpoint | harmonic | 0.05 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_stock | common_box_raw_endpoint_carry | harmonic | 0.01 | 2.3099999999999947 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | common_box_raw_endpoint_carry | harmonic | 0.01 | 2.2199999999999966 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=0 source_line=1120 |
| flowstar_stock | one_step_tube | van_der_pol | 0.01 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | one_step_tube | van_der_pol | 0.01 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | one_step_raw_endpoint | van_der_pol | 0.01 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | one_step_raw_endpoint | van_der_pol | 0.01 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | one_step_tube | van_der_pol | 0.02 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | one_step_tube | van_der_pol | 0.02 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | one_step_raw_endpoint | van_der_pol | 0.02 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | one_step_raw_endpoint | van_der_pol | 0.02 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | common_box_raw_endpoint_carry | van_der_pol | 0.005 | 0.105 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | common_box_raw_endpoint_carry | van_der_pol | 0.005 | 0.075 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | native_representation | van_der_pol | 0.005 | 0.14 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | deliberate_low_order_stress | van_der_pol | 0.005 | 0.14 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | common_box_raw_endpoint_carry | van_der_pol | 0.01 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_candidate_reinjection_diagnostic | common_box_raw_endpoint_carry | van_der_pol | 0.01 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | native_representation | van_der_pol | 0.01 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |
| flowstar_stock | deliberate_low_order_stress | van_der_pol | 0.01 | 0.0 | first_picard_inclusion_failed | Flowpipe::advance fixed-step/fixed-order first Picard image was not a subset; state=1 source_line=1120 |

Every observed failure has a structured category. `unknown_internal_failure` is retained only when no instrumented return record exists.

## 8. Torch endpoint semantics

For each validated segment, `endpoint_raw_tm` is the direct substitution of `tau=h` in the validated segment. `endpoint_tightened_tm` uses the fixed-time residual formula described in `TORCH_ENDPOINT_AUDIT.md`. `final_tm` remains the tightened endpoint for backward compatibility.

Riccati h=0.01 raw width: **0.100125000003**. Tightened width: **0.100100250143**. The primary protocols use only the former.

## 9. DiffReach endpoint semantics

The adapter invokes the saved upstream `CT_Dyn_Reach.step_once`, including upstream Picard construction, remainder refinement, and symbolic carry. It composes the returned local TM with the upstream parameterization, evaluates the full time box for the tube, and fixes time to h for the raw endpoint. There is no endpoint-specific residual recomputation. The sole numeric override changes the upstream float32 constructor default to explicit x64.

## 10. Corrected comparison protocols

- **A — one_step_tube:** identical ODE, initial box, and h; full segment.
- **B — one_step_raw_endpoint:** direct h-substitution in each validated segment.
- **C — common_box_raw_endpoint_carry:** only raw endpoint boxes are carried.
- **D — native_representation:** stock native carry; Torch raw and legacy tightened carry are distinct variants.
- **E — deliberate_low_order_stress:** Torch order 1, DiffReach affine, Flow* minimum legal order 2; diagnostic only.
- **F — known_working_tool_sanity:** original Flow* Van der Pol configuration.

## 11. Corrected one-step tube results

See `corrected_one_step_summary.csv`. Only rows passing their applicable correctness gates are interpretable; stock Flow* Riccati rows are retained as failed audit evidence.

## 12. Corrected raw-endpoint results

| tool | tool_variant | system | h | state_index | lower | upper | width | inflation_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | torch_order1 | riccati | 0.005 | 0 | -6.250001251250028e-06 | 0.1000562500012513 | 0.1000625000025025 | 1.0001246875250132 |
| torch_tm_flowpipe | torch_order1 | riccati | 0.01 | 0 | -1.2500001252500047e-05 | 0.1001125000012525 | 0.100125000002505 | 1.0002487500250252 |
| torch_tm_flowpipe | torch_order1 | riccati | 0.02 | 0 | -2.50000012550001e-05 | 0.100225000001255 | 0.10025000000251 | 1.00049500002505 |
| torch_tm_flowpipe | torch_order1 | riccati | 0.05 | 0 | -6.250000126250035e-05 | 0.1005625000012625 | 0.100625000002525 | 1.0012187500251242 |
| torch_tm_flowpipe | torch_order1 | harmonic | 0.005 | 0 | -0.1006250000012562 | 0.1006250000012562 | 0.2012500000025125 | 1.0012562552727948 |
| torch_tm_flowpipe | torch_order1 | harmonic | 0.005 | 1 | -0.1006250000012562 | 0.1006250000012562 | 0.2012500000025125 | 1.0012562552727948 |
| torch_tm_flowpipe | torch_order1 | harmonic | 0.01 | 0 | -0.1012500000012625 | 0.1012500000012625 | 0.2025000000025251 | 1.002525042508619 |
| torch_tm_flowpipe | torch_order1 | harmonic | 0.01 | 1 | -0.1012500000012625 | 0.1012500000012625 | 0.2025000000025251 | 1.002525042508619 |
| torch_tm_flowpipe | torch_order1 | harmonic | 0.02 | 0 | -0.102500000001275 | 0.102500000001275 | 0.2050000000025501 | 1.0051003465573287 |
| torch_tm_flowpipe | torch_order1 | harmonic | 0.02 | 1 | -0.102500000001275 | 0.102500000001275 | 0.2050000000025501 | 1.0051003465573287 |
| torch_tm_flowpipe | torch_order1 | harmonic | 0.05 | 0 | -0.1062500000013125 | 0.1062500000013125 | 0.2125000000026251 | 1.0131307179506457 |
| torch_tm_flowpipe | torch_order1 | harmonic | 0.05 | 1 | -0.1062500000013125 | 0.1062500000013125 | 0.2125000000026251 | 1.0131307179506457 |
| torch_tm_flowpipe | torch_order1 | van_der_pol | 0.0025 | 0 | 1.0992343749987463 | 1.4068906250012536 | 0.3076562500025073 | nan |
| torch_tm_flowpipe | torch_order1 | van_der_pol | 0.0025 | 1 | 2.331791249998718 | 2.458828750001255 | 0.1270375000025367 | nan |
| torch_tm_flowpipe | torch_order1 | van_der_pol | 0.005 | 0 | 1.098468749998743 | 1.4137812500012568 | 0.3153125000025136 | nan |
| torch_tm_flowpipe | torch_order1 | van_der_pol | 0.005 | 1 | 2.313582499998688 | 2.467657500001258 | 0.1540750000025705 | nan |
| torch_tm_flowpipe | torch_order1 | van_der_pol | 0.01 | 0 | 1.096937499998737 | 1.427562500001263 | 0.3306250000025261 | nan |
| torch_tm_flowpipe | torch_order1 | van_der_pol | 0.01 | 1 | 2.277164999998627 | 2.485315000001265 | 0.2081500000026381 | nan |
| torch_tm_flowpipe | torch_order1 | van_der_pol | 0.02 | 0 | 1.0938749999987245 | 1.4551250000012756 | 0.3612500000025511 | nan |
| torch_tm_flowpipe | torch_order1 | van_der_pol | 0.02 | 1 | 2.204329999998505 | 2.5206300000012787 | 0.3163000000027738 | nan |
| diffreach | diffreach_affine | riccati | 0.005 | 0 | -1.251876542446231e-05 | 0.1000500469342393 | 0.1000625656996637 | 1.0001253441681397 |
| diffreach | diffreach_restricted_quasi_quadratic | riccati | 0.005 | 0 | -1.2518777162792577e-05 | 0.1000500375475044 | 0.1000625563246672 | 1.000125250465049 |
| diffreach | diffreach_affine | riccati | 0.01 | 0 | -2.507513055141797e-05 | 0.1001001879815906 | 0.100125263112142 | 1.0002513784902989 |
| diffreach | diffreach_restricted_quasi_quadratic | riccati | 0.01 | 0 | -2.5075224614962788e-05 | 0.100100150387582 | 0.1001252256121969 | 1.0002510038658476 |
| diffreach | diffreach_affine | riccati | 0.02 | 0 | -5.0301053927827846e-05 | 0.1002007538706064 | 0.1002510549245342 | 1.0005055281468522 |
| diffreach | diffreach_restricted_quasi_quadratic | riccati | 0.02 | 0 | -5.030180895979276e-05 | 0.1002006031164548 | 0.1002509049254146 | 1.000504031155638 |
| diffreach | diffreach_affine | riccati | 0.05 | 0 | -0.0001268916086385 | 0.1005047485163336 | 0.1006316401249721 | 1.001284819243473 |
| diffreach | diffreach_restricted_quasi_quadratic | riccati | 0.05 | 0 | -0.0001269035258413 | 0.1005037991338118 | 0.1006307026596531 | 1.0012754914635489 |
| diffreach | diffreach_affine | harmonic | 0.005 | 0 | -0.100502512562121 | 0.100502512562121 | 0.201005025124242 | 1.0000374595995027 |
| diffreach | diffreach_affine | harmonic | 0.005 | 1 | -0.100502512562121 | 0.100502512562121 | 0.201005025124242 | 1.0000374595995027 |

## 13. Corrected common-box-carry results

See `corrected_common_time_summary.csv`. Widths are compared only at equal absolute time. Failed segments contribute no later points.

## 14. Native-representation results

Native rows are configuration-specific and do not constitute a common-basis ranking. Torch raw-endpoint carry and legacy tightened-endpoint carry are separate variants.

## 15. Deliberate low-order stress results

These rows intentionally use different legal minima and are never described as same-order performance.

## 16. Runtime results with implementation caveats

Flow* compile time, DiffReach JIT time, first execution, and steady per-step time are separate schema fields. No combined winner is reported.

## 17. Numerical soundness differences

Flow* uses directed MPFR interval arithmetic, Torch uses float64 tensor interval operations, and DiffReach uses JAX float64 interval-style arithmetic. Analytic references are proofs for the two closed-form systems; deterministic Van der Pol trajectories are bug-catching checks, not proofs.

## 18. Claim-by-claim correction

| old_claim | status | confounder | corrected_wording |
| --- | --- | --- | --- |
| Torch is tightest on Riccati | invalid | endpoint postprocessing mismatch | Torch tightening is supplemental; raw endpoint comparisons are required |
| Flow* fails around t=0.08 | invalid | constrained stress configuration and generic failure code | that configuration fails; stock Flow* reaches T=10 under its original settings |
| Flow* order 2 is less capable | invalid | different minimum legal bases and resource settings | order-2 fixed stress is diagnostic, not general capability |
| DiffReach is tighter on harmonic | unresolved | tool-specific local bases remain unmatched | report matched raw semantics and configuration caveats only |
| DiffReach is faster | invalid | incomparable one-time and steady costs | compile, JIT, warmup, and steady costs are separate |
| common-box comparison is fair | corrected | box reset hides native dependency-preservation behavior | common-box controls carry representation but is not a native-method ranking |
| all correctness gates passed | invalid | stock Flow* analytic violation was excluded by postprocessing | stock Riccati exact-reference checks fail; Outcome B applies |
| Flow* refinement was unvalidated | confirmed | no full-Picard post-refinement recheck had been executed | the remainder-only refined image fails a regenerated full-Picard inclusion check |

## 19. Confirmed facts

- The historical Flow* adapter overwrote native remainders.
- Torch fixed-time tightening materially changes Riccati width.
- The original Flow* benchmark reaches T=10 in the audited build.
- Stock Flow* refined Riccati under-enclosure is reproducible.
- Full-Picard revalidation rejects the remainder-only refined scalar image.

## 20. Unresolved questions

The audit isolates the invalid refinement acceptance but does not supply a general upstream proof or patch for every ODE/order/precision combination. The exact internal reason cached remainder-only evaluation diverges from a regenerated full Picard image remains an upstream algorithm question.

## 21. Recommendation and decision

**Outcome B.** A valid three-way width ranking is not currently possible. Publish the corrected Torch-versus-DiffReach raw-semantic tables and the Flow* original-parity sanity result separately.

## Correctness counts

```json
{
  "exact_reference_failed": 316,
  "exact_reference_passed": 14160,
  "exact_reference_rows": 14476,
  "failure_rows": 78,
  "flowstar_stock_exact_failures": 316,
  "raw_rows": 17931,
  "torch_diffreach_trajectory_checks_failed": 0,
  "trajectory_checks_failed": 308,
  "trajectory_checks_passed": 17544
}
```

## Repository provenance

```json
{
  "diffreach": {
    "path": "/srv/local/shengenli/DiffReach",
    "sha": "dd628eb443b517d6415de93e7035b4baef73963e"
  },
  "flowstar_audit": {
    "path": "/srv/local/shengenli/flowstar_three_way_audit",
    "sha": "316128e46202605c47d8131870dff265f96b7f3b"
  },
  "flowstar_original": {
    "path": "/srv/local/shengenli/flowstar",
    "sha": "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
  },
  "torch_base_branch": "codex/three-way-common-contract-comparison",
  "torch_base_sha": "7251adfe8d2f3a5f3fd7a4a89f4b5a2075a19b10",
  "torch_repair": {
    "path": "/srv/local/shengenli/torch_tm_flowpipe_three_way_repair",
    "sha": "04f1f1a5d277f77ab09bad903f05f645a4778419"
  }
}
```

## Reproduction

```bash
cd /srv/local/shengenli/torch_tm_flowpipe_three_way_repair
export FLOWSTAR_ROOT=/srv/local/shengenli/flowstar_three_way_audit
experiments/three_way_comparison_repair/run_smoke.sh
experiments/three_way_comparison_repair/launch_background.sh
```

The tmux launcher prints the exact session, command, log, result path, progress command, and safe stop command.
