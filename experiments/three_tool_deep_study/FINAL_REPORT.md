# Authoritative three-tool deep-study report

## Delivery status

This report describes the passed `20260730T015245Z` artifact on branch
`codex/torch-flowstar-diffreach-deep-study`. It is authoritative for this
repository; the default branch may not yet include it.

- Final acceptance: **True**
- Ten-repetition gate required: **True**
- Artifact-quality audit: **True**
- Parsed CSV rows: **195551**
- Mandatory plots: **18**
- Repeated native configuration/system rows: **24**
- Explicitly classified failure rows: **32**
- Curated artifact: `artifacts/authoritative/20260730T015245Z`
- Numerical-run complete matrix: **354 passed / 5 skipped / 0 failed**
- Final-code canonical sandbox matrix: **350 passed / 10 skipped / 0 failed**

The numerical producer is frozen at `129b633`; report-only checkpoint
`b0d20d4` removes supplemental tightened Torch endpoints from the broad
raw/native table and plot without changing numerical CSVs.  The five
additional sandbox skips reflect unavailable host CUDA/external interfaces;
`FINAL_DELIVERY_TEST_RECORD.md` records the exact groups and log digest.

## What changed

The earlier literal “order-1 winner” claim is retracted. Torch, DiffReach, and
Flow* do not attach the same basis, validator, carry/reset contract, or
arithmetic meaning to the same order label. The valid study therefore separates
common raw one-step output, common affine carry, common box carry, accurately
labelled native modes, and native practical tradeoffs. It never compares a
legacy-tightened Torch endpoint against another tool's raw endpoint.

Flow*'s stock Riccati miss was traced to a variable-leaf truncation contribution
present in the full evaluator but absent from the cached remainder-only replay.
The record/replay patch and the independent full-Picard revalidation both
restore containment. The fixed-order-2, `h=0.05` Riccati stress point can
reject its configured candidate remainder; that is a configuration rejection,
not a crash or an overall Flow* failure. No experiment overwrites a Flow*
remainder after `advance`.

The earlier adaptive Van der Pol collapsed endpoint miss is also closed.  The
audit compares stock upstream, the original and identical generated harnesses,
the variable-leaf patch, and the adaptive full-Picard fallback.  It localizes
the miss to collapsed endpoint restriction: Flow*'s native composed flowpipe
evaluated on `tau=[h,h]` contains every deterministic sample.  The raw endpoint
now carries the explicit hull delta as independent remainder.  Repair passed:
**True**; excluded from authoritative:
**False**.

## Repository provenance

| repository | SHA |
| --- | --- |
| diffreach | dd628eb443b517d6415de93e7035b4baef73963e |
| flowstar_audit | 2310c1ac55357d0b48af3b37495a82a3e10ea4ff |
| flowstar_original | b85a3211748cb77b736fe4ad42ee02d8d2b81148 |
| torch | 129b63322d7ec5e9617f54579a30ebdd6adc4c43 |
| torch_repaired_base | 9024a8a29bdc0ad668a7c0620bd53872f4313cc8 |

## Pushed study checkpoints included before the full run

| SHA | checkpoint |
| --- | --- |
| 3dd02ff | add common segment export and tool semantics |
| 6fe9120 | add Flowstar correctness-revalidated path |
| cddc921 | add controlled affine and box protocols |
| 08bebe5 | add native capability comparison regimes |
| ecc796d | add component and matched-basis ablations |
| ec82643 | add common polynomial defect diagnostics |
| 04708d3 | add Pareto analysis plots and final reporting |
| 94d93b3 | validate Flowstar large-step order-two candidates |
| f792a76 | validate Flowstar Van der Pol one-step candidates |
| 6cfea9f | soundly project nonlinear native reset endpoints |
| 50624d6 | cover every native multi-step configuration |
| 92ee79c | preserve every ordered common segment export |
| ef0a677 | deduplicate superseded common segment artifacts |
| 2cd48b5 | soundly project repeated Torch reset endpoints |
| a87708a | checkpoint recovered deep study implementation |
| 4194e34 | fix original Flowstar parity horizon collection |
| 0a6cd4b | document and reproduce Flowstar cache root cause |
| cf23002 | record passing Flowstar integrated smoke gates |
| c49570a | add explicit three-tool CIR v2 schema |
| f7621d8 | enforce matched basis and within-tool Pareto semantics |
| 8ddb829 | record passing integrated protocol smoke |
| 89456b8 | add range-only BERN feasibility and literature map |
| 3bf1e25 | add fail-closed final delivery workflow |
| 266bed4 | record interrupted authoritative run recovery |
| 9a60e74 | close adaptive Flowstar endpoint correctness gate |
| fab3141 | record checkpoint push infrastructure blocker |
| 7868274 | mark checkpoint push incident resolved |
| a781c70 | project CUDA Pareto endpoints before affine reset |
| 129b633 | close formal artifact provenance gates |
| b0d20d4 | separate tightened endpoints from raw reports |

## Interpretation contract

Pareto dominance is computed only within one tool, system, and absolute
evaluation time. Common affine and box carry control the propagated
representation, not the native local construction. Box reset discards
correlation, although recentering can still reduce a later measured width, so a
ratio below one is not “negative dependency loss.” Deterministic trajectory
sampling is a bug-finding sanity check and never a proof of containment.

## Basis availability

| tool | basis | status | reason |
| --- | --- | --- | --- |
| torch_common_engine | B1 | supported_experiment_adapter |  |
| torch_common_engine | B_DR | supported_experiment_adapter |  |
| torch_common_engine | B2 | supported_experiment_adapter |  |
| torch_common_engine | B3 | supported_experiment_adapter |  |
| diffreach | B1 | supported_native |  |
| diffreach | B_DR | supported_native |  |
| diffreach | B2 | capability_gap | no complete total-degree-2 native dictionary |
| diffreach | B3 | capability_gap | no quadratic state-cross dictionary with tau lift |
| flowstar | B1 | capability_gap | minimum legal fixed order is 2; no exact B1 selector |
| flowstar | B_DR | capability_gap | no exact restricted c/L/Lt dictionary selector |
| flowstar | B2 | supported_native |  |
| flowstar | B3 | capability_gap | order 3 is a strict cubic superset, not exact B3 |

## BERN decision

BERN is a range-only feasibility component, not a fourth reachability solver.
Its clean-room float64 prototype contains all
5 analytic cases and is stricter on
2 cancellation cases. This supports
further work only after a formally enclosed sparse roundoff backend; it does
not provide integration, Picard validation, truncation handling, endpoint
substitution, reset, or plant/controller composition.

## Detailed generated study

# Torch TM / DiffReach / Flow* deep comparative study

## Executive result

The study produces valid one-step, common-affine-carry, common-box-carry,
native-low-order, and native-practical comparisons.  It does **not** produce a
literal same-order winner, because the three order labels select different
monomial dictionaries, validators, reset contracts, and arithmetic backends.
The closest valid reconstruction of “first order” is the common affine carry
contract: every raw endpoint is projected to `x = c + A xi + I`, every removed
term is outward-ranged into a fresh independent interval, and every local
solver remains native.

Primary gates passed: **True**.
The collected tables contain 63813
native-validation checks, 17017 analytic
checks, 4095 exported point
containment checks,
2010 native/export
round-trip evaluations, 24054
non-proof nonlinear trajectory checks, and 32 explicitly
classified failure rows.

## Provenance and environments

| repository | SHA | path |
| --- | --- | --- |
| diffreach | dd628eb443b517d6415de93e7035b4baef73963e | /srv/local/shengenli/DiffReach |
| flowstar_audit | 2310c1ac55357d0b48af3b37495a82a3e10ea4ff | /srv/local/shengenli/flowstar_three_way_audit |
| flowstar_original | b85a3211748cb77b736fe4ad42ee02d8d2b81148 | /srv/local/shengenli/flowstar |
| torch | 129b63322d7ec5e9617f54579a30ebdd6adc4c43 | /srv/local/shengenli/torch_tm_flowpipe_three_tool_study |
| torch_repaired_base | 9024a8a29bdc0ad668a7c0620bd53872f4313cc8 | /srv/local/shengenli/torch_tm_flowpipe_three_way_repair |

- Torch: conda `py11`,
  Python 3.11.15, Torch
  2.5.1+cu121, CPU float64 study
  path; CUDA available:
  True.
- DiffReach: conda `diffreach312`,
  Python 3.12.13, JAX/JAXlib
  0.10.2/
  0.10.2, x64
  True.
- Flow*: `system-gcc-mpfr`, GCC
  gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0,
  MPFR 4.2.2,
  GMP 6.3.0.
- CPU: Intel(R) Xeon(R) Gold 6138 CPU @ 2.00GHz; batch size one.  Secondary
  accelerator availability and measurements are reported separately below.
- Frozen inputs unchanged:
  True.

## Flow* correction status

The exact cached-remainder defect is a missing variable-leaf truncation
interval: the full evaluator applies `ctrunc_normal` at a variable leaf, while
the cached remainder-only evaluator previously had no corresponding cached
entry.  The root-cause patch records and consumes this contribution.  A
separate full-Picard-revalidation variant atomically accepts a proposed
remainder only after regenerating the complete image and polynomial-difference
intervals.

Primary corrected/revalidated analytic rows:
64; analytic violations:
0; endpoint/tube
violations: 0;
export failures: 0.  Stock
analytic violations retained as evidence:
4.  The original generated
Van der Pol harness preserves the upstream schedule and the root-cause variant
reaches T=10: True
in 303 segments.  Corrected
refinement can legitimately choose a different adaptive schedule.  The
pre-repair collapsed adaptive endpoint has the independently retained
deterministic-trajectory failure:
True.  The endpoint-path audit classifies it
as `collapsed evaluate_time endpoint under-enclosure; native fixed-domain evaluation contains all deterministic samples`.  The authoritative raw
endpoint is now the explicit hull with Flow*'s native fixed-domain endpoint
evaluation; repair passed: True, excluded:
False.

Refinement/candidate control:

| mode | candidate | status | analytic violations | endpoint width |
| --- | --- | --- | --- | --- |
| refinement_disabled | 1e-06 | failed | 0 | n/a |
| stock_cached_refinement | 1e-06 | failed | 0 | n/a |
| full_picard_revalidated | 1e-06 | failed | 0 | n/a |
| root_cause_leaf_cache_patch | 1e-06 | failed | 0 | n/a |
| refinement_disabled | 0.0001 | success | 0 | 0.100126 |
| stock_cached_refinement | 0.0001 | success | 1 | 0.100125 |
| full_picard_revalidated | 0.0001 | success | 0 | 0.100126 |
| root_cause_leaf_cache_patch | 0.0001 | success | 0 | 0.100125 |
| refinement_disabled | 0.01 | success | 0 | 0.100167 |
| stock_cached_refinement | 0.01 | success | 1 | 0.100125 |
| full_picard_revalidated | 0.01 | success | 0 | 0.100167 |
| root_cause_leaf_cache_patch | 0.01 | success | 0 | 0.100125 |

## RQ1 — one-step local enclosure

Every row uses the same ODE, state order, initial box, `h`, and raw
tube/endpoint distinction.  Primary raw-endpoint maxima are:

| tool | variant | system | h | max width |
| --- | --- | --- | --- | --- |
| diffreach | upstream_affine_flag | coupled_quadratic | 0.0025 | 0.0401219 |
| diffreach | upstream_affine_flag | coupled_quadratic | 0.005 | 0.0402458 |
| diffreach | upstream_affine_flag | coupled_quadratic | 0.01 | 0.0404992 |
| diffreach | upstream_affine_flag | coupled_quadratic | 0.02 | 0.0410296 |
| diffreach | upstream_affine_flag | harmonic | 0.005 | 0.201005 |
| diffreach | upstream_affine_flag | harmonic | 0.01 | 0.20202 |
| diffreach | upstream_affine_flag | harmonic | 0.02 | 0.204082 |
| diffreach | upstream_affine_flag | harmonic | 0.05 | 0.210526 |
| diffreach | upstream_affine_flag | riccati | 0.005 | 0.100063 |
| diffreach | upstream_affine_flag | riccati | 0.01 | 0.100125 |
| diffreach | upstream_affine_flag | riccati | 0.02 | 0.100251 |
| diffreach | upstream_affine_flag | riccati | 0.05 | 0.100632 |
| diffreach | upstream_affine_flag | van_der_pol | 0.0025 | 0.300273 |
| diffreach | upstream_affine_flag | van_der_pol | 0.005 | 0.300591 |
| diffreach | upstream_affine_flag | van_der_pol | 0.01 | 0.301377 |
| diffreach | upstream_affine_flag | van_der_pol | 0.02 | 0.303604 |
| diffreach | upstream_restricted_quasi_quadratic | coupled_quadratic | 0.0025 | 0.0401212 |
| diffreach | upstream_restricted_quasi_quadratic | coupled_quadratic | 0.005 | 0.0402426 |
| diffreach | upstream_restricted_quasi_quadratic | coupled_quadratic | 0.01 | 0.0404864 |
| diffreach | upstream_restricted_quasi_quadratic | coupled_quadratic | 0.02 | 0.040978 |
| diffreach | upstream_restricted_quasi_quadratic | harmonic | 0.005 | 0.201003 |
| diffreach | upstream_restricted_quasi_quadratic | harmonic | 0.01 | 0.20201 |
| diffreach | upstream_restricted_quasi_quadratic | harmonic | 0.02 | 0.204041 |
| diffreach | upstream_restricted_quasi_quadratic | harmonic | 0.05 | 0.210263 |
| diffreach | upstream_restricted_quasi_quadratic | riccati | 0.005 | 0.100063 |
| diffreach | upstream_restricted_quasi_quadratic | riccati | 0.01 | 0.100125 |
| diffreach | upstream_restricted_quasi_quadratic | riccati | 0.02 | 0.100251 |
| diffreach | upstream_restricted_quasi_quadratic | riccati | 0.05 | 0.100631 |
| diffreach | upstream_restricted_quasi_quadratic | van_der_pol | 0.0025 | 0.300257 |
| diffreach | upstream_restricted_quasi_quadratic | van_der_pol | 0.005 | 0.30053 |
| diffreach | upstream_restricted_quasi_quadratic | van_der_pol | 0.01 | 0.301122 |
| diffreach | upstream_restricted_quasi_quadratic | van_der_pol | 0.02 | 0.302514 |
| flowstar | flowstar_root_cause_patch | coupled_quadratic | 0.0025 | 0.0401221 |
| flowstar | flowstar_root_cause_patch | coupled_quadratic | 0.005 | 0.0402446 |
| flowstar | flowstar_root_cause_patch | coupled_quadratic | 0.01 | 0.0404903 |
| flowstar | flowstar_root_cause_patch | coupled_quadratic | 0.02 | 0.0409905 |
| flowstar | flowstar_root_cause_patch | harmonic | 0.005 | 0.201005 |
| flowstar | flowstar_root_cause_patch | harmonic | 0.01 | 0.20202 |
| flowstar | flowstar_root_cause_patch | harmonic | 0.02 | 0.204082 |
| flowstar | flowstar_root_cause_patch | harmonic | 0.05 | 0.210527 |
| flowstar | flowstar_root_cause_patch | riccati | 0.005 | 0.100075 |
| flowstar | flowstar_root_cause_patch | riccati | 0.01 | 0.10015 |
| flowstar | flowstar_root_cause_patch | riccati | 0.02 | 0.100301 |
| flowstar | flowstar_root_cause_patch | riccati | 0.05 | 0.100755 |
| flowstar | flowstar_root_cause_patch | van_der_pol | 0.0025 | 0.300264 |
| flowstar | flowstar_root_cause_patch | van_der_pol | 0.005 | 0.300559 |
| flowstar | flowstar_root_cause_patch | van_der_pol | 0.01 | 0.301248 |
| flowstar | flowstar_root_cause_patch | van_der_pol | 0.02 | 0.303094 |
| torch_tm_flowpipe | complete_total_degree_1 | coupled_quadratic | 0.0025 | 0.0407325 |
| torch_tm_flowpipe | complete_total_degree_1 | coupled_quadratic | 0.005 | 0.041465 |
| torch_tm_flowpipe | complete_total_degree_1 | coupled_quadratic | 0.01 | 0.04293 |
| torch_tm_flowpipe | complete_total_degree_1 | coupled_quadratic | 0.02 | 0.04586 |
| torch_tm_flowpipe | complete_total_degree_1 | harmonic | 0.005 | 0.20125 |
| torch_tm_flowpipe | complete_total_degree_1 | harmonic | 0.01 | 0.2025 |
| torch_tm_flowpipe | complete_total_degree_1 | harmonic | 0.02 | 0.205 |
| torch_tm_flowpipe | complete_total_degree_1 | harmonic | 0.05 | 0.2125 |
| torch_tm_flowpipe | complete_total_degree_1 | riccati | 0.005 | 0.100063 |
| torch_tm_flowpipe | complete_total_degree_1 | riccati | 0.01 | 0.100125 |
| torch_tm_flowpipe | complete_total_degree_1 | riccati | 0.02 | 0.10025 |
| torch_tm_flowpipe | complete_total_degree_1 | riccati | 0.05 | 0.100625 |
| torch_tm_flowpipe | complete_total_degree_1 | van_der_pol | 0.0025 | 0.307656 |
| torch_tm_flowpipe | complete_total_degree_1 | van_der_pol | 0.005 | 0.315313 |
| torch_tm_flowpipe | complete_total_degree_1 | van_der_pol | 0.01 | 0.330625 |
| torch_tm_flowpipe | complete_total_degree_1 | van_der_pol | 0.02 | 0.36125 |
| torch_tm_flowpipe | complete_total_degree_2 | coupled_quadratic | 0.0025 | 0.0400827 |
| torch_tm_flowpipe | complete_total_degree_2 | coupled_quadratic | 0.005 | 0.0401658 |
| torch_tm_flowpipe | complete_total_degree_2 | coupled_quadratic | 0.01 | 0.0403333 |
| torch_tm_flowpipe | complete_total_degree_2 | coupled_quadratic | 0.02 | 0.0406732 |
| torch_tm_flowpipe | complete_total_degree_2 | harmonic | 0.005 | 0.201003 |
| torch_tm_flowpipe | complete_total_degree_2 | harmonic | 0.01 | 0.202013 |
| torch_tm_flowpipe | complete_total_degree_2 | harmonic | 0.02 | 0.20405 |
| torch_tm_flowpipe | complete_total_degree_2 | harmonic | 0.05 | 0.210313 |
| torch_tm_flowpipe | complete_total_degree_2 | riccati | 0.005 | 0.100063 |
| torch_tm_flowpipe | complete_total_degree_2 | riccati | 0.01 | 0.100125 |
| torch_tm_flowpipe | complete_total_degree_2 | riccati | 0.02 | 0.10025 |
| torch_tm_flowpipe | complete_total_degree_2 | riccati | 0.05 | 0.100625 |
| torch_tm_flowpipe | complete_total_degree_2 | van_der_pol | 0.0025 | 0.300312 |
| torch_tm_flowpipe | complete_total_degree_2 | van_der_pol | 0.005 | 0.300752 |
| torch_tm_flowpipe | complete_total_degree_2 | van_der_pol | 0.01 | 0.302026 |
| torch_tm_flowpipe | complete_total_degree_2 | van_der_pol | 0.02 | 0.306251 |

These rows are not relatively ranked because the native local bases are not
exactly matched.  This is a local-construction result, not a long-horizon
wrapping claim.  Flow* exposes a complete higher-order expansion
with MPFR intervals; DiffReach's restricted quasi-quadratic form stores
constant/linear plus local-time cross structure; Torch's complete basis exposes
more local monomials as its order increases.  Raw and legacy-tightened Torch
endpoints remain separate everywhere.

## RQ2 — common affine carry

The requested-final-time rows are listed below.  They are valid controlled
carry observations, but they are not relatively ranked because each native
local solver still uses a different construction basis, range evaluation, and
validator.  Carried degree and endpoint projection are controlled.  A failed
short prefix is never compared against a solver that reached the requested
final time.

| tool | variant | system | h | time | max width |
| --- | --- | --- | --- | --- | --- |
| diffreach_experimental_strict_affine |  | coupled_quadratic | 0.005 | 0.25 | 0.0547715 |
| diffreach_experimental_strict_affine |  | coupled_quadratic | 0.01 | 0.25 | 0.0556381 |
| diffreach_experimental_strict_affine |  | harmonic | 0.01 | 4 | 10.9211 |
| diffreach_experimental_strict_affine |  | riccati | 0.01 | 1 | 0.114071 |
| flowstar | flowstar_root_cause_patch | coupled_quadratic | 0.005 | 0.25 | 0.0430496 |
| flowstar | flowstar_root_cause_patch | coupled_quadratic | 0.01 | 0.25 | 0.0431334 |
| flowstar | flowstar_root_cause_patch | harmonic | 0.01 | 4 | 0.42945 |
| flowstar | flowstar_root_cause_patch | riccati | 0.01 | 1 | 0.113904 |
| torch_tm_flowpipe | complete_total_degree_1 | coupled_quadratic | 0.005 | 0.25 | 127.975 |
| torch_tm_flowpipe | complete_total_degree_1 | coupled_quadratic | 0.01 | 0.25 | 0.848351 |
| torch_tm_flowpipe | complete_total_degree_1 | harmonic | 0.01 | 4 | 2.9605e+38 |
| torch_tm_flowpipe | complete_total_degree_1 | riccati | 0.01 | 0.48 | 68.1299 |
| torch_tm_flowpipe | complete_total_degree_1 | van_der_pol | 0.005 | 0.15 | 125.087 |
| torch_tm_flowpipe | complete_total_degree_1 | van_der_pol | 0.01 | 0.23 | 82.6159 |

Configurations that did not reach their requested common time remain useful
successful-horizon evidence but are unavailable for the final-time ranking:

| tool | variant | protocol | system | h | last valid time | max width |
| --- | --- | --- | --- | --- | --- | --- |
| diffreach_experimental_strict_affine |  | common_affine_carry | van_der_pol | 0.005 | 0.625 | 8.16479 |
| diffreach_experimental_strict_affine |  | common_affine_carry | van_der_pol | 0.01 | 0.55 | 5.20896 |
| flowstar | flowstar_root_cause_patch | common_affine_carry | van_der_pol | 0.005 | 0.13 | 0.391469 |
| flowstar | flowstar_root_cause_patch | common_affine_carry | van_der_pol | 0.01 | 0 | 0.3 |
| flowstar | flowstar_root_cause_patch | common_box_carry | harmonic | 0.01 | 2.29 | 1.99789 |
| flowstar | flowstar_root_cause_patch | common_box_carry | van_der_pol | 0.005 | 0.1 | 0.340685 |
| flowstar | flowstar_root_cause_patch | common_box_carry | van_der_pol | 0.01 | 0 | 0.3 |

## RQ3 — common box carry and native low order

Box carry removes generator correlations.  The measured final width ratios are:

| tool | system | h | time | affine | box | box/affine |
| --- | --- | --- | --- | --- | --- | --- |
| flowstar | coupled_quadratic | 0.005 | 0.25 | 0.0430496 | 0.0430491 | 0.999987 |
| flowstar | coupled_quadratic | 0.01 | 0.25 | 0.0431334 | 0.0431326 | 0.999981 |
| flowstar | riccati | 0.01 | 1 | 0.113904 | 0.113982 | 1.00068 |
| torch_tm_flowpipe | coupled_quadratic | 0.005 | 0.25 | 127.975 | 0.0588145 | 4.5958e-04 |
| torch_tm_flowpipe | coupled_quadratic | 0.01 | 0.25 | 0.848351 | 0.0596531 | 0.0703165 |
| torch_tm_flowpipe | harmonic | 0.01 | 4 | 2.9605e+38 | 28.7768 | 9.7202e-38 |

Primary native low-order rows are deliberately labelled with their actual
bases.  Supplemental Torch tightened endpoints are excluded from this
cross-tool raw/native table:

| tool | variant | system | h | time | max width |
| --- | --- | --- | --- | --- | --- |
| diffreach | affine_flag | coupled_quadratic | 0.005 | 0.25 | 0.0540625 |
| diffreach | affine_flag | coupled_quadratic | 0.01 | 0.25 | 0.0542317 |
| diffreach | affine_flag | harmonic | 0.01 | 4 | 11.1417 |
| diffreach | affine_flag | riccati | 0.01 | 1 | 0.113899 |
| diffreach | affine_flag | van_der_pol | 0.005 | 0.615 | 7.93661 |
| diffreach | affine_flag | van_der_pol | 0.01 | 0.53 | 4.97027 |
| diffreach | quasi_window10_round3 | coupled_quadratic | 0.005 | 0.25 | 0.0429779 |
| diffreach | quasi_window10_round3 | coupled_quadratic | 0.01 | 0.25 | 0.0429902 |
| diffreach | quasi_window10_round3 | harmonic | 0.01 | 4 | 0.34743 |
| diffreach | quasi_window10_round3 | riccati | 0.01 | 1 | 0.113895 |
| diffreach | quasi_window10_round3 | van_der_pol | 0.005 | 1 | 0.154427 |
| diffreach | quasi_window10_round3 | van_der_pol | 0.01 | 1 | 0.174849 |
| diffreach | quasi_window1_round1 | coupled_quadratic | 0.005 | 0.25 | 0.0949019 |
| diffreach | quasi_window1_round1 | coupled_quadratic | 0.01 | 0.25 | 0.0959395 |
| diffreach | quasi_window1_round1 | harmonic | 0.01 | 4 | 11.024 |
| diffreach | quasi_window1_round1 | riccati | 0.01 | 1 | 0.189025 |
| diffreach | quasi_window1_round1 | van_der_pol | 0.005 | 1 | 3.76135 |
| diffreach | quasi_window1_round1 | van_der_pol | 0.01 | 0.95 | 3.71086 |
| diffreach | restricted_quasiquadratic | coupled_quadratic | 0.005 | 0.25 | 0.0429776 |
| diffreach | restricted_quasiquadratic | coupled_quadratic | 0.01 | 0.25 | 0.0429889 |
| diffreach | restricted_quasiquadratic | harmonic | 0.01 | 4 | 0.303186 |
| diffreach | restricted_quasiquadratic | riccati | 0.01 | 1 | 0.113895 |
| diffreach | restricted_quasiquadratic | van_der_pol | 0.005 | 1 | 0.140647 |
| diffreach | restricted_quasiquadratic | van_der_pol | 0.01 | 1 | 0.151332 |
| flowstar | root_cause_fixed_order_2 | coupled_quadratic | 0.005 | 0.25 | 0.0429918 |
| flowstar | root_cause_fixed_order_2 | coupled_quadratic | 0.01 | 0.25 | 0.0430177 |
| flowstar | root_cause_fixed_order_2 | harmonic | 0.01 | 4 | 0.426398 |
| flowstar | root_cause_fixed_order_2 | riccati | 0.01 | 1 | 0.113904 |
| flowstar | root_cause_fixed_order_2 | van_der_pol | 0.005 | 1 | 0.179374 |
| flowstar | root_cause_fixed_order_2 | van_der_pol | 0.01 | 1 | 0.23915 |
| torch_tm_flowpipe | order1_range_only | coupled_quadratic | 0.005 | 0.25 | 0.115255 |
| torch_tm_flowpipe | order1_range_only | coupled_quadratic | 0.01 | 0.25 | 0.115211 |
| torch_tm_flowpipe | order1_range_only | harmonic | 0.01 | 4 | 28.7768 |
| torch_tm_flowpipe | order1_range_only | riccati | 0.01 | 1 | 0.114066 |
| torch_tm_flowpipe | order1_range_only | van_der_pol | 0.005 | 0.49 | 385.941 |
| torch_tm_flowpipe | order1_range_only | van_der_pol | 0.01 | 0.48 | 241.814 |
| torch_tm_flowpipe | order1_raw_dependency | coupled_quadratic | 0.005 | 0.22 | 137.543 |
| torch_tm_flowpipe | order1_raw_dependency | coupled_quadratic | 0.01 | 0.25 | 3.79488 |
| torch_tm_flowpipe | order1_raw_dependency | harmonic | 0.01 | 4 | 2.9605e+38 |
| torch_tm_flowpipe | order1_raw_dependency | riccati | 0.01 | 0.49 | 68.1127 |
| torch_tm_flowpipe | order1_raw_dependency | van_der_pol | 0.005 | 0.11 | 73.1903 |
| torch_tm_flowpipe | order1_raw_dependency | van_der_pol | 0.01 | 0.16 | 38.5151 |

For preservation and Torch-internal diagnosis only, the tightened endpoints
are listed separately below.  They are not juxtaposed with or ranked against
another tool's raw/native endpoint:

| tool | variant | system | h | time | max width |
| --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | order1_legacy_tightened | coupled_quadratic | 0.005 | 0.25 | 0.0675591 |
| torch_tm_flowpipe | order1_legacy_tightened | coupled_quadratic | 0.01 | 0.25 | 0.0688671 |
| torch_tm_flowpipe | order1_legacy_tightened | harmonic | 0.01 | 4 | 30.2297 |
| torch_tm_flowpipe | order1_legacy_tightened | riccati | 0.01 | 1 | 0.114289 |
| torch_tm_flowpipe | order1_legacy_tightened | van_der_pol | 0.005 | 0.48 | 249.696 |
| torch_tm_flowpipe | order1_legacy_tightened | van_der_pol | 0.01 | 0.46 | 113.176 |

DiffReach's affine flag and restricted quasi-quadratic mode are not the same
basis.  The latter can preserve `tau^2` and `tau*xi` structure before endpoint
evaluation and symbolic reset; it still omits general state-state and cubic
families.  Whether that helps depends on whether the missing state-state terms
or the retained time-state dependence dominates.  Torch dependency carry can
deteriorate because old generators and interval remainders remain correlated
through every new polynomial operation; normalized affine/QR resets exchange
some local polynomial detail for much better conditioning.  Flow* benefits
when complete higher-order terms, normalized composition, symbolic remainder,
or adaptive steps prevent the same information from being repeatedly ranged.

## RQ4 — native practical tradeoffs and within-tool Pareto frontiers

Width/runtime dominance was computed only within one tool at identical system
and absolute time.  Cross-tool native rows are not relatively ranked.
24 selected configuration/system rows have ten
full-configuration repetitions.  Within-tool nondominated rows:

| tool | variant | system | time | width | horizon | steady s | repetitions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | order1_range_only | riccati | 1 | 0.114066 | 1 | 1.30803 | 1 |
| torch_tm_flowpipe | order4_qr_reset | riccati | 1 | 0.113962 | 1 | 12.7594 | 1 |
| torch_tm_flowpipe | order6_affine_reset | riccati | 1 | 0.113961 | 1 | 29.5615 | 1 |
| torch_tm_flowpipe | order1_raw_dependency | harmonic | 4 | 2.9605e+38 | 4 | 10.0671 | 1 |
| torch_tm_flowpipe | order1_range_only | harmonic | 4 | 28.7768 | 4 | 10.1752 | 1 |
| torch_tm_flowpipe | order4_qr_reset | harmonic | 4 | 0.282089 | 4 | 44.1665 | 1 |
| torch_tm_flowpipe | order6_affine_reset | harmonic | 4 | 10.4942 | 4 | 43.2684 | 1 |
| torch_tm_flowpipe | order1_raw_dependency | coupled_quadratic | 0.25 | 3.79488 | 0.25 | 0.959531 | 1 |
| torch_tm_flowpipe | order1_range_only | coupled_quadratic | 0.25 | 0.115211 | 0.25 | 0.964526 | 1 |
| torch_tm_flowpipe | order4_affine_reset | coupled_quadratic | 0.25 | 0.0429663 | 0.25 | 14.3484 | 1 |
| torch_tm_flowpipe | order6_affine_reset | coupled_quadratic | 0.25 | 0.0429663 | 0.25 | 43.1093 | 1 |
| torch_tm_flowpipe | order2_affine_reset | van_der_pol | 1 | 1.6188 | 1 | 36.1624 | 1 |
| torch_tm_flowpipe | order2_affine_reset | van_der_pol | 1 | 1.94055 | 1 | 18.4304 | 1 |
| torch_tm_flowpipe | order4_affine_reset | van_der_pol | 1 | 0.911288 | 1 | 73.9007 | 1 |
| torch_tm_flowpipe | order4_qr_reset | van_der_pol | 1 | 0.18656 | 1 | 85.0001 | 1 |
| diffreach | quasi_window1_round1 | harmonic | 4 | 11.024 | 4 | 0.00997088 | 1 |
| diffreach | quasi_window10_round3 | harmonic | 4 | 0.34743 | 4 | 0.0130273 | 1 |
| diffreach | quasi_window10_round3 | coupled_quadratic | 0.25 | 0.0429779 | 0.25 | 0.0033193 | 1 |
| diffreach | affine_flag | coupled_quadratic | 0.25 | 0.0542317 | 0.25 | 0.00159874 | 1 |
| diffreach | quasi_window1_round1 | coupled_quadratic | 0.25 | 0.0959395 | 0.25 | 0.00118503 | 1 |
| diffreach | restricted_quasiquadratic | van_der_pol | 1 | 0.140647 | 1 | 0.0117766 | 1 |
| diffreach | quasi_window10_round3 | van_der_pol | 1 | 0.174849 | 1 | 0.00436561 | 1 |
| flowstar | root_cause_fixed_order_2 | riccati | 1 | 0.113904 | 1 | 0.01835 | 1 |
| flowstar | root_cause_fixed_order_2 | harmonic | 4 | 0.426398 | 4 | 0.0909423 | 1 |
| flowstar | root_cause_fixed_order_6 | harmonic | 4 | 0.282089 | 4 | 0.243677 | 1 |
| flowstar | root_cause_fixed_order_2 | coupled_quadratic | 0.25 | 0.0429918 | 0.25 | 0.0228812 | 1 |
| flowstar | root_cause_fixed_order_3 | coupled_quadratic | 0.25 | 0.042969 | 0.25 | 0.0945063 | 1 |
| flowstar | root_cause_fixed_order_6 | coupled_quadratic | 0.25 | 0.0429677 | 0.25 | 1.13996 | 1 |
| flowstar | root_cause_fixed_order_2 | coupled_quadratic | 0.25 | 0.0430177 | 0.25 | 0.0132682 | 1 |
| flowstar | root_cause_fixed_order_3 | coupled_quadratic | 0.25 | 0.0429705 | 0.25 | 0.0478553 | 1 |
| flowstar | root_cause_fixed_order_6 | coupled_quadratic | 0.25 | 0.0429677 | 0.25 | 0.579965 | 1 |
| flowstar | root_cause_fixed_order_2 | van_der_pol | 1 | 0.179374 | 1 | 0.0871194 | 1 |
| flowstar | root_cause_fixed_order_3 | van_der_pol | 1 | 0.116934 | 1 | 0.321783 | 1 |
| flowstar | root_cause_fixed_order_6 | van_der_pol | 1 | 0.112557 | 1 | 8.8882 | 1 |
| flowstar | root_cause_fixed_order_2 | van_der_pol | 1 | 0.23915 | 1 | 0.0459363 | 1 |
| flowstar | root_cause_fixed_order_3 | van_der_pol | 1 | 0.121515 | 1 | 0.16295 | 1 |
| flowstar | root_cause_fixed_order_6 | van_der_pol | 1 | 0.11256 | 1 | 4.57606 | 1 |
| flowstar | adaptive_order4_symbolic100 | van_der_pol | 10 | 0.520954 | 10 | 1.51036 | 1 |
| torch_tm_flowpipe | order2_affine_reset_selected | harmonic | 4 | 10.9731 | 4 | 12.7018 | 10 |
| torch_tm_flowpipe | order4_affine_reset_selected | harmonic | 4 | 10.4942 | 4 | 23.083 | 10 |
| torch_tm_flowpipe | order2_affine_reset_selected | coupled_quadratic | 0.25 | 0.0430271 | 0.25 | 6.14871 | 10 |
| torch_tm_flowpipe | order2_affine_reset_selected | coupled_quadratic | 0.25 | 0.0430391 | 0.25 | 3.07957 | 10 |
| diffreach | restricted_quasi_window100_round5_selected | riccati | 1 | 0.113895 | 1 | 0.00171849 | 10 |
| diffreach | restricted_quasi_window100_round5_selected | harmonic | 4 | 0.303186 | 4 | 0.0150983 | 10 |
| diffreach | restricted_quasi_window100_round5_selected | coupled_quadratic | 0.25 | 0.0429776 | 0.25 | 0.00359141 | 10 |
| diffreach | restricted_quasi_window100_round5_selected | coupled_quadratic | 0.25 | 0.0429889 | 0.25 | 0.00210494 | 10 |
| diffreach | restricted_quasi_window100_round5_selected | van_der_pol | 1 | 0.151332 | 1 | 0.00606031 | 10 |
| flowstar | root_cause_order4_selected | harmonic | 4 | 0.282091 | 4 | 0.115274 | 10 |
| flowstar | root_cause_order4_selected | coupled_quadratic | 0.25 | 0.0429677 | 0.25 | 0.252625 | 10 |
| flowstar | root_cause_order4_selected | coupled_quadratic | 0.25 | 0.0429677 | 0.25 | 0.125071 | 10 |
| flowstar | root_cause_order4_selected | van_der_pol | 1 | 0.112822 | 1 | 1.10728 | 10 |
| flowstar | root_cause_order4_selected | van_der_pol | 1 | 0.11324 | 1 | 0.557119 | 10 |

No cross-tool winner follows from these rows.  A width/runtime point at T=1 is
not ranked against Flow*'s adaptive T=10 point.  Compile/JIT/build costs remain
separate from steady full-horizon execution, and backend throughput is not
presented as pure algorithmic speed.  Any configuration with a deterministic
trajectory sanity failure has `primary_numerical_eligible=false` and is not a
frontier candidate.

## RQ5 — component and matched-basis attribution

The component table contains 828 rows.  It separates polynomial
range, exposed independent remainder, structured remainder where available,
and residual dependency/reset inflation.  The strongest causal controls are
within-tool: changing only carry/reset in Torch, only affine/quasi and
window/refinement settings in DiffReach, and only order/adaptation/symbolic
remainder/refinement in Flow*.

The one-engine matched-basis result is:

| basis | h | max width | max independent remainder | discard records |
| --- | --- | --- | --- | --- |
| B1 | 0.0025 | 0.0401524 | 1.5242e-04 | 19 |
| B1 | 0.005 | 0.0403072 | 3.0719e-04 | 19 |
| B1 | 0.01 | 0.0406239 | 6.2385e-04 | 19 |
| B1 | 0.02 | 0.0412862 | 0.00128619 | 19 |
| B2 | 0.0025 | 0.0400325 | 2.5345e-06 | 19 |
| B2 | 0.005 | 0.0400651 | 5.1387e-06 | 19 |
| B2 | 0.01 | 0.0401306 | 1.0559e-05 | 19 |
| B2 | 0.02 | 0.0402623 | 2.2352e-05 | 19 |
| B3 | 0.0025 | 0.040032 | 1.8168e-07 | 15 |
| B3 | 0.005 | 0.0400641 | 7.3117e-07 | 15 |
| B3 | 0.01 | 0.0401285 | 2.9603e-06 | 15 |
| B3 | 0.02 | 0.0402581 | 1.2127e-05 | 15 |
| B_DR | 0.0025 | 0.0400325 | 2.5345e-06 | 19 |
| B_DR | 0.005 | 0.0400651 | 5.1387e-06 | 19 |
| B_DR | 0.01 | 0.0401306 | 1.0559e-05 | 19 |
| B_DR | 0.02 | 0.0402623 | 2.2352e-05 | 19 |

Exact cross-tool basis capability is:

| tool | basis | status | mapping | reason |
| --- | --- | --- | --- | --- |
| torch_common_engine | B1 | supported_experiment_adapter | sound finite-dictionary projection |  |
| torch_common_engine | B_DR | supported_experiment_adapter | sound finite-dictionary projection |  |
| torch_common_engine | B2 | supported_experiment_adapter | sound finite-dictionary projection |  |
| torch_common_engine | B3 | supported_experiment_adapter | sound quadratic-dependency/time-lift projection |  |
| diffreach | B1 | supported_native | TRUNCATE_TO_AFFINE=true |  |
| diffreach | B_DR | supported_native | c/L/Lt restricted quasi-quadratic dictionary |  |
| diffreach | B2 | capability_gap | unavailable | no complete total-degree-2 native dictionary |
| diffreach | B3 | capability_gap | unavailable | no quadratic state-cross dictionary with tau lift |
| flowstar | B1 | capability_gap | unavailable | minimum legal fixed order is 2; no exact B1 selector |
| flowstar | B_DR | capability_gap | unavailable | no exact restricted c/L/Lt dictionary selector |
| flowstar | B2 | supported_native | fixed complete order 2 |  |
| flowstar | B3 | capability_gap | unavailable | order 3 is a strict cubic superset, not exact B3 |

All four use one order-3 arithmetic ceiling, two Picard iterations, validator,
range backend, dtype, step, initial set, and reset.  B3 is not a general cubic
basis: it adds the one-local-time lift of quadratic state dependency, including
`tau*xi_i*xi_j`, while excluding cubic state terms.  The coupled quadratic
activates that cross-term family.  Thus B1-to-B_DR/B2/B3
changes are attributable to the retained dictionary inside one implementation,
not JAX versus Torch versus C++.

## Common defect and certificates

The shared CPU implementation differentiates and composes exported sparse
polynomials, outward-ranges the defect, bounds the Jacobian on the native tube,
and reports a Gronwall comparison radius separately from the native remainder.
Median infinity-norm defect bounds by tool are:
diffreach: 0.0260566, flowstar: 5.0000e-04, torch_tm_flowpipe: 0.1.
Tiny Riccati and coupled-polynomial identities use exact rational unit tests.
The common radius is diagnostic; it does not erase the numerical distinction
between Flow* MPFR intervals and the floating-point enclosure candidates from
Torch/DiffReach.

## Runtime decomposition

| tool | variant | system | repetitions | build/JIT s | median full s | min s | max s | memory KiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torch_tm_flowpipe | order2_affine_reset_selected | riccati | 10 | 0 | 3.63884 | 3.63222 | 3.64449 | 3.9650e+05 |
| torch_tm_flowpipe | order4_affine_reset_selected | riccati | 10 | 0 | 18.5627 | 11.1952 | 19.1347 | 3.9650e+05 |
| torch_tm_flowpipe | order2_affine_reset_selected | harmonic | 10 | 0 | 12.7018 | 12.6768 | 13.0376 | 3.9650e+05 |
| torch_tm_flowpipe | order4_affine_reset_selected | harmonic | 10 | 0 | 23.083 | 23.0328 | 23.1343 | 3.9650e+05 |
| torch_tm_flowpipe | order2_affine_reset_selected | coupled_quadratic | 10 | 0 | 6.14871 | 6.13964 | 7.20609 | 3.9650e+05 |
| torch_tm_flowpipe | order4_affine_reset_selected | coupled_quadratic | 10 | 0 | 41.3882 | 25.3217 | 41.4785 | 3.9650e+05 |
| torch_tm_flowpipe | order2_affine_reset_selected | coupled_quadratic | 10 | 0 | 3.07957 | 3.07463 | 3.09086 | 3.9650e+05 |
| torch_tm_flowpipe | order4_affine_reset_selected | coupled_quadratic | 10 | 0 | 17.3487 | 12.666 | 20.7854 | 3.9650e+05 |
| torch_tm_flowpipe | order2_affine_reset_selected | van_der_pol | 10 | 0 | 52.5547 | 31.8875 | 52.6328 | 3.9650e+05 |
| torch_tm_flowpipe | order4_affine_reset_selected | van_der_pol | 10 | 0 | 164.89 | 109.831 | 178.29 | 3.9650e+05 |
| torch_tm_flowpipe | order2_affine_reset_selected | van_der_pol | 10 | 0 | 26.3473 | 26.2914 | 26.3593 | 3.9650e+05 |
| torch_tm_flowpipe | order4_affine_reset_selected | van_der_pol | 10 | 0 | 103.199 | 80.3231 | 105.856 | 3.9650e+05 |
| diffreach | restricted_quasi_window100_round5_selected | riccati | 10 | 0.892665 | 0.00171849 | 0.00162839 | 0.00176778 | 4.3169e+05 |
| diffreach | restricted_quasi_window100_round5_selected | harmonic | 10 | 1.45548 | 0.0150983 | 0.0149623 | 0.0152998 | 5.2684e+05 |
| diffreach | restricted_quasi_window100_round5_selected | coupled_quadratic | 10 | 3.23863 | 0.00359141 | 0.00314788 | 0.00402944 | 6.2162e+05 |
| diffreach | restricted_quasi_window100_round5_selected | coupled_quadratic | 10 | 3.26751 | 0.00210494 | 0.00180193 | 0.00241235 | 6.5749e+05 |
| diffreach | restricted_quasi_window100_round5_selected | van_der_pol | 10 | 3.46132 | 0.0131082 | 0.0128069 | 0.0136518 | 6.9298e+05 |
| diffreach | restricted_quasi_window100_round5_selected | van_der_pol | 10 | 3.49124 | 0.00606031 | 0.00596008 | 0.00781845 | 7.2704e+05 |
| flowstar | root_cause_order4_selected | riccati | 10 | 1.70055 | 0.0614244 | 0.0606429 | 0.0628469 | n/a |
| flowstar | root_cause_order4_selected | harmonic | 10 | 1.70896 | 0.115274 | 0.114469 | 0.120858 | n/a |
| flowstar | root_cause_order4_selected | coupled_quadratic | 10 | 1.71217 | 0.252625 | 0.24996 | 0.256386 | n/a |
| flowstar | root_cause_order4_selected | coupled_quadratic | 10 | 1.71084 | 0.125071 | 0.124116 | 0.133501 | n/a |
| flowstar | root_cause_order4_selected | van_der_pol | 10 | 1.7142 | 1.10728 | 1.10082 | 1.11377 | n/a |
| flowstar | root_cause_order4_selected | van_der_pol | 10 | 1.75713 | 0.557119 | 0.553954 | 0.564788 | n/a |

Flow* build and process execution, DiffReach JIT and after-JIT execution, and
Torch orchestration/arithmetic/validation are distinct categories.  JAX
fusion/JIT and C++ compilation are backend effects; term count, Picard
refinement, range operations, and resets are algorithmic effects.

## Secondary native acceleration

These rows compare implementation/hardware throughput, not algorithmic
fairness.  The CPU and accelerator rows for a given tool use the same selected
full configuration on the cross-term-active coupled quadratic benchmark.

| tool | backend | status | system | h | repetitions | median full s | speedup vs same-tool CPU | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| diffreach | jax_cpu | available | coupled_quadratic | 0.005 | 10 | 0.00359141 | 1 |  |
| flowstar | flowstar_cpu | available | coupled_quadratic | 0.005 | 10 | 0.252625 | 1 |  |
| torch_tm_flowpipe | torch_cpu | available | coupled_quadratic | 0.005 | 10 | 41.3882 | 1 |  |
| torch_tm_flowpipe | torch_cuda | available | coupled_quadratic | 0.005 | 10 | 96.1866 | 0.43029 |  |
| diffreach | jax_cuda | unavailable | coupled_quadratic | 0.005 | 0 | n/a | n/a | installed JAX/JAXlib exposes no GPU device |

## RQ6 — BERN/IBF feasibility and literature position

BERN is evaluated as a range-query component, not a fourth reachability tool.
The clean-room CPU prototype preserves the polynomial's cross terms and applies
the Bernstein coefficient hull to five analytic terms drawn from cancellation,
the coupled quadratic system, and Van der Pol:

| case | purpose | current width | Bernstein width | ratio | exact contained | median Bernstein s |
| --- | --- | --- | --- | --- | --- | --- |
| difference_squared | cancellation_and_cross_term | 4 | 1.5 | 0.375 | True | 9.7844e-05 |
| difference_fourth | higher_order_cancellation | 16 | 1.25 | 0.078125 | True | 1.4781e-04 |
| coupled_quadratic_x1_x2 | study_cross_term | 0.012 | 0.012 | 1 | True | 5.3553e-05 |
| coupled_quadratic_x1_squared_minus_x2 | study_rhs_range | 0.048 | 0.048 | 1 | True | 7.3977e-05 |
| van_der_pol_x1_squared_x2 | study_nonlinear_cross_term | 1.9585 | 1.9585 | 1 | True | 6.1502e-05 |

All analytic ranges were contained:
True; the
Bernstein candidate was strictly tighter in
2 cases.  This is
exact-arithmetic Bernstein evidence plus a conservative float64 allowance, not
a complete roundoff proof.  The external CUDA code is not placed in the primary
correctness path, and no singleton GPU timing is reported.  The evidence-backed
decision is to continue only toward a formally enclosed sparse range backend.
BERN does not supply local-time integration, truncation-to-remainder, Picard
validation, endpoint substitution, or multi-step reset.  NN controller methods
remain indirect because this benchmark is plant-only.  See
`BERN_FEASIBILITY.md` and `LITERATURE_MAP.md`.

## Direct answers to the eleven final questions

1. **Why same order is impossible.** Torch order 1 is a complete affine
   total-degree cap; DiffReach's two low-order flags retain different
   time-cross dictionaries; Flow*'s minimum legal fixed order is 2.  Their
   resets, remainders, and validators also differ.
2. **Closest valid first-order experiment.** Common affine carry is the closest:
   native local solve, raw endpoint, then one sound `c + A xi + I` carry
   projection.
3. **Widths under affine carry.** The per-system rows are reported without a
   cross-tool winner because local basis/construction, range bounding, and
   validator remain unmatched after controlling the carried representation.
4. **Box-carry control.** The table above reports the exact measured ratios.
   The box operation discards correlation, but width need not increase
   monotonically because re-normalization can improve later interval
   conditioning; values above one are observed wrapping inflation, while
   values below one are not “negative dependency loss.”
5. **DiffReach low-order terms.** `tau^2` and `tau*xi` can reduce local-time
   truncation relative to a purely affine form, while missing general
   state-state/cubic terms can dominate on coupled or Van der Pol dynamics.
6. **When Flow* helps.** Complete higher order helps when nonlinear terms remain
   useful through composition; adaptive step and symbolic remainder help on
   longer nonlinear horizons.  The corrected Van der Pol path reaches T=10.
   Its former collapsed `evaluate_time` endpoints failed the deterministic
   trajectory check; the authoritative endpoint now includes the native
   fixed-domain evaluation hull and passes all such checks.  The collapsed
   rows remain diagnostic and never enter a width ranking.
7. **Why Torch dependency propagation deteriorates.** Reusing an increasingly
   complicated generator polynomial and independent remainder amplifies
   dependency and range overestimation.  Recentered affine/QR reset controls
   this at the cost of explicitly ranged discarded terms.
8. **Basis versus reset versus validator.** Matched basis isolates basis;
   affine-versus-box and Torch reset rows isolate carry/reset; stock/full/root
   Flow* rows isolate validator cache behavior.  Remaining cross-tool gaps
   cannot be assigned to one component alone.
9. **Runtime causes.** JIT, Python dispatch, C++ build/startup, and MPFR are
   backend effects.  polynomial support, range calls, Picard/refinement rounds,
   symbolic windows, and resets are algorithmic workload.
10. **Next Torch work.** Make normalized affine/QR reset a supported policy;
    add a documented restricted time-state basis option; improve polynomial
    range bounding and local-time overflow attribution; expose validator timing
    and defect diagnostics; and add a strict directed-rounding/MPFR validation
    backend before making proof-strength claims.
11. **BERN-NN-IBF.** The range-only prototype contains all five analytic
    cases and is strictly tighter on
    2 cancellation cases.
    Continue only toward a sparse, formally enclosed polynomial range backend.
    It is not a fourth solver and this plant-only evidence does not justify
    NN/CROWN integration.

## Validity limits and unresolved questions

- Valid: common one-step raw tube/endpoint, common affine carry, common box
  carry, accurately labelled native low-order, and same-time native Pareto
  comparisons.
- Not valid: a universal same-order ranking, width rankings across different
  absolute times, or proof-strength equivalence between MPFR and
  floating-point candidates.
- Flow* QR off/on remains unavailable through a stable public switch in this
  checkout and is labelled unavailable rather than emulated.
- DiffReach does not expose a separately width-valued structured remainder in
  its public result, limiting that decomposition.
- The corrected adaptive Flow* Van der Pol run reaches T=10.  The old
  collapsed endpoint excluded DOP853 samples in early segments; the
  source-level audit localizes this to endpoint restriction and repairs the
  raw endpoint with the native fixed-domain hull.  The repaired configuration
  has zero trajectory failures and remains eligible.
- Torch CPU/CUDA throughput is measured when `torch.cuda` exposes a device.
  This DiffReach environment exposes ['cpu:0'];
  a missing JAX GPU backend is recorded as unavailable rather than inferred
  from Torch's CUDA visibility.

The three-tool study therefore satisfies the original research request in its
scientifically valid form: it identifies the valid controlled comparisons,
retains native capability comparisons without equating their orders, and
attributes the major differences with matched-basis, reset, validation, defect,
and runtime controls.

## Reproduction

```bash
cd /srv/local/shengenli/torch_tm_flowpipe_three_tool_study
experiments/three_tool_deep_study/run_all.sh experiments/three_tool_deep_study/results/20260730T015245Z
```

Full scratch output is in `/srv/local/shengenli/torch_tm_flowpipe_three_tool_study/experiments/three_tool_deep_study/results/20260730T015245Z`.  The curated authoritative bundle is
published under
`experiments/three_tool_deep_study/artifacts/authoritative/20260730T015245Z`;
the eighteen figures are in its `plots/` directory.
