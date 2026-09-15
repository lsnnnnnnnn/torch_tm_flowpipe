# Experiment reading map

Generated from [the single registry](../experiments/review_suite/registry.yaml) by review-suite summarize.
Each entry follows question → frozen configuration/code → original evidence → derived table/figure → limited claim.

## Van der Pol original-box fixed 1000-step horizon

**Question.** How far does the original box solve, and how do all endpoint/tube x/y enclosures compare with native Flow*?

**Scope and role.** original unpartitioned B1, fixed 1000 accepted steps, T10 current main full-horizon comparison.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** artifacts/runs/live_gpu_packets_20260910T023603Z/full_horizon/van_der_pol-Gp

**Raw data.** ['artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/vdp_full/models.jsonl.gz', 'artifacts/runs/live_gpu_packets_20260910T023603Z/full_horizon/van_der_pol-Gp/full_horizon_steps.jsonl.gz', 'artifacts/runs/xiangru_adoption_20260907T032448Z/raw_minimal/native_vdp/models.jsonl.gz']

**Derived table.** [results/review/summary.csv](../results/review/summary.csv)

**Figures.** results/review/figures/vdp_x_bounds.svg, results/review/figures/vdp_y_bounds.svg, results/review/figures/vdp_width_ratios.svg

**Reproduction level.** reused_recomputed.

**Observed conclusion.** All three saved routes have 1000 accepted fixed steps and complete all-step endpoint/tube x/y ranges; GPU-range is CPU-led, resident full horizon is not established.

**Run.** python -m experiments.review_suite.cli run --experiment vdp-fixed-full --backend cpu --out <new_dir>

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Research CPU reference has scoped arithmetic repairs, not a complete formal proof; current strict observer is re-executed on saved CPU/Flow* models; internal Flow* validator and history policy differ.

## Brusselator original-box fixed 1000-step horizon

**Question.** How far does the original box solve, and how do all endpoint/tube x/y enclosures compare with native Flow*?

**Scope and role.** original unpartitioned B1, fixed 1000 accepted steps, T20 current main full-horizon comparison.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** artifacts/runs/live_gpu_packets_20260910T023603Z/full_horizon/brusselator-Gp

**Raw data.** ['artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/brusselator_full/models.jsonl.gz', 'artifacts/runs/live_gpu_packets_20260910T023603Z/full_horizon/brusselator-Gp/full_horizon_steps.jsonl.gz', 'artifacts/runs/xiangru_adoption_20260907T032448Z/raw_minimal/native_brusselator/models.jsonl.gz']

**Derived table.** [results/review/summary.csv](../results/review/summary.csv)

**Figures.** results/review/figures/brusselator_x_bounds.svg, results/review/figures/brusselator_y_bounds.svg, results/review/figures/brusselator_width_ratios.svg

**Reproduction level.** reused_recomputed.

**Observed conclusion.** All three saved routes have 1000 accepted fixed steps and complete all-step endpoint/tube x/y ranges; GPU-range is CPU-led, resident full horizon is not established.

**Run.** python -m experiments.review_suite.cli run --experiment brusselator-fixed-full --backend cpu --out <new_dir>

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Research CPU reference has scoped arithmetic repairs, not a complete formal proof; current strict observer is re-executed on saved CPU/Flow* models; internal Flow* validator and history policy differ.

## Van der Pol native adaptive T10

**Question.** Does the endpoint-repaired CPU path retain its separately defined adaptive T10 completion?

**Scope and role.** native adaptive h_min=.002, h_max=.1, T10; separate from fixed 1000-step timing CPU research reference; adaptive schedule.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/vdp_adaptive

**Raw data.** artifacts/runs/endpoint_roundoff_repair_20260908/raw_minimal/vdp_adaptive/models.jsonl.gz

**Derived table.** [results/review/adaptive.csv](../results/review/adaptive.csv)

**Figures.** None needed.

**Reproduction level.** reused_recomputed.

**Observed conclusion.** The endpoint-repaired CPU reference completed its separately defined adaptive T10 contract in 246 accepted steps and 35 rejected attempts.

**Run.** python -m experiments.review_suite.cli run --experiment vdp-adaptive --backend cpu --out <new_dir>

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Adaptive step sequence and validator are distinct from the fixed-step comparison; no complete formal proof.

## Van der Pol cross-step symbolic error history

**Question.** Did retaining linear relations among past errors avoid premature widening and terminal rejection?

**Scope and role.** fixed VDP controlled historical implementations, not a current arithmetic-proof claim historical controlled mechanism comparison.

**Configuration.** [artifacts/runs/vdp_c3_cross_step_causal_closure_20260827/EVIDENCE_CONTRACT.json](../artifacts/runs/vdp_c3_cross_step_causal_closure_20260827/EVIDENCE_CONTRACT.json)

**Original evidence.** artifacts/runs/vdp_c3_cross_step_causal_closure_20260827

**Raw data.** artifacts/runs/vdp_c3_cross_step_causal_closure_20260827/RESULT.json

**Derived table.** [results/review/mechanisms.csv](../results/review/mechanisms.csv)

**Figures.** results/review/figures/mechanism_comparisons.svg

**Reproduction level.** historical_replay.

**Observed conclusion.** The historical VDP native adaptive C3 run reached T10 where its C2 control stopped around 6.715; at the common fixed 632-step boundary C3 was materially narrower.

**Run.** Replay the source commit in an isolated checkout using the package README; not a current review CLI backend.

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Precedes later scoped numerical repairs; mechanism evidence does not certify all retained-coefficient operations.

## Brusselator accepted-remainder refinement

**Question.** Can safely continuing the remainder map after first acceptance recover the fixed T20 horizon?

**Scope and role.** Brusselator SR1000 legacy and C4 on their frozen fixed contract historical controlled mechanism comparison.

**Configuration.** [artifacts/runs/brusselator_sr1000_c4_closure_20260828/raw/contract.json](../artifacts/runs/brusselator_sr1000_c4_closure_20260828/raw/contract.json)

**Original evidence.** ['artifacts/runs/brusselator_sr1000_c4_closure_20260828', 'artifacts/runs/brusselator_live_range_c5_20260828']

**Raw data.** ['artifacts/runs/brusselator_sr1000_c4_closure_20260828/CLOSURE_RESULT.json', 'artifacts/runs/brusselator_live_range_c5_20260828/RESULT.json']

**Derived table.** [results/review/mechanisms.csv](../results/review/mechanisms.csv)

**Figures.** results/review/figures/mechanism_comparisons.svg

**Reproduction level.** historical_replay.

**Observed conclusion.** The frozen Brusselator SR1000 legacy stopped after 357 steps before queue fill/reset, whereas C4 refined completed 1000; no C5 algorithm was implemented.

**Run.** Replay fixed historical commit in an isolated checkout; no current C5 backend exists.

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** SR1000 legacy rejects before queue fill/reset; C5 was an audit label without a new algorithm; older arithmetic scope applies.

## Versioned roundoff and state-commit witnesses

**Question.** Which known missed enclosures were repaired, and did full-horizon conclusions change?

**Scope and role.** time substitution, powers, coefficient arithmetic, history matrix/scale, commit/rollback; versioned paths scoped regression evidence.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** artifacts/runs/endpoint_roundoff_repair_20260908

**Raw data.** artifacts/runs/endpoint_roundoff_repair_20260908/after_endpoint_oracles.json

**Derived table.** [results/review/numerical_witnesses.csv](../results/review/numerical_witnesses.csv)

**Figures.** None needed.

**Reproduction level.** reused_recomputed.

**Observed conclusion.** Scoped endpoint/request/block arithmetic repairs address exact local misses; the repaired fixed horizons still complete with tiny all-step width changes, without a whole-solver proof.

**Run.** python -m experiments.review_suite.cli smoke --backend cpu

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Fixes have local entry boundaries; endpoint repair does not imply whole-solver proof.

## CPU prepared remainder and ordered tensor range

**Question.** Which repeated work was actually removed without changing accepted numerical objects?

**Scope and role.** original single complete paired runs and separate short repeated pairings paired implementation performance evidence.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** ['artifacts/runs/repaired_solver_performance_20260908T034636Z', 'artifacts/runs/boundary_execution_20260908T172756Z']

**Raw data.** ['artifacts/runs/repaired_solver_performance_20260908T034636Z/RESULT.json', 'artifacts/runs/boundary_execution_20260908T172756Z/RESULT.json']

**Derived table.** [results/review/cpu_pairings.csv](../results/review/cpu_pairings.csv)

**Figures.** results/review/figures/cpu_pairings.svg

**Reproduction level.** reused_recomputed.

**Observed conclusion.** Both paired full runs preserved complete saved numerical objects, but the benefits have one complete pair each and the ordered-range denominator already enables prepared replay.

**Run.** Historical fixed-run replay only; no new optimization campaign in review CLI.

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Second denominator already has the first optimization enabled; timing dates/observation cost differ.

## CUDA range batch to online packet and resident block

**Question.** How much local acceleration survived in complete online work?

**Scope and role.** local request, online prefix, packet full B1 horizons, resident B32x20 and small prefix; no resident B1 full horizon optional GPU research route, resident block default disabled.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** ['artifacts/runs/range_batch_device_20260909T030609Z', 'artifacts/runs/live_range_solver_20260909T053007Z', 'artifacts/runs/live_gpu_packets_20260910T023603Z', 'artifacts/runs/resident_tm_block_20260914T032650Z']

**Raw data.** ['artifacts/runs/range_batch_device_20260909T030609Z/RESULT.json', 'artifacts/runs/live_range_solver_20260909T053007Z/RESULT.json', 'artifacts/runs/live_gpu_packets_20260910T023603Z/RESULT.json', 'artifacts/runs/resident_tm_block_20260914T032650Z/RESULT.json']

**Derived table.** [results/review/gpu_route.csv](../results/review/gpu_route.csv)

**Figures.** results/review/figures/gpu_local_vs_online.svg

**Reproduction level.** reused_recomputed.

**Observed conclusion.** Offline local CUDA rates were large; the measured online B32x20 gains were much smaller, and resident missed its 1.20 paired-median goal despite winning all five pairs.

**Run.** python -m experiments.review_suite.cli smoke --backend gpu-range

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Local request/block rates are not solver rates; resident engineering 1.20 target missed; whole GPU engine is not implemented.

## Matched B32 by 20-step Flow* scale

**Question.** What is the remaining same-workload wall-time gap?

**Scope and role.** 32 distinct 8x4 subboxes, each 20 steps; five resident/Gp pairs per plant and one Flow* invocation per plant resident prototype versus native Flow* timing scale.

**Configuration.** [artifacts/runs/resident_tm_block_20260914T032650Z/BLOCK_SELECTION_PROFILE.json](../artifacts/runs/resident_tm_block_20260914T032650Z/BLOCK_SELECTION_PROFILE.json)

**Original evidence.** artifacts/runs/resident_tm_block_20260914T032650Z

**Raw data.** ['artifacts/runs/resident_tm_block_20260914T032650Z/timings_raw.csv', 'artifacts/runs/resident_tm_block_20260914T032650Z/matched_flowstar.csv']

**Derived table.** [results/review/matched_flowstar.csv](../results/review/matched_flowstar.csv)

**Figures.** results/review/figures/matched_flowstar_scale.svg

**Reproduction level.** reused_recomputed.

**Observed conclusion.** Native Flow* finished the same 32-distinct-box by 20-step workload in one invocation per plant; resident median wall time remained roughly 356x and 44x higher.

**Run.** Current resident review prefix only; matched Flow* legacy command is in original package.

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Main parameters/CPU resource matched, internal validation/history algorithms not identical; timing is engineering scale, not pure hardware rate.

## Van der Pol resident-composition original B1 prefix

**Question.** Does the optional device-resident composition run a genuine continuous original-box prefix after review packaging?

**Scope and role.** original B1 20 consecutive accepted steps, not fixed 1000-step T10/T20 optional default-off resident prototype.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** artifacts/runs/resident_tm_block_20260914T032650Z

**Raw data.** artifacts/runs/resident_tm_block_20260914T032650Z/RESULT.json

**Derived table.** [results/review/gpu_route.csv](../results/review/gpu_route.csv)

**Figures.** results/review/figures/gpu_local_vs_online.svg

**Reproduction level.** not_run.

**Observed conclusion.** Explicit optional resident prefix only; no original-box 1000-step resident completion is claimed.

**Run.** python -m experiments.review_suite.cli run --experiment vdp-resident-prefix --backend resident --out <new_dir>

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Block-level scoped checks, no whole solver proof or full original-box resident horizon.

## Brusselator resident-composition original B1 prefix

**Question.** Does the optional device-resident composition run a genuine continuous original-box prefix after review packaging?

**Scope and role.** original B1 20 consecutive accepted steps, not fixed 1000-step T10/T20 optional default-off resident prototype.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** artifacts/runs/resident_tm_block_20260914T032650Z

**Raw data.** artifacts/runs/resident_tm_block_20260914T032650Z/RESULT.json

**Derived table.** [results/review/gpu_route.csv](../results/review/gpu_route.csv)

**Figures.** results/review/figures/gpu_local_vs_online.svg

**Reproduction level.** not_run.

**Observed conclusion.** Explicit optional resident prefix only; no original-box 1000-step resident completion is claimed.

**Run.** python -m experiments.review_suite.cli run --experiment brusselator-resident-prefix --backend resident --out <new_dir>

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Block-level scoped checks, no whole solver proof or full original-box resident horizon.

## Van der Pol resident-composition fixed B2 two-step sample

**Question.** Can the optional resident composition complete two consecutive steps on two distinct frozen partition subboxes?

**Scope and role.** two distinct outward 8x4 partition subboxes, each two accepted steps; not original B1 and not T10/T20 optional default-off resident prototype.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** artifacts/runs/resident_tm_block_20260914T032650Z

**Raw data.** artifacts/runs/resident_tm_block_20260914T032650Z/RESULT.json

**Derived table.** [results/review/gpu_route.csv](../results/review/gpu_route.csv)

**Figures.** results/review/figures/gpu_local_vs_online.svg

**Reproduction level.** not_run.

**Observed conclusion.** Pending fresh fixed B2 two-step witness; existing B32x20 evidence is separate and no full original-box resident horizon is claimed.

**Run.** python -m experiments.review_suite.cli run --experiment vdp-resident-b2 --backend resident --out <new_dir>

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Block-level scoped checks, no whole solver proof or full original-box resident horizon.

## Brusselator resident-composition fixed B2 two-step sample

**Question.** Can the optional resident composition complete two consecutive steps on two distinct frozen partition subboxes?

**Scope and role.** two distinct outward 8x4 partition subboxes, each two accepted steps; not original B1 and not T10/T20 optional default-off resident prototype.

**Configuration.** [benchmarks/review/fixed_profiles.json](../benchmarks/review/fixed_profiles.json)

**Original evidence.** artifacts/runs/resident_tm_block_20260914T032650Z

**Raw data.** artifacts/runs/resident_tm_block_20260914T032650Z/RESULT.json

**Derived table.** [results/review/gpu_route.csv](../results/review/gpu_route.csv)

**Figures.** results/review/figures/gpu_local_vs_online.svg

**Reproduction level.** not_run.

**Observed conclusion.** Pending fresh fixed B2 two-step witness; existing B32x20 evidence is separate and no full original-box resident horizon is claimed.

**Run.** python -m experiments.review_suite.cli run --experiment brusselator-resident-b2 --backend resident --out <new_dir>

**Recompute.** python -m experiments.review_suite.cli summarize --all --out results/review

**Numerical limit.** Block-level scoped checks, no whole solver proof or full original-box resident horizon.
