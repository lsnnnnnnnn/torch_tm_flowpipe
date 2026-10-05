# torch-tm-flowpipe

**Current ARCH-COMP26 delivery — evidence through 2026-10-05:** [rewritten Chinese report](docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md) · [editable Word and matching PDF](docs/evidence/results/archcomp26_report_20261005/README.md) · [all times](docs/evidence/results/archcomp26_report_20261005/timing/current_selected_four_way.csv) · [all state widths](docs/evidence/results/archcomp26_report_20261005/widths/summary.csv) · [why each incomplete case cannot finish](docs/evidence/results/archcomp26_report_20261005/blockers/README.zh.md) · [new P3 speed study](research/p3_speed_tightness_20261005/README.md). The 16×4 matrix still has 38 new full numerical horizons, eight audited historical full horizons, 14 incomplete cells, and four contract-blocked discrete cells. Five complete P3 candidates lower recorded runtime while retaining the compared saved ranges; no universal speed or tightness win is claimed. Python plotting remains supported. The original 289-attempt ledger and 13 new optimization/qualification stages are separate; reporting does not rerun frozen experiments.

**Historical checkpoint — 2026-09-30:** [GPU/QUAD source and results](research/gpu_verified_20260930/README.md) · [current status](research/gpu_verified_20260930/STATUS.md) · [next-stage goal](research/gpu_verified_20260930/GOAL_HUAN_QUAD_PLOTTING_ARCHCOMP_20260930.md) · [English slides](research/gpu_verified_20260930/report/slides.pdf) · [Chinese script](research/gpu_verified_20260930/report/chinese_verbatim_script.docx).

This branch collects the exact source associated with the completed 1024-box QUAD P3/trig experiment, its model/configuration, and the latest reporting evidence. It preserves the earlier project below and keeps external CUDA/environment/checkpoint dependencies explicit. The native QUAD six-hour run is now known to have timed out with 600 steps recorded in complete periods; this supersedes the older 400-step observation. Publication did not rerun the solver.

Current next-stage work is tracked in the [Huan QUAD mode/speed audit](docs/HUAN_QUAD_SPEED_AND_MODES.md), the [flowpipe plotting guide](docs/flowpipe_plotting.md), its [archived-data validation](docs/FLOWPIPE_PLOT_VALIDATION_20261001.md), and the [ARCH-COMP26 non-VCAS contract scaffold](benchmarks/archcomp26/README.md). The [2026-10-01 remote recheck](docs/REMOTE_READONLY_RECHECK_20261001.md) records the unchanged native terminal state. That dated checkpoint preceded the later explicit resumption; current state is recorded in the goal and handoff above.

## Earlier supported CPU and whole-engine review

**2026-09-23 progress report:** [中文详细报告](docs/progress_review_20260923/REPORT.md) · [Report PDF](docs/progress_review_20260923/report.pdf) · [Beamer slides](docs/progress_review_20260923/slides.pdf) · [LaTeX sources and evidence](docs/progress_review_20260923/README.md). This newer review records matched timing, full-trajectory widths, exact configurations and incomplete cases for our engine, Huan, Xiangru and native Flow*. Work is paused after this requested reporting delivery; the overall acceleration goal is not complete.


This working branch also contains the completed whole-engine feasibility experiment: see the [adapter run guide](experiments/whole_engine_feasibility/README.md) and [2026-09-21 execution report](docs/WHOLE_ENGINE_EXECUTION_20260921.md). The original review below retains its saved sources; the new complete-engine experiment has separate versions, outputs, and CPU/CUDA performance conclusions.

This branch is a research review of plant-only polynomial ODE Taylor-model flowpipes. The sole numerical package, `src/torch_tm_flowpipe`, implements CPU-led validated propagation and optional CUDA range services. The two primary systems are Van der Pol and Brusselator. This branch organizes complete accepted horizons, all-step enclosure curves, mechanism tests, numerical repair boundaries, and actual timing; it does not introduce a new solver algorithm.

The current original-box fixed contracts both complete 1,000 accepted steps: Van der Pol with binary64 `h=0.01` reaches nominal T10; Brusselator with binary64 `h=0.02` reaches nominal T20. The endpoint-repaired CPU path and packet GPU range service have complete saved four-channel data. Native Flow* complete objects have been re-observed under the same current strict CPU range observer for the comparison. The packet route is a CPU-led solver that consumes real CUDA range results. The added `whole-engine` route instead runs the complete external strict PyTorch engine on an explicit CPU or CUDA device; its separately versioned full-horizon evidence is linked above. The optional resident composition block has measured B32×20 and continuous-prefix evidence, but no new original-box 1,000-step horizon and remains disabled by default.

The central [bounds and width-ratio figures](results/review/figures/) come from 24,000 normalized lower/upper curve rows and 24,000 relationship rows, recalculated from immutable complete saved records. On the current saved VDP comparison, the GPU-range route's median four-channel width ratios against Flow* are roughly 1.17–1.20; against the CPU reference they are at roundoff scale. Brusselator's relationship to Flow* changes by channel and time, so a single "tightest" rank is unwarranted. The numerical reference has scoped repairs and known retained-coefficient limitations; neither completion nor a local strict observer is a whole-solver formal proof. The [provenance map](results/review/provenance.json) names each source and observer.

The separate [fresh confirmation table](results/review/fresh_confirmations.csv) reports completed new fixed-source CPU and GPU-range horizons and the default-off resident small scopes, with each raw-run path, accepted step count and exact old/new bound comparison. These single reruns verify that the organized branch still executes; the main figures and historical paired performance decisions retain their original saved scientific sources.

Read in this order:

1. [English project review for Xiangru and Huan](docs/PROJECT_REVIEW.md), then the [Chinese reporting version](docs/PROJECT_REVIEW_ZH.md).
2. [Experiment map](docs/EXPERIMENTS.md) and [method-to-code map](docs/METHOD_AND_CODE_MAP.md).
3. [Reproduction guide](docs/REPRODUCING.md) and [numerical/external-code scope](docs/NUMERICAL_SCOPE_AND_EXTERNAL_CODE.md).

The frozen [review configuration](benchmarks/review/fixed_profiles.json) records equations through the referenced source contracts, exact decimal initial boxes, actual outward binary64 initial ranges, retained orders, binary64 steps, remainder budgets, cutoff, validation mode, queue capacity, and observer boundary. The [single registry](experiments/review_suite/registry.yaml) assigns source versions and evidence levels. Historical mechanism and withdrawn result packages remain versioned and recoverable; see [reorganization notes](docs/maintenance/REORGANIZATION.md).

## Quick start

Use a Python environment with PyTorch, PyYAML and matplotlib. CUDA is optional for CPU reading and reruns. The working environment for this review is Python 3.11, PyTorch 2.5.1+cu121, with V100 CUDA available.

```bash
python -m pip install -e .
python -m experiments.review_suite.cli list
python -m experiments.review_suite.cli smoke --backend cpu
python -m experiments.review_suite.cli plot --all --out results/review/figures
```

To re-observe the selected complete raw models and regenerate all managed tables before plotting:

```bash
python -m experiments.review_suite.cli summarize --all --out results/review
```

A new 1,000-step run needs a clean source checkout and a new output directory outside it; the runner records the source HEAD and refuses an existing directory. For example:

```bash
mkdir -p ../review_runs
python -m experiments.review_suite.cli run --experiment vdp-fixed-full --backend cpu --out ../review_runs/vdp_cpu_001
```

The equivalent supported Brusselator fixed, adaptive VDP, GPU-range, and complete-engine commands are in [REPRODUCING.md](docs/REPRODUCING.md). The GPU-range and resident smoke commands report clearly when CUDA is unavailable. Flow* is required only for a new native Flow* rerun, not for reading or plotting the saved comparison.

The original review did not run the then-deferred whole-engine feasibility/adoption task; the separate experiment linked above now records its results. The old August common-prefix, DiffReach fixed-support, S1, TORA, and withdrawn "fastest/tightest" statements are historical context, not the current main result.
