# torch-tm-flowpipe

This branch is a research review of plant-only polynomial ODE Taylor-model flowpipes. The sole numerical package, `src/torch_tm_flowpipe`, implements CPU-led validated propagation and optional CUDA range services. The two primary systems are Van der Pol and Brusselator. This branch organizes complete accepted horizons, all-step enclosure curves, mechanism tests, numerical repair boundaries, and actual timing; it does not introduce a new solver algorithm.

The current original-box fixed contracts both complete 1,000 accepted steps: Van der Pol with binary64 `h=0.01` reaches nominal T10; Brusselator with binary64 `h=0.02` reaches nominal T20. The endpoint-repaired CPU path and packet GPU range service have complete saved four-channel data. Native Flow* complete objects have been re-observed under the same current strict CPU range observer for the comparison. The packet route is a CPU-led solver that consumes real CUDA range results; a whole GPU engine has not been implemented. The optional resident composition block has measured B32×20 and continuous-prefix evidence, but no new original-box 1,000-step horizon and remains disabled by default.

The central [bounds and width-ratio figures](results/review/figures/) come from 24,000 normalized lower/upper curve rows and 24,000 relationship rows, recalculated from immutable complete saved records. On the current saved VDP comparison, the GPU-range route's median four-channel width ratios against Flow* are roughly 1.17–1.20; against the CPU reference they are at roundoff scale. Brusselator's relationship to Flow* changes by channel and time, so a single "tightest" rank is unwarranted. The numerical reference has scoped repairs and known retained-coefficient limitations; neither completion nor a local strict observer is a whole-solver formal proof. The [provenance map](results/review/provenance.json) names each source and observer.

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

The equivalent supported Brusselator fixed, adaptive VDP, and GPU-range commands are in [REPRODUCING.md](docs/REPRODUCING.md). The GPU-range and resident smoke commands report clearly when CUDA is unavailable. Flow* is required only for a new native Flow* rerun, not for reading or plotting the saved comparison.

This branch does not run the deferred whole-engine feasibility/adoption task. The old August common-prefix, DiffReach fixed-support, S1, TORA, and withdrawn "fastest/tightest" statements are historical context, not the current main result.
