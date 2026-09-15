# Reproducing the research review

Run commands from the repository root in a clean checkout of this review branch. The working environment used to assemble the branch is Python 3.11.15 and PyTorch 2.5.1+cu121; CPU operations need PyTorch, PyYAML and matplotlib, and CUDA operations need a compatible NVIDIA device/NVRTC. The three kernel sources are packaged with `torch_tm_flowpipe`. A CPU install does not compile CUDA. The stock Flow* executable is not needed to read or redraw saved data.

The [single registry](../experiments/review_suite/registry.yaml) distinguishes source-run hashes from delivery revisions and names each raw/config/figure path. The fixed [review profile](../benchmarks/review/fixed_profiles.json) is checked against the live source configuration, exact frozen source bytes, and outward binary64 original boxes before any rerun. The endpoint repair still uses the tracked, SHA-checked [matched-contract package](../artifacts/runs/xiangru_adoption_20260907T032448Z/MATCHED_CONTRACTS.json) and imports its original helpers; it does not read a server scratch directory.

## Read existing results

Open [the main summary](../results/review/summary.csv), [all-step normalized bounds](../results/review/bounds/curves.csv.gz), [four-channel relationships](../results/review/bounds/width_ratios.csv.gz), and [figures](../results/review/figures/). The [provenance JSON](../results/review/provenance.json) records each immutable CPU/GPU/Flow* source, the strict common observer, versions, and row counts. [EXPERIMENTS.md](EXPERIMENTS.md) is generated from the registry. Reading these files does not run an ODE.

The fixed main comparison has one original unpartitioned B1 box per system and 1000 actual accepted steps. The endpoint view is the range at the end of a step; tube is the whole accepted segment over its time interval. The x/y channels are separate. The current strict CPU range observer is re-executed on the saved complete CPU and native Flow* Taylor-model objects; the GPU run contributes its own accepted-step recorded bounds. Internal Flow* validation, refinement and history are not identical to our solver.

## Redraw from the original saved records

```bash
python -m pip install -e .
python -m experiments.review_suite.cli list
python -m experiments.review_suite.cli summarize --all --out results/review
python -m experiments.review_suite.cli plot --all --out results/review/figures
```

`summarize` reads only paths selected by the registry. It verifies the complete GPU accepted-state record, re-observes 1000 CPU and 1000 Flow* models per plant, and rebuilds 24,000 lower/upper curve rows, 24,000 width-relation rows, the paired timing/mechanism tables, provenance, and the experiment reading map. It does not download evidence, compile CUDA, or solve an ODE. `plot` reads those managed tables and redraws ten SVGs. It does not infer values from older PNGs. These commands can replace managed derived files but do not write the frozen `artifacts/runs` inputs.

If you choose another derived root, use matching paths for both operations, such as `--out ../review_derived` for summarize and `--out ../review_derived/figures` for plot. The raw source pointers remain repository-relative.

## Run small accepted examples

```bash
python -m experiments.review_suite.cli smoke --backend cpu
python -m experiments.review_suite.cli smoke --backend gpu-range
python -m experiments.review_suite.cli smoke --backend resident
```

CPU smoke runs two real accepted steps on both systems; GPU-range smoke uses the packet route inside the actual solver chain; resident smoke explicitly enables the otherwise disabled composition prototype. CUDA smoke requires a device/NVRTC; a missing dependency yields a nonzero exit and a diagnostic, never a fabricated PASS. Temporary smoke evidence is discarded. Use the `run` prefix commands below for retained resident evidence.

## New complete experiments

First use a clean source checkout. The CPU research runner binds the scientific source HEAD and rejects tracked or untracked worktree changes at its start. All review `run` commands reject an existing output directory. Make only the parent directory, and put outputs outside the source tree while running:

```bash
mkdir -p ../review_runs
python -m experiments.review_suite.cli run --experiment vdp-fixed-full --backend cpu --out ../review_runs/vdp_cpu_001
python -m experiments.review_suite.cli run --experiment brusselator-fixed-full --backend cpu --out ../review_runs/bruss_cpu_001
python -m experiments.review_suite.cli run --experiment vdp-fixed-full --backend gpu-range --out ../review_runs/vdp_gpu_001
python -m experiments.review_suite.cli run --experiment brusselator-fixed-full --backend gpu-range --out ../review_runs/bruss_gpu_001
python -m experiments.review_suite.cli run --experiment vdp-adaptive --backend cpu --out ../review_runs/vdp_adaptive_001
```

Fixed CPU/GPU-range commands use the original B1 mathematical box, actual outward binary64 construction, order 4/h=0.01/queue 100 for VDP and order 6/h=0.02/queue 1000 for Brusselator, with 1000 steps, remainder radius 1e-4, cutoff 1e-10, and the frozen validation/refinement/range policy. They output actual accepted endpoint/tube x/y bounds at every step. If a run stops, keep the real prefix and failure record; do not fill to 1000 or change the contract. The adaptive VDP T10 uses its separate h_min=.002/h_max=.1 frozen scheduler and is not ranked with fixed-step seconds.

A full rerun can take many minutes and requires storage. One elapsed time is a single observation, not a stable throughput estimate; source metadata separates solve, observer, evidence/export and startup where the runner records them. The packet GPU-range route does not enable resident composition and is still CPU-led. A new native Flow* rerun uses the fixed original package's external build/observer instructions; it is unnecessary for the saved main curves.

## Optional resident evidence

```bash
python -m experiments.review_suite.cli run --experiment vdp-resident-prefix --backend resident --out ../review_runs/vdp_resident_20
python -m experiments.review_suite.cli run --experiment brusselator-resident-prefix --backend resident --out ../review_runs/bruss_resident_20
python -m experiments.review_suite.cli run --experiment vdp-resident-b2 --backend resident --out ../review_runs/vdp_resident_b2
python -m experiments.review_suite.cli run --experiment brusselator-resident-b2 --backend resident --out ../review_runs/bruss_resident_b2
```

The first two commands are continuous 20-step original-box B1 prefixes, not T10/T20. The next two request two distinct fixed 8×4 partition subboxes with two steps each (B2), not the original box or a full horizon. The resident block is default-off; it is explicitly enabled only in this route. The saved B32×20 performance pairs and matched Flow* scale are reused/recomputed in this branch; they are not rerun by summarize or plot. The deferred complete-device-engine feasibility/adoption goal was not executed here.

Historical mechanism runs require their original fixed source checkout and environment as documented in their packages. Do not silently import a historical numerical package into the current review process. [The recovery map](../archive/index.csv) identifies archived old current-tree material by exact commit and path.
