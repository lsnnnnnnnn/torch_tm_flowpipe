# flowstar_gpu

GPU-accelerated **batched Taylor-model reachability analysis** for nonlinear ODEs — a PyTorch
reimplementation of the [Flow*](https://github.com/chenxin415/flowstar) flowpipe-construction
algorithm, built for neural-network-controlled-system (NNCS) verification: a GPU-batched NN
verifier (CROWN / auto_LiRPA) needs thousands of flowpipes computed in parallel under
branch-and-bound, and Flow* on a CPU core is the bottleneck this project removes.

**Design goals, in priority order:**

1. **Soundness** — every result is a rigorous over-approximation of the reachable set despite
   float64 round-to-nearest GPU hardware (directed rounding emulated via `torch.nextafter`
   outward stepping + Rump-style a-priori reduction bounds; `mode="strict"` is provably
   *sounder than Flow\* itself* — see `docs/GOTCHAS.md` #15/#16/#19).
2. **Parity with Flow\*** — the same algorithm stage-for-stage, validated against a C++ oracle
   that dumps Flow*'s exact Taylor models as hex floats (`tools/oracle/`). **12 of the 13**
   Flow* continuous benchmarks have parity evidence (6 under frozen automated gates, 6
   session-measured — strength per config in `docs/PARITY.md`); the 13th,
   coupled_vanderpol, has NONE and is blocked on the recorded adaptive+SR gap
   (`docs/ARCHITECTURE.md` scope table).
3. **Throughput** — honest speedups. Against the **80-core** CROWN-Reach + Flow*
   baseline on the continuous-time ARCH-COMP NNCS suite, the M10 sparse engine is
   par-or-faster on 8 of 11 benchmarks — up to **46× faster** (dp_less 4.3 s vs 196.9 s,
   nav_standard 11.8 s vs 329.1 s, quad_reduced 9.3 s vs 137.7 s) with all verdicts
   identical, and it VERIFIES the 1024-cell `quad_official` in 83 s where the baseline
   never finishes (`CROWN-Reach/comparison/REPORT.md`). Engine internals:
   `docs/OPTIMIZATION.md`; reproduction: `docs/REPRODUCE.md`. (The older
   flowstar-native dense-engine numbers — 73.6× vs one CPU core at batch 4096 —
   are in `docs/BENCHMARKS.md`.)

## Status

All planned milestones are complete, through three generations of execution engine.
Tags: `m0-oracle`, `m2-parity`, `m3-batching`, `m4-symbolic-remainder`,
`m5-nonpoly-interval`, `m6-adaptive-safety`, `m7-nncs` (M1's gates were folded into
`m2-parity`), then `m9-crown-reach` (the CROWN-Reach ARCH-COMP integration,
`integrations/crown_reach/gpu_driver.py`) and `m10-sparse-kernels` (the current
engine: union-support sparsity + directed-rounding CUDA kernels + one-launch
tape-interpreter kernels — `docs/OPTIMIZATION.md` is the deep-dive; M11 added the
XLA-style compilation study, adopted opt-in). Capabilities: polynomial and non-polynomial ODEs
(sin/cos/exp/log/sqrt/div), interval-coefficient ODEs (`[a,b]` literals), symbolic remainders,
fixed + adaptive stepsize, safety checking, and the auto_LiRPA-based NNCS closed loop
(`examples/nncs_demo.py`). Notable byproducts: three genuine Flow* soundness bugs found and
documented (`docs/GOTCHAS.md` #16, #18, #19 — the sin/cos series bug is an outright error in
Flow*'s trig enclosures). Known de-scopes are recorded in `docs/ARCHITECTURE.md` (scope table);
the backlog lives in `docs/DEVELOPMENT.md` §7. Cross-tool comparisons (CPU
baseline, DiffReach) with their data and runners are indexed at
`CROWN-Reach/comparison/README.md`.

**Since 2026-07-30 the project also ships a NEW harder ARCH-COMP-style AINNCS benchmark
suite** in `benchmarks/` — 12 directories = 11 runnable instances + 1 open challenge
(`Vehicle-Drift`), each with plant, controller, provenance and configs, run three-engine
(this engine / CPU Flow\* via CROWN-Reach / DiffReach). It is also where this engine's
limits show, and they are reported as such: on `F16-GCAS` the CPU tool beats us
**11.9×** end-to-end on equal work, and `ACC-Chain` stops at N=5 for us against N=9 for
the other two engines (GOTCHAS S2, `DEVELOPMENT.md` §7). Start at `benchmarks/README.md`;
design and phase gates in `docs/BENCHMARK_PLAN.md` (proposal: `docs/BENCHMARK_PROPOSAL.md`);
results in `CROWN-Reach/comparison/SUITE_RESULTS.md`. Engine features that effort added —
composed-TM spec checking (`refine_specs`), spec time windows, per-variable
`remainder_estimation`, the symbolic-remainder queue reset — are documented in
`docs/ARCHITECTURE.md` and `docs/GOTCHAS.md` (#6, #22, E1-*, F2-*). Three upstream
auto_LiRPA defects found on the way have fileable reports in
`docs/UPSTREAM_AUTOLIRPA_BUG{,2_SAME_SLOPE,3_DTYPE}.md` (GOTCHAS X1, X4, X5).

## Taking over development?

**In your first hour, in this order:**

| # | read / do | why |
|---|---|---|
| 1 | `docs/DEFECTS.md` — **the box at the top, then the index** | every defect this project found in four external codebases and in our own, with a status and *what you must do*. Its first item is a one-command check that the `auto_LiRPA` patch survived; without it 15 of the 30 official controllers cannot be bounded at all |
| 2 | `docs/REPRODUCE.md` §0–§1 | setup, and **the four external checkouts** every `file:line` citation in this repo depends on |
| 3 | `benchmarks/README.md` → **"Your first run"** | the one instance in the suite whose bracket is decided in both directions, and what you should see |
| 4 | `docs/TROUBLESHOOTING.md` | when it does not do that. Message-first; also the **failure taxonomy** — `Flow* terminated. / Broken branch: 0` is the suite's modal outcome and usually *not* a bug |
| 5 | `docs/DEVELOPMENT.md` | mental model, the three-tape compiler invariant, workflows, test policies, the **`dense` vs `sparse` decision rule**, the **enforcement index** (§8.5: invariant → the test that pins it), and the backlog |

Then, as you need them: `docs/ARCHITECTURE.md` (layouts/tiers/scope),
`docs/ALGORITHM.md` (Flow* ↔ our-code map), `docs/GOTCHAS.md` (read **before** trusting
Flow* as a reference), `docs/OPTIMIZATION.md` (the M10/M11 execution stack — required
before touching the sparse path), `docs/CONFIG_FORMAT.md` (the benchmark YAML, key by
key — read it before writing a config), `docs/PARITY.md` + `docs/BENCHMARKS.md`
(evidence), `docs/PROGRESS.md` (history — read its index first). Only then the deeper
benchmark-suite side: `docs/BENCHMARK_PLAN.md` →
`CROWN-Reach/comparison/SUITE_RESULTS.md` (results; prefer the `.csv`).

## Setup

Requires: [uv](https://docs.astral.sh/uv/); an NVIDIA GPU — developed on V100/**sm_70**, which
is why torch is pinned to the **cu126** wheel index (CUDA 13 wheels drop Volta; never install a
plain-PyPI torch here); the Flow* toolbox checkout at `../flowstar` with `libflowstar.a` built
(GMP/MPFR/GSL/GLPK; GCC 15 needs `-Wno-template-body`) for the parity oracle.

What you can run depends on what you have (rough timings on this machine):

```sh
# Tier 1 - needs only uv + a CUDA GPU:
uv sync                  # .venv with torch 2.13 cu126 + dev tools
make test                # fast dev loop, ~1 min (unit + property suites)
make cov                 # the gate: 100.00% line+branch coverage, ~3 min (needs CUDA)
make soundness           # MC trajectory containment (independent of Flow*), ~2 min
make adversarial         # adversarial soundness: PGD attacks, splitting,
                         #   mutation detector-power, ~3 min (-full: slow sweeps)

# Tier 2 - additionally needs the Flow* checkout at ../flowstar (parity work only;
# NOT needed for the dev loop - golden oracle slices are committed):
make oracle              # build the C++ ground-truth dumper
make oracle-data         # regenerate Flow* dumps into oracle_data/, ~10 min
make parity              # parity suites + regenerate docs/PARITY.md, ~2 min fast / ~30 min with slow
make bench               # speedup protocol (refuses busy GPUs - shared server)

# Tier 3 - additionally needs the internal auto_LiRPA tree (see Makefile comment
# for the public-GitHub fallback):
make nncs-deps           # install auto_LiRPA into the venv
uv run python examples/nncs_demo.py   # 256-cell closed-loop NNCS demo, ~1 min
```

Quick sanity run (vanderpol, symbolic remainders, strict mode):

```python
import torch
from flowstar_gpu.config import Settings
from flowstar_gpu.flowpipe import reach

settings = Settings(step=0.02, order=5, sr_queue=100, mode="strict", device="cuda")
result = reach(["y", "(1 - x^2) * y - x", "1"], ["x", "y", "t"],
               torch.tensor([[[1.1, 1.4], [2.35, 2.45], [0.0, 0.0]]]).double(),
               time_horizon=10.0, settings=settings)
print(result.steps_completed.tolist(), result.status.tolist())   # -> [500] [2]  (2 = DONE)
# ~18 s wall at B=1 (measured 17.6 / 18.2 s, V100, 2026-08-01; supersedes the
# "~45 s" recorded at the 2026-07-27 docs pass). B=1 is the GPU worst case;
# batching is the point - see docs/BENCHMARKS.md.
```

## Layout

| Path | Contents |
|---|---|
| `src/flowstar_gpu/` | the library (module map in `docs/ARCHITECTURE.md`) |
| `integrations/crown_reach/gpu_driver.py` | the production entry point: the CROWN-Reach NNCS loop on this engine (`--engine sparse`) |
| `benchmarks/` | the new ARCH-COMP-style AINNCS suite (11 runnable + 1 open challenge) + `_tools/` (config generation, remainder-estimation recipe, report builder) |
| `tools/oracle/` | C++ ground-truth dumper linking Flow*'s `libflowstar.a` |
| `tests/{unit,properties,parity,soundness,adversarial}/` | the five suites (policies in `docs/DEVELOPMENT.md` §5) |
| `tests/golden/` | committed oracle slices (regenerable via `make golden`) |
| `bench/` | speedup protocol + published results + profilers + the order×B tradeoff matrix |
| `examples/` | runnable demos (NNCS closed loop) |
| `scripts/` | oracle-data generation, golden extraction, parity report, GPU-idle gate |
| `docs/` | **start here: DEFECTS** (every defect + what to do) and **TROUBLESHOOTING** (message-first triage + the failure taxonomy). engine: DEVELOPMENT / ARCHITECTURE / ALGORITHM / GOTCHAS / OPTIMIZATION / PARITY / BENCHMARKS / REPRODUCE / PROGRESS / PLAN (historical). benchmark suite: BENCHMARK_PLAN / BENCHMARK_PROPOSAL / **CONFIG_FORMAT** (the YAML both engines read). upstream bug reports: UPSTREAM_AUTOLIRPA_BUG{,2_SAME_SLOPE,3_DTYPE} |
