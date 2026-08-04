# Xiangru complete-Q3 versus our Torch sparse TM: code audit

The executed Xiangru checkout is clean `27d29050a5f214b56f211ca9cb411e734ed80230`.
Our executed core is clean base `308b735ac577cfea39172976a4c08716f1e54d2f`;
this branch changes only evidence/docs/comparators, not solver code.  Fresh evidence
abbreviations below are:

- **Q3-B48**: `outputs/native_reproduction_no_adapters/20260804T081205Z/xiangru/
  x3_complete_q3_b48_native_workspace/s3r_q3_b48_rep1.json` (T=20);
- **Q3-B12/B24**: corresponding `x4_*_workspace/*.json` (T=15/T=18);
- **Torch-H10**: `.../torch/native_order4_vdp_h10_artifacts/` (best adaptive
  reached T=6.049038, then wall cap);
- **X1**: `.../xiangru/x1_crownreach_flowstar_full_rootless/` (T=20).

Line references are to those two executed SHAs.  “High” confidence means directly
read code plus fresh output; “medium” means the mechanism is direct but causal
performance attribution has no matched experiment.

## Required axes

| axis | Xiangru 27d code | our 308b code | evidence-based conclusion | confidence / fresh evidence |
|---|---|---|---|---|
| 1. target problem | `run_s0_tora_static_partition_sweep.py:327-344` selects DR13/Q3 for the TORA closed-loop lane; `:407-451` refreshes held control every ten 0.1 segments; `tensor_fixed_basis.py:760-837` is the TORA RHS | `flowstar_raw_remainder_compat_h10_right_map_centering.py:236-323` fixes the 2-D autonomous Van der Pol box and calls the sparse Flow*-style step; no controller | These are not the same mathematical workload. Xiangru is a sample-held 4-state/1-control TORA NNCS with property; ours is plant-only VDP plus Flow* compatibility reference. | high; Q3-B48 vs Torch-H10 |
| 2. state/uncertainty representation | `generic_fixed_basis.py:30-54` defines immutable exponent support; `:79-108` gives 13-term DR support versus complete total-degree support. Q3 with six variables has 84 slots. | `polynomial.py:132-162` stores only present monomials in a Python dictionary; `:235-250` truncates dynamically by total order. `TaylorModel` carries one independent interval remainder (`taylor_model.py:220-284`). | Q3 is dense over a fixed complete support; ours is sparse/dynamic and not a fixed 84-slot basis. Q3 can retain correlations that DR13 discards, but “complete Q3” does not imply smaller remainder after other operations. | high; Q3-B48/Q3-B12/B24 and Torch-H10 |
| 3. local time/order | `tensor_fixed_basis.py:88-113` records time degree; `:149-162` integrates exponent 0 (local time) and routes overflow; `generic_fixed_basis.py:94-108` counts local time in total degree. | `flowpipe.py:1750-1783` integrates explicit `tau_index` then truncates all exponents by requested order; `polynomial.py:292-301` implements integration. | Both retain explicit local time, but Q3 uses a predeclared total-degree-3 support and ours constructs up to order 4 per current VDP config. Equal “order” labels would not imply equal support. | high; Q3-B48 and Torch-H10 |
| 4. Picard construction | `tensor_fixed_basis.py:838-858` performs exactly two polynomial-only Picard rounds; `:874-906` builds the interval image and ten default remainder rounds. | `flowpipe.py:1750-1783` defaults polynomial Picard iterations to `order` (four here), dropping polynomial overflow for later validation; `:1786+` validates candidate remainders. | Q3's “2 polynomial + 10 remainder” algorithm is structurally different from our four polynomial iterations plus adaptive target-remainder validation. Iteration counts cannot be normalized into one order number. | high; Q3-B48 and Torch-H10 |
| 5. remainder | `tensor_fixed_basis.py:526-580` routes integration overflow and integrated input remainder; `:605-678` includes polynomial overflow and all polynomial/remainder products; `:696-725` bounds sine truncation; `:884-926` adds polynomial roundoff difference to each refinement candidate. | `taylor_model.py:47-91,270-284` includes polynomial×remainder, remainder×remainder and truncated polynomial range; `flowpipe.py:2529-2890` reconstructs the selected Flow* raw-remainder compatibility residual and fails subset tests. | Matching support would still leave different range evaluation, sine composition, roundoff accounting, seed/candidate and compatibility-residual construction; a wider remainder after support matching is plausible but is not quantified here. | high on code, medium on causal width; Q3-B48 and Torch-H10 |
| 6. cross-step propagation/reset | `tensor_closed_loop.py:332-368` projects exact time to an affine controller boundary; `:429-487` advances symbolic parameterization then composes the segment; `:511-517` retains current polynomial/parameterization/symbolic state. | `flowpipe.py:1304-1487` composes the endpoint into the prior right map, optionally centers it and normalizes center/scales; `:3920-3969` selects insertion/queue/box reset. The executed script carries `seg.reset_tm` and `flowstar_normal_state` (`...h10...py:421-434`). | Both avoid an unconditional endpoint box reset in the executed modes, but Xiangru uses a TORA-specific affine/symbolic parameterization and ours uses Flow*-style normalized insertion/right-map state. | high; Q3-B48 and Torch-H10 |
| 7. validation/failure | `run_s0...py:466-526` requires certificate, full-tube property and finiteness for every leaf; `:614-634` stops on any rejected leaf; `:649-690` keeps replay validity, first failure and certified horizon separate. | `...h10...py:347-420` requires native step status, finite tube/final box and reset; a failed step stops. `flowpipe.py:2782-2815` treats exceptions/nonfinite/subset failure as failure. | Both executed lanes fail closed at leaf/step scope. Q3-B12/B24 preserve first failing segments; Torch-H10 preserves wall-cap and validation failures. | high; all three fresh paths |
| 8. interval strictness | `controller_outward.py:32-96` uses host `nextafter` for controller composition, but dynamics `remainder_backends.py:39-116` uses ordinary Torch add/mul/min/max and `:147-198` ordinary sin/cos endpoints. | `interval.py:36-41,135-192` applies `torch.nextafter` after interval add/mul/powers; polynomial coefficient convolution itself is ordinary floating point (`polynomial.py:223-245`). | Xiangru's *controller* composition is explicitly outward, not the full Q3 dynamics. Our interval shell nudges outward more broadly, but neither fresh path has a closed end-to-end formal-rounding proof. Q3 is empirical; ours remains unknown. | high; Q3-B48 and Torch-H10 |
| 9. controller | `autolirpa_controller_worker.py:17-52` builds a persistent normalized float64 auto_LiRPA graph; `:54-103` times bound and outward composition; `run_s0...py:407-451` calls it once per control period. | The executed VDP entrypoint has no NN controller. `flowpipe_step_flowstar_style_adaptive` accepts optional box/affine control (`flowpipe.py:1735-1747`) but it is unused here. | No controller-runtime, slope, normalization or transfer comparison is valid between these fresh runs. X1 and Q3 use the same saved controller bytes, but different native controller/plant routes. | high; Q3-B48, X1, Torch-H10 |
| 10. tensorization/GPU | `tensor_fixed_basis.py:88-295` precomputes product/integration/composition routes; coefficients are `[batch, outputs, slots]` (`:366-374`). `tensor_compiled.py:173-235` builds reusable CUDA graphs with dynamic batch; `run_s0...py:353-390` warms/compiles and synchronizes. | Sparse products iterate Python dictionaries (`polynomial.py:223-245`) and interval evaluation loops monomials (`:316-329`). The executed script does not select CUDA or compile a graph. | Xiangru's dynamics acceleration mechanism is visibly batch/route tensorization plus Torch compilation. The code also performs replay/validation, so speed cannot be attributed simply to “less work”; no matched speedup is computed. | high on mechanism, medium on performance attribution; Q3-B48 and Torch-H10 |
| 11. generality/hard-coding | `tensor_compiled.py:188-194` requires six variables and CUDA for compilation; `tensor_fixed_basis.py:748-837` hard-codes the five-output TORA RHS; `controller_outward.py:10` fixes four state dimensions. The support algebra itself is generic. | `Polynomial`/`TaylorModel` are generic in variable count, while the executed experiment fixes 2-D VDP bounds and imports a VDP ODE (`...h10...py:23-40,251-323`). | Xiangru's route/support primitives are reusable, but the successful closed-loop engine is TORA/4-state/5-output/six-variable specific. Our core is more generic than this experiment, but its authoritative H10 lane is VDP-specific. | high; Q3-B48 and Torch-H10 |
| 12. timing boundaries | `run_s0...py:374-392` synchronizes and excludes compile/warm; `:425-464` separates controller/default dynamics; `:642-715` reports validation-excluded solver wall and total wall. | `...h10...py:256-278,478-522` reports per-lane wall including Python validation/sample checks; `:907-948` runs four lanes sequentially. Driver wall is separate. | Q3 controller/dynamics/validation/compile fields and Torch per-lane/process wall are different boundaries. RTX5090 reference, V100 fresh Q3, CPU Torch and CPU Flow* timings cannot form a speedup. | high; command.json plus Q3-B48/Torch-H10 |

## The three central questions

### Why complete-Q3 survives longer than restricted DiffReach support

The direct code difference is 84 complete degree-≤3 monomials versus DR13's
constant, six linears and six `time×variable` terms.  Product and integration
overflow is immediately intervalized when its exponent has no slot
(`tensor_fixed_basis.py:135-162,605-648`).  Complete-Q3 therefore postpones much
more dependency loss.  Fresh B48 completes T=20, while the exact U0 CPU artifact
has only 66.22% initial shrink flags and exploding ranges.  Because the U0 GPU path
does not reach step one on this V100 and its controller/backend details also differ,
the support explanation is a strong mechanism, not a clean single-variable causal
experiment.

### Why equal support would not guarantee Flow*-sized remainders

Support is only one term.  Natural interval monomial evaluation, sine remainder,
ordinary floating-point roundoff, candidate seeds, accepted-refinement policy,
exact-time composition and reset all remain different.  Flow* also has a known
scalar-affine endpoint/collapsed-path blocker, so its apparent width cannot be used
as a formal target until that gate is closed.  No matched B48 Flow* raw internal
remainder exists here; a numerical tightness claim would overreach the evidence.

### What produces the GPU speed mechanism

The code precomputes static route tensors, batches leaves on the first axis, caches
factor intervals and compiles reusable Picard/composition graphs.  CUDA
synchronization brackets the measured regions.  Validation is not simply absent:
every leaf checks certificate/property/finiteness and selected segments get CPU
replay.  However, ordinary float64 interval operations are less rigorous than a
complete directed-rounding implementation.  Thus “tensorized and compiled” is
supported; “faster because it validates less” and any numeric speedup are not.

## Migration boundary

Reasonable later candidates for our sparse solver are immutable exponent indexing,
precomputed multiply/integrate routes, batch-first coefficient tensors, cached
factor intervals, explicit CUDA synchronization and Xiangru's leafwise fail-closed
status/timing schema.  The TORA RHS, fixed 4-state/1-control layout, six-variable
compiled assumption, 84-slot support and controller-period machinery cannot be
copied as a generic solution.  No migration is performed on this reproduction
branch.
