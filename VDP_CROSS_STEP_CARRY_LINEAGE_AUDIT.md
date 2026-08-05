# VDP order-4 cross-step transition and carry-lineage audit

## Plain-language answers

1. **First behavior-relevant divergence.** The native schedules agree through accepted step 11. At `t=0.18187433604506256`, both propose `h=0.019615177354506262`; the authoritative Torch lane accepts it, while stock Flowstar rejects it and accepts `0.009807588677253131`. This is not a serialization tolerance issue. The runs are already structurally unmatched at step 0, and the schedule split occurs where Torch's preregistered proactive four-leaf polynomial-truncation range is active while stock Flowstar uses its native range path.
2. **What call 44 contains.** Stable call hash `b7613d231be82bea5afc477540342cfa04ee17a6448d7bfb5921dc70ae341a9d` identifies the validation-attempt-1 raw-remainder y expression `-x²y`, not merely the ordinal 44. Its 1,141 discarded multiplication routes collapse to 145 exponent groups. Every route comes from the terminal Picard candidate's current-state polynomial. The left parents are the retained/cutoff result of the immediately preceding `x*x`; the right parents are candidate-y terms.
3. **Where dependency becomes interval.** The first trajectory-wide observed cross-step interval enclosure is the normalized-insertion/right-map transition after accepted step 0: output-remainder widths are `2.5000060000000046e-6` (x) and `5.0117054490339935e-5` (y). The first non-underflow insertion degree truncation is after accepted step 1. The terminal roots cross the same kind of boundary after step 307; call 44 then intervalizes its own discarded polynomial to `[-0.029997247026804494, 0.02187259686437867]`.
4. **Verdict: Case C, evidence still insufficient for a carry root-cause claim.** No local Torch implementation bug was found. The evidence shows an earlier Torch intervalization boundary and a material stock Flowstar symbolic-remainder queue, but the executions do not satisfy one fully matched numerical contract. The first native schedule split is confounded by the intentional range-policy difference, and the stock scheduler hook does not expose the exact rejected Picard fields needed for a common-basis causal comparison.
5. **Next smallest action.** Add one deeper observation-only Flowstar hook inside the exact symbolic `advance_adaptive_stepsize` overload, immediately around `insert_ctrunc_normal`, `Picard_ctrunc_normal`, and the subset test. Export the pre/post insertion polynomial, J/Phi_L queue references, discarded terms, candidate/image remainders, and accepted predicate in a tested common basis for only the last common state and first divergent candidate.
6. **What this audit cannot claim.** It cannot claim Torch/Flowstar end-to-end parity, that symbolic carry alone causes the T=6.397 failure, that Flowstar is an oracle, that any per-term interval contribution sums to the terminal violation, or that T=10 is fixed.

## Scope and provenance

Work used an isolated worktree and branch `codex/vdp-cross-step-carry-lineage-audit-20260806`; the original dirty Torch and Flowstar worktrees were not reset, stashed, deleted, or incorporated. Root `AGENTS.md` and the requested starting `handoff.md` were absent and are recorded as missing sources rather than invented.

The relevant lineage is:

| Role | Commit |
|---|---|
| H1 formal numerical source | `a1fb3527bb7c12ce23aa2fb49d66f6380c463c90` |
| H1 packaging/report source | `2e4507220a631a21dbe5227a7f9a5201948aedde` |
| audited starting HEAD | `455146df23940caa6f168877ffe6ec6f508c43a4` |
| stock Flowstar | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` |
| frozen R4 checkpoint | `dcb8f646d45c9742e0cff23fea596c12e53d8ccd00d1544f70564a44a7463420` |

The provenance audit hashes numerical sources, runner/config/checkpoint/manifests, formal and packaging states, remote fetch/ls-remote output, repository state, Flowstar state, environment, and raw commands under `outputs/vdp_cross_step_carry_lineage_20260806/provenance/`.

## Frozen baseline closure

The previous 14 lanes were discovered from the actual runner rather than reconstructed from prose and replayed at the clean formal source commit. All values match the packaged evidence:

- A0 natural;
- A1 polynomial truncation;
- A2 integration overflow;
- A3 polynomial/remainder products;
- A4 truncation plus overflow;
- A5 all registered contexts;
- A6 maximum subdivision;
- D0 natural;
- D1 subdivision;
- D2 registered Horner;
- D3 subdivision plus Horner;
- D4 each of three fixed variable orders.

The frozen state is exactly `t_pre=6.397083942944808`, `h=0.003623635847674574`, candidate coefficient SHA256 `bc1433d0d3c89339fca6091e41c0a6667d70c92d2dd4e35ae8b14236d131863c`, and support SHA256 `d0aa354b9057267556d5bb3bc09a36ed4162b36fb44588b0b930dd9e935041e9`. D3 still rejects with y margin `-1.5859969428028492e-5`. No repair, relaxed predicate, changed order, enlarged remainder, lower `h_min`, sparse fallback, or external endpoint substitution was used.

The fresh observation-only T=6.5 run has full checkpoint SHA256 `54185608f6f1920517a30e2fc37888dd8d66fbb78f13fe0037d727632c544a0e`, rather than the frozen artifact's `dcb8f646...`, because command/provenance packaging is part of the full hash. This is not numerical drift: current-state, `tmv_pre`, and `tmv_right` TMVector hashes are respectively `17de6d46dae3f3c1123627d507756741d02ebcb0f2dbda7754b4a6134563bc5e`, `efe776ac16eedc29b5582e7de979f5442efa79ed3ef7092f28484089a49b04ad`, and `c721ccf4c02099afd7064a79dd3235759f453df6c8315d2a4e8745ecd7ed3bb3` in both artifacts. The lineage replay now requires the exact T=6.5 manifest and rejects the numerically equal but differently packaged T10 checkpoint.

## Matched-contract result

The physical ODE, x/y initial box, order 4, local-time domain after index mapping, `h_min=0.002`, `h_max=0.1`, half-on-reject, 1.1-on-accept, candidate radius `1e-4`, cutoff `1e-10`, endpoint semantics, and segment semantics match.

The end-to-end numerical contract does **not** match:

- Flowstar stores time plus three normalized generators with local time first; Torch stores x/y generators with tau last.
- Torch uses proactive four-leaf subdivision for the authoritative polynomial-truncation range; stock Flowstar does not.
- stock Flowstar constructs `Symbolic_Remainder sr(initialSet, 100)`; the frozen Torch state has `symbolic_queue_present=false`.
- interval backends, fully exposed Picard validation fields, normalized insertion internals, stored tube objects, and state dimensions differ.

For that reason coefficient comparison fails closed unless a tested algebraic common-basis transform exists. None was introduced in this audit. See `contract/matched_contract.json`, `matched_contract.md`, and `field_map.md`.

## Observation-only transition traces

### Torch

`TransitionTraceWriter` records decimal `max_digits10` and hexfloat for times, coefficients, centers/scales, intervals, support, lifecycle stages, and accepted-attempt margins. It never changes a numerical object. Missing per-term attribution is null with a reason; call-44 terms are emitted only by the dedicated lineage replay.

Short on/off and two-run checks establish byte-deterministic data files and unchanged status, horizon, steps, rejected-attempt count, endpoint, and segment. Tests cover schema fields, canonical exponents, number round-trip, duplicates, interval ordering, missing fields, deterministic serialization, non-mutation, and accepted-margin consistency.

The fresh T=6.5 trace reproduces the expected fail-closed result in 307 accepted steps and 48 rejected attempts: completed horizon `6.397083942944808`, terminal attempted `h=0.003623635847674574`, x/y margins `9.963763341523255e-5` and `-1.99995911680722e-5`, no fallback/repair/sample violation, 4,312 range subdivisions, and 17,248 leaf evaluations. The trace contains 355 attempt rows, 6,461 transitions, 48,158 polynomial-term rows, and 5,530 remainder rows. Its captured exit code is 1 because minimum-step failure is the expected audited numerical outcome, not a command crash.

### Flowstar

The benchmark source is the exact stock commit. The only source patch adds read-only term accessors and a post-advance JSONL hook; it is always called **“stock Flowstar + observation-only instrumentation”**, never stock Flowstar itself. GCC 15 required `-fpermissive` for a pre-existing stock const-correctness error; no numerical source fix was made.

The final instrumentation exposes only objects valid at the scheduler hook: `current.tmv`/`result.tmv` as stored right maps and `result.tmvPre` as the accepted raw Picard/next pre map. Pre-scaling insertion, composed pre-state, Torch-style normalized reset, inner rejected margins, and partitioned discarded terms remain null rather than being replaced by look-alike objects.

The uninstrumented and instrumented binaries both exit 0, print the same 290-segment schedule, finish identically, and produce byte-identical x/y plot files:

- x SHA256 `63facf7f12f58c0e034942e7d568bba2bea62cf37f2027332b9a1fd61f6c4bd4`;
- y SHA256 `c734e3427ccea50d4c373ce69df7e887046d01732a839519ae0f6550522d6533`.

Flowstar retry rows are explicitly marked as scheduler-derived from previous accepted `h*1.1` and repeated half steps; they are not mislabeled as observed inner Picard attempts.

## First divergences

The streaming comparator synchronizes native attempts and preserves raw mismatches. It checks absolute tolerances `0`, `1e-15`, `1e-12`, and `1e-9`; the accepted-sequence divergence persists at all four.

### First trace-visible field divergence

At accepted step 0, the local bases/available lifecycle objects already differ. Torch records `[u0,u1]` or `[u0,u1,tau]`; Flowstar records `[tau,r0,r1,r2]` and cannot export the physical composed pre-state at the chosen hook. This is a structural semantic/representation difference but does not itself change the first decision.

### First native schedule divergence

| Field | Torch | stock Flowstar |
|---|---:|---:|
| last common accepted step | 11 | 11 |
| `t_pre` | `0.18187433604506256` | `0.18187433604506256` |
| proposed h | `0.019615177354506262` | `0.019615177354506262` |
| decision | accept | reject |
| next decision | n/a | accept `0.009807588677253131` |

Torch source selects the named range policy in `batched_dense_tm.py` and its adaptive scheduler in `flowpipe.py`. Stock Flowstar reduces the step within the symbolic adaptive advance in `Continuous.cpp`. The physical ODE permits either sound enclosure to make a different containment decision; because the range policies/backends are not matched and stock rejected-attempt internals are missing, the audit cannot declare one implementation mathematically wrong.

The precise source anchors are Torch `src/torch_tm_flowpipe/batched_dense_tm.py:1114` for policy-selected term ranging and `src/torch_tm_flowpipe/flowpipe.py:4185` for the half/grow scheduler. At exact stock Flowstar commit `b85a321...`, the relevant symbolic overload begins at `flowstar-toolbox/Continuous.cpp:2715`; J/Phi_L propagation is at 2753–2768, normalized insertion at 2781/2843, Picard construction at 2940, and the first subset decision at 2962. The fail-closed coordinate guard is `experiments/compare_vdp_transition_traces.py:179`, while stable call-44 capture starts from `experiments/trace_vdp_call44_lineage.py:28` and materializes the replay at line 481.

No shadow replay is mixed into the native result. The comparator windows and configuration are under `comparison/`.

## Terminal call-44 lineage

The stable identity combines:

- stage `validation_attempt_1_raw_remainder_compat`;
- y component;
- operation `negative_x_squared_times_y`;
- basis fingerprint `8835901234c525e7d67983af797266b78cb12999d9290c10ad7878765f386634`;
- effective max degree 3;
- left/right input hashes;
- route coefficient and exponent hashes;
- frozen checkpoint/candidate/support hashes.

The ordinal `range_call_index=44` is secondary. It maps to multiplication operation 11, the second multiplication in `x*x*y`; operation 10 supplies `x*x`.

The DAG contains:

- candidate-x and candidate-y root terms;
- every retained `x*x` route;
- equal-exponent scatter aggregates;
- each cutoff keep/zero result;
- all 1,141 discarded `x²*y` routes;
- all 145 discarded exponent aggregates.

Coverage is 1,141/1,141, with no unresolved parent ID. Direct reconstruction produces exactly the captured and historical D3 selected interval `[-0.029997247026804494,0.02187259686437867]` (width `0.05186984389118317`, maximum error 0).

The structural source class is `current_state_polynomial`. That does not mean history is irrelevant: each terminal candidate root is built from the fresh affine state produced after boundary 307, after older insertion/right-map dependence was enclosed. The checkpoint's physical output-remainder widths before rescaling are `1.3258522458175754` and `3.3898122731383684`; its stored normalized right-map remainder widths are `1.7535966119029434` and `1.9524801704183852`. Insertion degree-truncation widths are `3.951176031054346e-9` and `2.7567676105938294e-6`.

The path to the terminal y decision is:

```text
call44 discarded polynomial range
→ raw RHS y remainder
→ integration by tau in [0,h]
→ base remainder + polynomial-difference enclosure
→ self-map image y
→ terminal subset margin
```

This is structural lineage, not a causal allocation. Subdivision, Horner selection, interval multiplication, and hull operations are non-additive; the audit deliberately does not claim that individual term intervals sum to the final violation.

Flowstar has a corresponding `Picard_ctrunc_normal` stage in its symbolic adaptive advance, but no demonstrated one-to-one call/basis identity. No “Flowstar call 44” was fabricated.

## Case decision and blocker

**Case C is selected.** The audit aligns native behavior through accepted step 11, explains the first schedule split at the level of the unmatched policy/decision, and fully explains Torch call 44 structurally. It still lacks the stock fields and common-basis transform needed to prove that an earlier Torch intervalization rather than another unmatched local semantic causes the later T=6.397 rejection.

Already excluded:

- JSON order/formatting and ordinary tolerance;
- candidate/support drift at the frozen terminal;
- extra subdivision depth/leaves and more Horner permutations;
- integration-overflow range as the decisive local cause;
- endpoint repair, sampling tightening, external substitution, hidden sparse fallback, nonfinite behavior, and wall timeout;
- a trace-visible local mapping/degree/double-add omission in call 44.

Still missing:

- stock rejected-attempt candidate and image remainder intervals at the exact first divergent h;
- pre/post `insert_ctrunc_normal` polynomials and discarded terms;
- J/Phi_L/scalar queue lineage at the same state;
- tested conversion to a common x/y/time basis with center/scale equality;
- the same fields at a forced common h after the last common state.

Therefore this turn makes no numerical fix and proposes no new carry representation. A representation experiment would be premature until that observation gap is closed.

## Verification and evidence map

The required raw commands, streams, exit codes, environment, and hashes are preserved under:

- `provenance/`: start states, remotes, source/evidence hashes, exact frozen-lane reruns;
- `contract/`: machine-readable match table and field map;
- `torch_trace/`: deterministic short traces, prefix trace, and fresh T=6.5 trace;
- `flowstar_trace/`: stock run, final instrumented raw trace, schema split, and equivalence gate;
- `comparison/`: all required first-divergence/window/config files;
- `lineage/`: identity, route table, DAG, intervalization events, coverage, and reconstruction;
- `tests/`: installation, baseline/full/focused pytest logs and commands;
- `manifest.sha256`: final sorted SHA256 inventory.

The focused audit suite passes 8 tests in 7.48 seconds. The final repository gate passes **441 tests with 2 skipped in 63.34 seconds** under Python 3.11.15 and PyTorch 2.5.1+cu121; raw stdout, stderr, exit code, command, clean source commit `7a40428bfdf018a0995daf5f9777afc9c807fe88`, CUDA availability, and CPU count are in `tests/focused_final/` and `tests/final_pytest/`. The final fresh T=6.5 trace is an observation-only reproduction of the known authoritative lane; this report does not reinterpret its fail-closed exit 1 as an execution error.
