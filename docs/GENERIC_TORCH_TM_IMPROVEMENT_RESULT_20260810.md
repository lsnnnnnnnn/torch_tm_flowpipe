# Generic Torch TM improvement result

Date: 2026-08-10  
Candidate: F1 complete polynomial normalized carry  
Decision: **CANDIDATE_REJECTED**

## What changed

The experimental reset mode
`normalized_insertion_complete_polynomial` changes exactly one variable: after
a validated endpoint-time substitution, it carries the complete retained
endpoint polynomial and its validated interval remainder into the next step.
The baseline instead begins the next step from the normalized affine reset.
No term is silently range-reduced: diagnostics report the coefficient SHA,
retained term count, maximum degree, remainder width, and zero intervalized
retained terms.

The primitive is generic over state count, polynomial variables, complete
support/order, domain coordinates, dtype/device, and batch. Sparse `TMVector`
is the current adaptive runner path; dense `[B, state, slot]` carry is tested at
B=1/8/64 and benchmarked through B=512 on CPU and V100. It clones coefficient,
remainder, domain, and ledger tensors exactly. It uses no VDP state index in the
carry kernel. Work and storage are linear in `B * state * slots`.

## Soundness argument

Endpoint substitution preserves a validated Taylor model. Cloning every
retained coefficient and the complete interval remainder produces the same
set, so it cannot exclude a prior endpoint. The implementation rejects empty,
non-finite, dimension/domain-inconsistent, or above-order inputs. Unit tests
sample generic correlated polynomials, require exact remainder equality, and
require batch-permutation equivariance. The carry itself is bit-preserving;
ordinary dense Torch arithmetic around it retains its separately declared
soundness class.

## One-step paired gate

Each h in `.1, .05, .025, .0125, .005, .002` starts independently from the
authoritative initial box and attempts exactly one fixed-h segment. Because
carry is applied only after validation, baseline and candidate have identical
segment and endpoint coefficient SHA256 values, remainder images, subset
margins, status, work, and decision for every h. At every accepted h the
candidate's next-step initial coefficient SHA is exactly the raw endpoint SHA.
This is a zero-mismatch implementation gate, not a long-horizon benefit claim.

## Required horizon ladder

Every row is an independent request from t=0 using O4, target remainder
`[-1e-4,1e-4]^2`, cutoff `1e-10`, h range `[.002,.1]`, the same raw-remainder
validator, and the same proactive depth-1 polynomial-truncation range policy.

| requested T | completion | highest validated T | accepted / rejected attempts | runtime s |
|---:|---|---:|---:|---:|
| 0.1 | partial | `0.04345468750000001` | 4 / 7 | `2.9082528073340654` |
| 0.5 | partial | `0.04345468750000001` | 4 / 7 | `2.971852839924395` |
| 1.0 | partial | `0.04345468750000001` | 4 / 7 | `2.9807011717930436` |
| 4.0 | partial | `0.04345468750000001` | 4 / 7 | `3.0237683802843094` |
| 6.0 | partial | `0.04345468750000001` | 4 / 7 | `3.085928037762642` |
| 6.5 | partial | `0.04345468750000001` | 4 / 7 | `2.95764989964664` |
| 7.5 | partial | `0.04345468750000001` | 4 / 7 | `2.9240406062453985` |
| 10.0 fresh | partial | `0.04345468750000001` | 4 / 7 | `2.9779219031333923` |

The terminal attempted `h=0.0022876562500000006` has minimum target margins
`9.542574395200346e-5` in x and `-1.0996808713007844e-6` in y. Halving it would
fall below h_min. The last finite raw endpoint is
x `[1.2004151011978565, 1.502803775116484]`,
y `[2.17663116337463, 2.3675732461975776]`; the last segment tube is
x `[1.195489597378759, 1.502995131280772]`,
y `[2.1763980788788215, 2.3744968319364004]`. Sampling found no violation.
There was no fallback, repair, endpoint tightening, or sub-minimum publication.

The candidate therefore fails both short-certificate preservation and the
primary `+0.5` horizon threshold. It is not promoted and is not the default.

## Batch and timing boundary

The actual-independent-partition short fixed-support lane and the complete-O4
single-step-plus-carry kernel both ran B=1/8/64/256/512 on CPU and pinned V100.
The complete candidate's adaptive outer scheduler remains batch-one; its batch
rows are explicitly kernel diagnostics, not multi-step certificates.

| lane / device | B1 warm min s | B64 | B512 | B512 peak bytes | result |
|---|---:|---:|---:|---:|---|
| fixed support CPU, 10 steps | `0.4772511` | `0.5164393` | `0.6959472` | `408989696` | completed T=.1 |
| fixed support V100, 10 steps | `1.2503044` | `1.2641291` | `1.2596511` | `27262976` | completed T=.1 |
| complete O4 + carry CPU, one step | `0.1168205` | `1.2420704` | `9.3192035` | `501600256` | validated, exact carry |
| complete O4 + carry V100, one step | `0.3070624` | `1.5533992` | `10.4725577` | `54525952` | validated, exact carry |

All values include required range/validation and the aggregate host inclusion
gate; CUDA is synchronized. The V100 is slower at every measured point because
the current O4 routes and audit path synchronize frequently. No GPU speedup or
Pareto claim is valid. Ordinary CPU/CUDA rows are `empirically sampled only`;
the complete baseline's sparse interval path remains separately classified
`formally outward by construction`.

## Interpretation and next operation

Exact full-polynomial carry preserves correlation at one boundary but lets the
same original dependency variables and accumulated validated remainder enter
subsequent nonlinear Picard products without Flow*'s structured
preconditioning/symbolic-remainder decomposition. It therefore expands the
next-step raw y image almost immediately. The evidence-supported next step is
not another full carry or range-order variant: retain a bounded fixed-shape set
of the terminal `integration_overflow` and `polynomial_truncation` terms as
structured symbols across normalization, with deterministic sound collapse,
then replay the frozen terminal pre-state before any horizon sweep.

## Evidence

Proposal: `docs/proposals/GENERIC_CARRY_CANDIDATE_20260810.md`. Raw results:
`outputs/mainline_realignment_20260810/20260810T025910Z/04_generic_carry_candidate/`.
Batch results:
`outputs/mainline_realignment_20260810/20260810T025910Z/05_batch_scaling/`.

