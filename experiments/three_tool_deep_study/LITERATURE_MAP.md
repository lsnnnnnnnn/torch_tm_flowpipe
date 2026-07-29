# Literature map

## Material availability audit

The requested attachment filenames were searched exactly under the workspace
and across the readable server filesystem on 2026-07-29.  None was present.
The exact 16-file list and the resulting evidence boundary are in
`MATERIALS_MISSING.md`.

Accordingly, this map does **not** claim attachment-specific page numbers or
that any missing PDF was read.  The paper entries below use their public
landing-page metadata.  Course-attachment metadata explicitly supplied for
this study is listed separately from the public course schedule; the two
numbering systems are not treated as interchangeable.

## Scope rule

The primary evidence for this repository is the executed plant-only
Torch/Flow*/DiffReach study.  Neural-network robustness papers explain possible
controller-bound extensions; they do not validate an ODE flowpipe, establish
Taylor-model remainder soundness, or substitute for the Flow* C++ traces and
analytic containment gates.

Relation labels used below:

- **direct** — defines or evaluates the plant reachability/Taylor-model
  operation under study;
- **component-direct** — directly addresses a reusable component such as a
  polynomial range query, but not the complete flowpipe;
- **indirect/controller** — becomes relevant only when a neural controller or
  learned dynamics is composed with the plant;
- **methodology** — informs fair verification benchmarking, attacks, or
  training, but is not evidence for these plant enclosures.

## Direct plant and Taylor-model evidence

| source | contribution | relation and use here |
|---|---|---|
| Chen, Ábrahám, Sankaranarayanan, [“Flow*: An Analyzer for Non-Linear Hybrid Systems,” CAV 2013](https://home.cs.colorado.edu/~srirams/papers/cav2013-flowstar.html) | Taylor-model flowpipe construction for nonlinear continuous/hybrid systems, with adaptive step/order capabilities. | **Direct.** Establishes the intended tool family. The present result is tied more narrowly to audited commit `b85a321`/patched `fa39f7a` and the exact Riccati/Van der Pol executions. |
| Berz and Makino, [“Verified Integration of ODEs and Flows Using Differential Algebraic Methods on High-Order Taylor Models”](https://doi.org/10.1023/A:1024467732637) | High-order Taylor polynomial plus rigorous remainder for verified ODE integration and dependency control. | **Direct.** Supports the mathematical role of higher-order dependency retention; it does not prove this repository’s floating-point implementation. |
| Revol, Makino, Berz, [“Taylor models and floating-point arithmetic: proof that arithmetic operations are validated in COSY”](https://perso.ens-lyon.fr/nathalie.revol/publis/RevolBerzMakino2005-Taylor-COSY.pdf) | Proves containment for a particular floating-point Taylor-model implementation. | **Direct boundary.** Shows why a mathematical TM algorithm is insufficient without implementation-level roundoff treatment. It is not automatically transferable to Torch float64. |
| Bünger, [“Preconditioning of Taylor models, implementation and test cases”](https://doi.org/10.1587/nolta.12.2) | Reviews parallelepiped, QR, curvilinear, and identity preconditioning for verified ODE integration. | **Direct.** Positions the observed reset/preconditioning effects and the need to distinguish native dependency carry from box controls. |
| Shen and Chou, [“Parallel Differentiable Reachability for Learning and Planning with Certified Neural Dynamics and Controllers”](https://arxiv.org/abs/2605.25346) | DiffReach: JAX-parallel differentiable Taylor-model reachability with affine dependency and CROWN-style NN bounds. | **Direct for DiffReach.** The study executes repository `dd628eb`; the paper explains the broader neural/controller design, while the benchmark here uses its analytic plant path only. |

## BERN-NN-IBF

Fatnassi et al.,
[“BERN-NN-IBF: Enhancing Neural Network Bound Propagation Through Implicit
Bernstein Form and Optimized Tensor Operations”](https://hpcforge.eng.uci.edu/publication/emsoft24-bern-nn-ibf/emsoft24-bern-nn-ibf.pdf)
introduces a factorized implicit Bernstein representation, polynomial
operations, CUDA coefficient-extrema routines, and quadratic ReLU bounds.

Relation: **component-direct** for polynomial storage/range bounding and
**indirect/controller** for NN propagation.  It is not an ODE integrator and
does not specify Taylor-model truncation-to-remainder, Picard inclusion,
endpoint substitution, or multi-step reset.  The executable decision and local
code audit are in `BERN_FEASIBILITY.md`; the minimal prototype deliberately
tests only the Bernstein range-query idea.

## Requested neural-verification readings

| identifier | method | what it actually establishes | relation to this study |
|---|---|---|---|
| [arXiv:1811.00866](https://arxiv.org/abs/1811.00866) | **CROWN** | Backward linear/quadratic activation relaxations for efficient incomplete robustness certification, including general activations. | **Indirect/controller.** Candidate NN controller bound inside a reachability composition; not a plant TM validator. |
| [arXiv:2103.06624](https://arxiv.org/abs/2103.06624) | **β-CROWN** | Adds optimizable parameters that encode per-neuron split constraints and couples bound propagation with branch-and-bound for complete or incomplete verification. | **Indirect/controller.** β-CROWN is not merely a newer name for CROWN: split constraints and BaB integration are the essential additions. |
| [arXiv:1810.12715](https://arxiv.org/abs/1810.12715) | **IBP** | Trains networks so simple forward interval propagation yields useful verified robustness bounds. | **Indirect/controller/training.** IBP may be a cheap controller abstraction, but its dependency loss is not evidence about common-box plant carry. |
| [arXiv:1711.00851](https://arxiv.org/abs/1711.00851) | **convex outer adversarial polytope** | Uses an LP outer relaxation and its network-shaped dual to train/certify ReLU classifiers. | **Indirect/controller.** Relevant to linear relaxations and verified training, not nonlinear ODE flowpipe construction. |
| [arXiv:1902.08722](https://arxiv.org/abs/1902.08722) | **convex-relaxation barrier** | Unifies a class of LP-relaxed verifiers and empirically studies the limit of even the optimal relaxation in that class. | **Methodology/indirect.** Warns against assuming a tighter implementation removes relaxation limits; says nothing about Flow* remainder caching. |
| [arXiv:1711.07356](https://arxiv.org/abs/1711.07356) | **MILP verification** | Encodes piecewise-linear networks as a mixed-integer program with tight formulations and presolve for complete robustness answers. | **Indirect/controller.** A possible exact controller oracle at much higher cost; no plant integration result. |
| [JMLR 21:42, article 19-468](https://www.jmlr.org/papers/v21/19-468.html) | **branch and bound (BaB)** | Organizes piecewise-linear NN verification around bounding and branching choices and introduces ReLU branching strategies/benchmarks. | **Indirect/controller.** Supplies the complete-verification search structure that β-CROWN accelerates; not a reachable-set reset method. |
| [arXiv:1706.06083](https://arxiv.org/abs/1706.06083) | **PGD/robust optimization** | Frames adversarial training as robust optimization and develops a strong first-order adversary. | **Methodology/training.** PGD finds attacks but is not a sound verifier; analogous trajectory sampling remains sanity/falsification only. |
| [arXiv:2312.16760](https://arxiv.org/abs/2312.16760) | **VNN-COMP 2023** | Defines standardized ONNX/VNN-LIB problems, equal-cost hardware evaluation, precommitted parameters, and competition reporting. | **Methodology.** Supports protocol discipline and explicit capability/unavailable rows. Competition standings cannot rank these plant solvers. |

### Family distinctions

- **CROWN** is efficient bound propagation using activation relaxations.
  **β-CROWN** augments it with optimizable split constraints and becomes a
  strong bounder inside BaB; complete verification comes from exhausting the
  search, not from a single relaxation pass.
- **IBP** propagates componentwise intervals forward.  It is fast and useful
  for verified training but typically loses correlations.  A common-box plant
  reset is an experimental control with superficially similar information
  loss, not an implementation of neural IBP.
- **Convex relaxations** provide outer bounds without enumerating every ReLU
  phase.  **MILP** encodes discrete phases explicitly, and **BaB** explores
  phase/input splits while using a relaxation as a bounder.  These have
  different completeness and cost semantics.
- **PGD** is an attack/falsifier.  Failure to find an adversarial example is
  not a proof, just as sampled ODE trajectories do not prove an enclosure.
- **VNN-COMP** compares neural verifiers under standardized NN properties.  It
  is a useful model for benchmark governance, not direct evidence about
  Flow*, Taylor models, or DiffReach plant flowpipes.

## Course attachments requested for this study

These are attachment identifiers, not public-schedule lecture numbers.  Since
the files are missing, titles/pages not explicitly supplied are left
unavailable rather than reconstructed.

| requested attachment | verified attachment metadata | relation if restored |
|---|---|---|
| `Week-4-1-2.pdf` | Distinct Week 4 attachment; exact title and relevant pages unavailable. | Must be mapped separately.  Likely method/course context only unless its pages explicitly cover plant ODE reachability. |
| `Week-4-2-3.pdf` | A second, distinct Week 4 attachment; exact title and relevant pages unavailable. | Must not be merged with `Week-4-1-2.pdf`; classify from its own content when restored. |
| `Lecture-9_-Neural-Network-Verification-Bound-Propagation-2.pdf` | Bound-propagation attachment; exact subtitle/pages unavailable. | **Methodologically indirect / future NNCS.** Controller bounding cannot replace plant flowpipe validation. |
| `Lecture-10_-Neural-Network-Verification-Bound-Propagation.pdf` | Bound-propagation attachment; exact subtitle/pages unavailable. | **Methodologically indirect / future NNCS.** Relevant to IBP/CROWN-style controller contracts. |
| `Lecture-11_-Neural-Network-Verification-Bound-Propagation.pdf` | Bound-propagation attachment; exact subtitle/pages unavailable. | **Methodologically indirect / future NNCS.** Exact method/page mapping awaits the file. |
| `Lecture-12.pdf` | **Modeling Physics**; covers dynamical systems, stability, and Lyapunov reasoning. Exact page mapping unavailable. | **Direct conceptual background** for dynamical systems/stability; it does not itself certify any Torch/Flow*/DiffReach enclosure. |
| `584_homework2.pdf` | **Homework 2**. Exact title, exercises, and pages unavailable. | Cannot be mapped until restored. It is not Homework 1 and is not inferred from the public assignment schedule. |

## Public ECE/CS 584 schedule (separate source)

The public [Spring 2025 ECE/CS 584
schedule](https://publish.illinois.edu/ece584-spring2025/course-schedule/)
places the material in the following sequence:

| item | public topic/readings | use in this project |
|---|---|---|
| Lecture 7 | SMT, linear real arithmetic, Simplex | Background for exact constraint-based verification; not executed by the three plant adapters. |
| Lecture 8 | Neural networks and the ML-system verification problem | Defines the controller-verification problem that is absent from this plant-only benchmark. |
| Lecture 9 | Verification as optimization; MILP and LP formulations | Connects arXiv:1711.07356 and arXiv:1711.00851 and clarifies exact encoding versus relaxation. |
| Lecture 10 | Bound propagation | Connects CROWN and the convex-relaxation barrier; motivates explicit bound semantics. |
| Lecture 11 | Branch-and-bound | Connects β-CROWN and JMLR 19-468; separates bound quality from search completeness. |
| Public Lecture 12 | Guest lecture by Haoze Wu | This public schedule row is not used to relabel attachment `Lecture-12.pdf`, whose supplied title is **Modeling Physics**. |
| Public Homework 1 listing | Announced after Lecture 5 and due 2025-02-17 | This is not the requested `584_homework2.pdf`; no attachment content is inferred from it. |

The course ordering is useful: SMT/optimization, bound propagation, and BaB
are layers of an NN-verification stack, while plant dynamics/reachability
appear later in the course.  That separation is preserved here.  The course
project guidance also explicitly distinguishes hybrid reachability tools such
as Flow* from specialized NN verifiers and treats their composition as an open
systems problem.

## Relevance partition

- **Directly relevant now:** ODEs, dynamical systems, reachability,
  Taylor-model flowpipes, validated remainder handling, endpoint restriction,
  stability, and Lyapunov concepts.
- **Methodologically indirect now:** IBP, CROWN, β-CROWN, convex relaxation,
  MILP, BaB, verified training, adversarial attacks, and VNN-COMP protocol
  governance.
- **Relevant to a future NNCS study:** any of those NN bounders used as an
  explicit controller contract and composed with the plant reachability
  contract.
- **Not interchangeable with this experiment:** classifier/controller
  verification material cannot replace plant-only ODE integration, Picard
  validation, Taylor truncation-to-remainder, endpoint evaluation, or
  multi-step carry evidence.

## Resulting research position

1. The Flow* root-cause claim rests on source traces and analytic/high-precision
   reproduction, not on neural-verification literature.
2. The three-tool comparison remains plant-only.  DiffReach’s CROWN-style
   capability is not exercised and gives it no automatic advantage in these
   rows.
3. BERN merits a guarded polynomial-range experiment because it can retain
   cross terms in a coefficient representation.  It does not, by itself,
   improve carried Taylor-model dependency or validate an ODE step.
4. A future NN-controlled study should compose one explicit controller
   contract (for example CROWN/β-CROWN, IBP, or a complete MILP/BaB oracle)
   with the CIR and re-run containment/defect gates.  It must not inherit the
   plant-only conclusions by analogy.
