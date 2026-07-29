# BERN-NN-IBF feasibility decision

## Decision

BERN-NN-IBF is not a fourth reachability tool in this study.  Its useful
near-term idea is a Bernstein coefficient hull as an optional range query for
an already constructed Torch Taylor-model polynomial.  Continue only as a
range-backend experiment after adding a formally justified roundoff enclosure
and a dimensional/storage guard.  Do not replace the Taylor-model
representation, validator, or reset policy with BERN on the current evidence.

The minimal executable prototype is `bern_feasibility.py`.  It implements the
standard power-to-tensor-product-Bernstein identity independently, preserves
cross terms, runs on CPU float64, and compares its coefficient hull with the
current interval range on cancellation-heavy and study-derived nonlinear
terms.  `bern_feasibility.csv` and `bern_feasibility.json` record exact
analytic ranges, deterministic sample sanity checks (never proof), timing,
storage payload estimates, and the explicit scope decision.

## Audited local evidence

- External repository:
  `/srv/local/shengenli/BERN-NN-Implicit` at
  `ebcf54a0e06597a5388db0387865493c1dc96c07`, read-only and clean.
- Recovered feasibility worktree:
  `/srv/local/shengenli/torch_tm_flowpipe_bern_ibf_study`, branch
  `codex/bern-ibf-tm-feasibility`, HEAD `dd82032`, with preserved uncommitted
  study code.  This deep-study prototype does not alter or depend on that
  working tree.
- The BERN checkout has no explicit license file or source-header grant.
  Therefore no BERN implementation source is copied here.
- BERN represents each power monomial by factorized univariate Bernstein
  coefficient rows.  Addition/degree elevation and term-pair multiplication
  preserve the polynomial algebraically, while implicit extrema ultimately
  enumerate tensor coefficient indices.
- The repository supplies neural-network polynomial bound propagation and
  CUDA kernels, not Taylor-model local-time integration, total-degree
  truncation with overflow transfer, independent remainder propagation,
  defect/Picard inclusion, endpoint substitution, or multi-step reset.
- The audited neural paths default to CUDA float32.  Several arithmetic
  extensions use raw `float*`; the extrema kernel dispatches float types, but
  that does not make the end-to-end arithmetic a directed-rounding enclosure.
  The prototype therefore labels its inflated float64 hull a numerical
  candidate, not a formal roundoff proof.

## Four requested questions

| question | evidence-backed answer |
|---|---|
| Polynomial range bounding | Promising for cancellation/cross-term polynomials when the Bernstein hull is tighter. Production use requires directed conversion/arithmetic or MPFR validation and must avoid dense coefficient explosion. |
| Cross-term/dependency preservation | Products such as `x1*x2` remain explicit inside one polynomial. This does not preserve correlations that were already destroyed by a box reset, nor does it solve dependence between a polynomial and its interval remainder. |
| NN controller bound | BERN is directly relevant to polynomial neural-network bounds, and CROWN/IBP/BaB literature is relevant if a controller is later introduced. The present plant-only benchmark contains no controller, so it provides no controller result. |
| Torch TM GPU batching | The factorized layout is batchable, but coefficient extrema still perform work exponential in dimension/degree-grid size. GPU claims require many simultaneous polynomials and an audited CUDA enclosure; singleton V100/CUDA timing is not evidence. |

## Soundness boundary

The Bernstein convex-hull theorem is exact-arithmetic evidence.  The current
prototype adds a conservative floating-point allowance and verifies known
analytic cases, but it is not a complete proof of every conversion operation.
It cannot participate in the primary Flow*/Torch/DiffReach correctness or
Pareto tables.  A future production path must first prove its numerical
enclosure, retain the native interval remainder unchanged, and pass the same
analytic/CIR/defect gates as the existing range backend.
