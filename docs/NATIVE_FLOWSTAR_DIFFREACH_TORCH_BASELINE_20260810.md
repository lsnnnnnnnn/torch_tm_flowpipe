# Native Flow*, DiffReach, and Torch VDP baseline

Date: 2026-08-10
Run ID: `20260810T025910Z`

## Outcome

All three native baselines were executed from pinned source with their official
or already-frozen entry point.  They are **design-point observations, not a
ranking**: their representation, partition, step, validation, carry, output,
and numerical-soundness contracts differ.

| lane | pinned source | native request | result | output object | qualification |
|---|---|---|---|---|---|
| stock Flow* | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` | official VDP, adaptive O4, T10 | completed T10; 290 accepted segments; core `0.441634 s`; reported safe | full-segment and full-horizon tubes | native reference only; scalar-affine gate proves stock numerical under-enclosure, so not primary formal comparator |
| stock DiffReach | `dd628eb443b517d6415de93e7035b4baef73963e` | official YAML, B64, h=0.01, 1,000 steps | completed T10; verify warmup `4.307 s`, after-JIT `0.360 s`; all returned initial inclusion flags pass | endpoints because `BOUND_TIME_STEP=True`; stock output has no full-step tube | native fixed-support throughput reference; later DR-RP retain masks are not returned as completion gates |
| Torch authoritative complete O4 | numerical lineage `4707abed5e8d28ec56c2b5e76b800bd284f0008b`; fresh run at `dc26187660609c7a313bcc27e1c684d5262359b8` | complete O4 + raw-remainder-compatible validator + normalized insertion + proactive depth-1 subdivision on polynomial truncation | expected fail-closed result: 307 accepted, 48 rejected, highest validated `6.397083942944808`; process `516.796263850294 s` | raw endpoint, last segment tube, full prefix tube | authoritative frozen Torch baseline; no repair, fallback, endpoint tightening, or sample violation |

## Stock Flow* reproduction and correctness gate

The stock benchmark was built and run in the compatible GCC 11.4/GSL 2.7
container from a detached clean source worktree.  The official last accepted
schedule row is `t=10`, `h=0.004872`, order 4.  The last full-segment tube is:

| state | last segment tube | width | full-horizon tube |
|---|---:|---:|---:|
| x | `[-1.469806, -1.175589]` | `0.294217` | `[-2.022605, 2.067539]` |
| y | `[-2.588135, -2.177639]` | `0.410496` | `[-2.754938, 2.721230]` |

The `.plt` path evaluates each Taylor flowpipe across its whole local-time
domain, so these are tubes, not endpoints.

The independent scalar-affine closure gate used `x'=1+2x` and a closed-form
MPFR oracle.  The first containment loss is the second accepted remainder
refinement.  Its maximum endpoint defect is
`3.3337554938839276e-10`; the final official-path defect is
`3.4938679727147814e-10`.  The missing term is the order-4
`tau^4 * initial_generator` dependency.  The selected outcome is
`F_clean_stock_flowstar_core_behavior`: native reproduction remains useful,
but primary formal comparison eligibility is false.  This result is a
numerical qualification of this stock build and workload; it is not a claim
that Flow*'s abstract algorithm is unsound.

## Stock DiffReach reproduction and semantic audit

The native configuration uses the official VDP initial box split 8 by 8,
giving B64, h=0.01, 1,000 steps, absolute seed `1e-2`, 10 DR-RP refinement
rounds, and symbolic-remainder window 1,000.  JAX x64 was enabled and GPU 0
was the V100 with UUID
`GPU-c1336362-1a12-45dd-8d3f-d2011d6f51ae`.

Although the launcher enabled JAX x64, the pinned `build_linear_tm` and
identity/symbolic-state builders default to float32 when the caller omits a
dtype; the official driver does omit it.  The native row is therefore a
mixed-builder-dtype observation, not a pure float64 run.

The fixed support is the exact seven-slot restricted basis
`[1, t, x1, x2, t^2, t*x1, t*x2]`, stored upstream as
`c`, `L[..., t,x1,x2]`, and `Lt[..., t,x1,x2]`.  Each step uses two
polynomial Picard iterates.  DR-RP first computes and returns an initial
elementwise inclusion mask.  Each later round accepts a narrower component
and otherwise retains the previous interval component.  Those later masks do
not leave the stock API; the launcher only warns on the returned initial
mask.  All 128,000 returned initial component flags were true and all saved
arrays were finite.

At T10 the aggregate endpoint is:

| state | endpoint | width |
|---|---:|---:|
| x | `[-1.3964494166017873, -1.2195860990449185]` | `0.1768633175568688` |
| y | `[-2.5022084859452010, -2.2803973198181486]` | `0.2218111661270524` |

Because `BOUND_TIME_STEP=True` fixes local time to h when saving each row,
stock DiffReach provides endpoints here.  A full-segment tube is
`UNAVAILABLE`; it must not be compared directly with Flow* `.plt` tubes.

## Torch baseline closure

The natural-range complete-O4 diagnostic independently reproduced failure at
`6.3172908799330765`.  The authoritative baseline additionally enables the
already-frozen proactive four-leaf polynomial-truncation range.  Its fresh
T=6.5 request exactly reproduced the frozen numerical terminal:

- highest validated time: `6.397083942944808`;
- terminal attempted h: `0.003623635847674574`;
- accepted/rejected work: 307/48;
- terminal x/y inclusion margins:
  `9.963763341523255e-5`, `-1.99995911680722e-5`;
- range work: 4,312 subdivisions and 17,248 leaf evaluations;
- raw endpoint at the validated prefix:
  x `[-0.1336886316181628, 1.3785118640319567]`,
  y `[0.8988221407067473, 4.371426383921656]`;
- no fallback, hidden sparse path, endpoint repair/tightening, or sampled
  trajectory violation.

The failure is a validator rejection at minimum step, not a process crash and
not T10 completion.  The previous causal audit's numerical state hashes remain
the frozen identities for `current`, `tmv_pre`, and `tmv_right`:
`17de6d46dae3f3c1123627d507756741d02ebcb0f2dbda7754b4a6134563bc5e`,
`efe776ac16eedc29b5582e7de979f5442efa79ed3ef7092f28484089a49b04ad`,
and `c721ccf4c02099afd7064a79dd3235759f453df6c8315d2a4e8745ecd7ed3bb3`.

## Comparability boundary

| field | Flow* | DiffReach | Torch authoritative |
|---|---|---|---|
| representation | complete total-degree O4 | fixed seven-slot restricted quadratic | complete total-degree O4 |
| partition | one initial box | 64 boxes | one initial box |
| step policy | adaptive native | fixed 0.01 | adaptive Flow*-compatible |
| validator | stock interval Picard | DR-RP | raw-remainder-compatible inclusion |
| carry | native normalization + symbolic queue | upstream normalization + symbolic queue | normalized insertion; no symbolic queue |
| saved geometric object | tubes | endpoints | endpoint plus explicit tubes |
| numerical qualification | `unsound/ineligible` for primary formal use after analytic counterexample | `empirically sampled only`; JAX x64 does not override the default float32 model builders | `formally outward by construction` for its Torch interval operations; lane still fails before T10 |

Consequently, completion and width values above answer whether each native
entry point ran and what it emitted.  They do not establish a speed or
tightness winner.

## Evidence

Machine-readable facts and artifact hashes are in
`outputs/mainline_realignment_20260810/20260810T025910Z/01_native_baselines/native_baselines.json`.
Raw logs and copied native artifacts live beside that file.  The Flow* scalar
gate is under `flowstar_scalar_affine_gate_expected_build/`.
