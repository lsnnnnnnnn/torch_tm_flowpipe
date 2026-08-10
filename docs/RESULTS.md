# Results

Date: 2026-08-10
Canonical run: `outputs/mainline_realignment_20260810/20260810T025910Z`

## Native design points

| lane | native result | exact output semantics | soundness / eligibility |
|---|---|---|---|
| stock Flow* `b85a321...` | T10, 290 segments, core `0.441634 s` | full-segment tubes | `unsound/ineligible` for primary formal comparison after scalar-affine MPFR defect |
| stock DiffReach `dd628eb...` | B64 h=.01 T10; all 128,000 returned initial masks pass | endpoint at local time h; no stock tube | `empirically sampled only`; mixed builder dtype |
| Torch complete O4 | partial; highest validated `6.397083942944808` | raw endpoint, last tube, prefix tube separate | `formally outward by construction`; ineligible as T10 completion |

These rows reproduce native behavior; they do not form a ranked comparison.

## Torch factorial

| representation + validator + carry | result | decision |
|---|---|---|
| fixed DR7 + two Picard + DR-RP + normalized affine/symbolic carry | B64 T10 completes; explicit-f64 fixture bit-exact | qualified DiffReach-like lane |
| complete O4 + raw-remainder-compatible validator + normalized insertion | partial through `6.397083942944808` | authoritative complete baseline |
| complete O4 + same validator + exact complete endpoint carry | partial through `0.04345468750000001` | candidate rejected |

The fixed DR7 B64 lane independently completes every requested
T=.1/.5/1/4/6/6.5/7.5 run and the fresh T10 run. The adaptive complete-O4
baseline independently requests T=7.5 and T=10 and stops at the same
`6.397083942944808` boundary. The complete-carry candidate independently
requests all eight horizons and stops at the same `0.04345468750000001`
boundary.

The validators cannot be blindly crossed: DR-RP is defined on the restricted
slot contract with component-retain semantics, while the complete raw-remainder
validator consumes a complete-basis truncation ledger. A forced cross would
change construction and acceptance semantics, not isolate one factor.

## Causal answer

At the first split, common-basis coefficient error is at most `1.421e-14` in y.
Flow* raw y remainder is
`[-1.0366239882151062e-4, 1.0359846643018429e-4]`, already outside the target;
Torch raw y is `[-9.14291532216261e-5, 9.358938647674799e-5]`, inside. Swapping
polynomials, endpoints, and right maps does not swap the receiving decision.

## Performance answer

Fixed-support CPU warm time for 10 steps rises from `0.4772511 s` at B1 to
`0.6959472 s` at B512; synchronized V100 remains about `1.25 s`. Complete O4
one-step-plus-carry rises from `0.1168205 s` to `9.3192035 s` on CPU and from
`0.3070624 s` to `10.4725577 s` on V100. Inputs are actual independent grid
partitions. The complete batch rows are a one-step kernel scope because the
outer adaptive scheduler remains batch-one.

No eligible cross-tool deployment speedup, GPU speedup, or precision-throughput
Pareto frontier is claimed.

## Claim status

- Valid: native entrypoints reproduced; fixed-support explicit-f64 semantics;
  exact causal stage; exact carry soundness; measured partial horizons.
- Invalid: global Torch-vs-Flow* tightness/speed; native/matched equivalence;
  ordinary CUDA as universally outward rounded; complete carry improvement.
- Blocked: formal stock Flow* primary comparison; formal ordinary
  fixed-support CPU/CUDA certificate; multi-step batched complete lane.
