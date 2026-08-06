# Q3 order and matched-contract audit

## Decision

This is **Case B**: the name Q3 exists and the Xiangru baseline reproduces, but the available benchmark/order contracts are not mathematically matched. A formal Torch–Xiangru or three-way winner comparison is not authorized.

Xiangru `complete_q3` means a dense, complete total-degree-at-most-3 monomial support over six variables: local time plus five current state/control parameters. It contains `C(9,3)=84` slots. Products and time integrations outside that support are intervalized. A direct unit example confirms that `1,x,t,x²,xt,t²,x³,x²t,xt²,t³` are retained and a degree-4 term is dropped.

Torch `Polynomial.truncate(3)` uses the same small retention predicate, `sum(exponent)<=3`. That does not make the complete algorithms equivalent: Xiangru uses a fixed dense six-variable basis, exactly two polynomial Picard iterates and ten DR-RP rounds; Torch's current VDP lane is sparse and defaults polynomial Picard iterations to `order`, hence three at order 3. More decisively, no common model exists in the current lanes.

## First failed gate

Gate 1 fails before any one-step comparison:

| Field | Xiangru Q3 baseline | Available Torch candidate | Stock Flowstar |
|---|---|---|---|
| Model | homogeneous TORA closed loop | Van der Pol plant only | Van der Pol plant only |
| State | `x1,x2,x3,x4,u1` | `x,y` | `x,y,t` |
| Controller | frozen ReLU NN, outward auto_LiRPA, 1 s update | none | none |
| Initial set | 4D box + held control, B48 | `[1.1,1.4]×[2.35,2.45]` | same VDP box |
| Horizon | 20 s | 10 s candidate contract | 10 s |
| Step | fixed 0.1 | adaptive 0.002–0.1 | adaptive 0.002–0.1 |
| Work/device | CUDA closed loop | CPU plant only | CPU plant only |

The unique decisive blocker is that Torch has no existing lane for the exact homogeneous-TORA plant, five-variable held-control representation, frozen controller weights/bounds, B48 partition, and T20 closed-loop workload. The task explicitly forbids adding an approximate sine or controller adapter merely to manufacture a match, so none was added.

## Flowstar role

Stock Flowstar at `b85a3211748cb77b736fe4ad42ee02d8d2b81148` remains an independent Van der Pol reference. Its VDP source declares the polynomial RHS, `[1.1,1.4]×[2.35,2.45]` box, adaptive 0.002–0.1 policy, order 4, symbolic remainder queue 100 and T10. Because it is not Xiangru's TORA model, it is excluded from formal three-way tightness/runtime tables.

## Machine evidence

- Checked retention outcomes: `outputs/xiangru_q3_matched_audit_20260806/contract/order_semantics_tests.json`.
- Source/operation mapping: `order_semantics.md` and `order_field_map.csv`.
- Per-tool required-field contracts: `xiangru_native_contract.json`, `torch_candidate_contract.json`, `flowstar_candidate_contract.json`.
- Fail-closed decision and all blockers: `matched_contract.json` and `matched_contract.md`.
- Selection/gates: `benchmark_selection/` and `matched_runs/*/gate_status.json`.

Every required contract field contains a value, matched state, source file/line, evidence, and reason. Unknown or false fields block authorization in tests.
