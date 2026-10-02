# QUAD first CROWN batch: conditional float32 bias corrections

**Outcome: a fully enumerated, conditional correction plan; no solver or plant run.** This CPU-only calculation reads the saved 1,024-box first [RPC](../native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json) and the new controller-only [same-slope receipt](../native_quad_crown_same_slope_batch_20261003_001/README.md). It confirms that the new batch's `lA/lbias/ubias` equal the old RPC and that `uA=lA` in all 36,864 slope entries. It then calculates the smallest per-side residual interval enlargement required to **directly transfer the assumed ideal CROWN affine bounds** through the native binary32 coefficient transport. It does not execute CROWN, the ONNX model, Flow*, or an ODE step.

For a saved input box `X`, let `A=lA=uA`, `A'=float32(A)`, `bL'=float32(lbias)`, and `bU'=float32(ubias)`. Assuming the original **real-arithmetic** affine statements `A·x+lbias ≤ f(x) ≤ A·x+ubias` hold on `X`, the transported slope requires

```text
required lower = lbias + min over x in X of (A−A')·x
required upper = ubias + max over x in X of (A−A')·x.
```

The minimum real expansions relative to the native transported biases are `max(0,bL'−required lower)` downward and `max(0,required upper−bU')` upward. [`compute_corrections.py`](compute_corrections.py) uses exact rational arithmetic for these values, lifting the parsed binary64 RPC inputs/coefficients and the binary32 transported values. It then keeps any already sufficient native endpoint, or chooses the nearest **outward binary32** lower/upper endpoint. It checks the chosen endpoint against the exact rational requirement and its adjacent binary32 value. This gives the minimum binary32 endpoint movement for this sufficient affine-certificate transfer; it does **not** give a minimum actual neural-network enclosure.

The calculation covers **3,072 output rows**. At least one endpoint changes on **2,456 rows**, spanning **1,014/1,024 source boxes**; 523 source boxes require a change in all three outputs. Percentiles below are sorted nearest-rank positions from the 1,024 values for each output; all exact values and further percentiles are in [`RESULT.json`](RESULT.json).

| Output | Boxes with lower / upper / either correction | Maximum required real lower / upper expansion | Median / maximum binary32 interval-width increase | Maximum relative width increase |
| --- | ---: | ---: | ---: | ---: |
| 1 | 573 / 562 / 833 | `6.5882e−7` / `6.5244e−7` | `9.5367e−7` / `1.90735e−6` | `2.9017e−5` |
| 2 | 615 / 539 / 828 | `2.1445e−10` / `9.3490e−10` | `9.3132e−10` / `1.97906e−9` | `1.1775e−7` |
| 3 | 533 / 567 / 795 | `2.5547e−10` / `4.8363e−10` | `9.3132e−10` / `1.39698e−9` | `8.4938e−8` |

For lane 0, output 1's native lower residual endpoint `9.930038452148438` becomes `9.930037498474121` while the upper endpoint remains `9.997137069702148`. Output 2's upper endpoint `0.015700342133641243` becomes `0.015700343996286392`; its lower endpoint remains. Output 3's lower endpoint `−0.005758021492511034` becomes `−0.005758021958172321`; its upper endpoint remains. The exact minimum **real** expansions are smaller than these discrete binary32 movements: about `3.23969e−7`, `3.74727e−10`, and `2.03706e−10`, respectively. All 3,072 rows, including required real bounds and corrected float32 endpoints, are in [`CORRECTIONS.csv`](CORRECTIONS.csv); [`CORRECTED_BIASES.json`](CORRECTED_BIASES.json) contains the two 1,024×3 endpoint arrays for a possible isolated overlay.

## How a bounded follow-up could use this

An isolated candidate could preserve the saved `T`, replace only the first-call `u_min/u_max` with the corrected arrays, then verify that the **actual constructed Taylor-model control hull** includes those corrected endpoints after the C++ center/radius arithmetic. It must check the same 1,024 input boxes and stop on the first construction or step refusal. Because the control sets would be wider, the earlier first-step plant and two-step lane-0 receipts would need fresh qualification for that candidate; none are claimed here. Do not edit the original RPC or promote these conditional values to production.

This computation assumes the CROWN real-arithmetic affine inequalities. The same-slope replay records equality of returned numbers, but it does not certify CROWN's floating-point soundness, the selected network's outputs, native center/radius rounding, later controls, or the `T=5` reach-and-remain property. A failed uncorrected transfer is **not** an actual NN counterexample. The native octagon production gate stays **CLOSED**. No hash or digest verification was performed.
