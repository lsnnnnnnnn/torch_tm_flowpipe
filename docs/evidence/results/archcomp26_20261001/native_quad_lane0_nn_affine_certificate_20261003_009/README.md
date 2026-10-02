# QUAD saved first-call lane 0: independent affine NN residual gate

**Outcome: UNDECIDED.** This is a CPU-only check of one saved first-call RPC input box against the locally selected `quad_controller_3_64_torch.onnx` mathematical map. It does not rerun CROWN, Flow*, ONNX inference, or any old experiment. The original native octagon production gate remains **CLOSED**.

[`check_lane0.py`](check_lane0.py) reads the saved [first-call RPC](../native_quad_allbox_firststep_recenter_gate_20261003_003/run/rpc.json). A deliberately restricted ONNX wire reader checks the exact seven-node `Gemm, Sigmoid, Gemm, Sigmoid, Gemm, Sigmoid, Gemm` chain, four float32 weight/bias pairs and Gemm attributes. The stored float32 parameters and the native-transported float32 `T`, `u_min`, and `u_max` are lifted to exact Decimal values. Input endpoints are lifted from the parsed binary64 RPC JSON values. This checks the **real-valued mathematical network with those exact parameters**, not a particular float32 inference implementation or the server model's byte identity.

For each neuron the checker keeps a correlated `A·(x-c)+b+R` enclosure. Linear arithmetic uses Decimal contexts with 80-digit directed rounding. At a Sigmoid it chooses a fixed exact decimal slope and encloses `sigmoid(z)-alpha*z` using monotonic derivative tests and bounded interval bisection. Decimal `exp` is correctly rounded to nearest; the checker brackets it by the adjacent decimal values. The script preserves per-neuron preactivation, relaxation error, and subdivision diagnostics in [`RESULT.json`](RESULT.json). It also uses `copy_negate()` so unary sign changes cannot silently round in the ambient 28-digit context.

The whole saved lane-0 box could **not** be enclosed within the three transported CROWN residual intervals:

| Output | Independent affine residual width | Saved residual width | Worst bound excess |
| --- | ---: | ---: | ---: |
| 1 | 0.09846842618 | 0.06709861755 | 0.01571918546 |
| 2 | 0.03865434202 | 0.01680773892 | 0.01092394512 |
| 3 | 0.03901185030 | 0.01644709567 | 0.01128668306 |

These excesses are **overestimation of this independent enclosure**, not neural-network counterexamples. The first failing comparison is output 1: independent lower `9.91431926669` versus saved lower `9.93003845215`, and independent upper `10.01278769287` versus saved upper `9.99713706970`. The saved upper-slope field `uA` is still absent from the RPC; this checker neither repairs nor disproves that contract.

An exploratory 64-subbox refinement of the six nonconstant axes was stopped before completion after several minutes; it produced **no coverage result** and is not counted. The final saved script and RESULT reproduce only the completed unsplit one-box check. No hash or digest verification was performed.

Run from the repository root with `python3 -B docs/evidence/results/archcomp26_20261001/native_quad_lane0_nn_affine_certificate_20261003_009/check_lane0.py`. A different graph, wider model family, or later control call is outside this check.
