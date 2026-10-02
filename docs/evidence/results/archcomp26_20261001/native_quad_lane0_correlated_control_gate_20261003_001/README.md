# QUAD first-box correlated control construction: single native trace and exact audit

**Outcome: native construction traced; the conditional inclusion gate is UNDECIDED.** The saved build/run/check exit-code receipts are **0/0/2**. The exact checker recorded the first refusal at **output 1, lower side**. The frozen first-refusal [`RESULT.json`](RESULT.json) reports a lower margin of `−3.33045230798093×10⁻¹⁶` and an upper margin of `+1.10465077204337×10⁻¹⁶`. The complete native [`LANE0_TM_TRACE.json`](LANE0_TM_TRACE.json), [`START.json`](START.json), build/run/check logs, and exit-code files are preserved. This was one controller-construction trace with **no CROWN, ONNX, Flow* reachability, or ODE call**; it did not restart an old experiment.

[`lane0_control_trace.cpp`](lane0_control_trace.cpp) reconstructed only lane 0 of the frozen 2026-paper QUAD initial partition. It read the saved first-batch RPC and separate conditional corrected-bias ledger, verified twelve native input intervals equal the saved RPC row, and followed the frozen control construction order: JsonCpp `.asFloat()` slope/bias conversion, double center/radius, `TaylorModelVec<Real>` addition, then symmetric remainder addition. It exported the actual input/output Taylor-model coefficients, remainders, normalized domain, transported slopes, and bias endpoints as binary64 hex strings. The copied Flow* uses 53-bit MPFR `Real`/`Interval`; the exporter refused any `Real` that could not round-trip exactly through binary64, avoiding the prior 15-digit `Real` stream output.

The old [compound construction gate](../native_quad_corrected_control_construct_20261003_001/README.md) stopped at lane 0/output 1 because it demanded that the new TM interval enclose a separately interval-summed **RPC coordinate box**. That box can be wider than the image of the shared input symbols. This gate instead compares the *same normalized symbols* at each input and output. For each output, [`check_lane0.py`](check_lane0.py) forms the exact rational affine difference

```text
delta(xi) = sum_i float32(T_i) * input_polynomial_i(xi)
            - constructed_output_polynomial(xi)
```

over the saved native domain. It requires the constructed output remainder `[Rlo,Rhi]` to cover the conditional corrected bias interval `[bL,bU]`, the exact min/max of `delta`, and the transported input remainders:

```text
Rlo <= bL + min(delta) + sum_i min(T_i * input_remainder_i)
Rhi >= bU + max(delta) + sum_i max(T_i * input_remainder_i).
```

The checker refuses nonlinear or unexpected-degree terms instead of silently approximating them. It checks that each initial input TM uses only its own normalized symbol, that the joint TM image covers the physical source box, and that the image lies inside the saved RPC box. It also matches the replayed `uA=lA` first row and recomputes the conditional ideal-affine-to-float32 transfer margin. It uses Python `Fraction` on exact binary64/binary32 values for all decisions; displayed decimals are only summaries. The first negative margin means this sufficient conditional construction check did **not** pass. It is not an NN counterexample or proof that the actual network output escapes the control set.

## Local input audit

The [`INPUT_AUDIT.json`](INPUT_AUDIT.json) confirms twelve saved source intervals are inside their matching first-row RPC intervals; the replayed first-row `uA=lA` and existing affine coefficients match; and all three proposed corrected bias pairs satisfy the exact conditional transfer inequalities. The smallest reported input-transfer margin is about `1.85e-11` on output 2's lower side. The native TM trace now supplies the separate construction data.

## Three-output read-only analysis

The frozen checker stopped after output 1. A separate, **read-only** [`analyze_all_outputs.py`](analyze_all_outputs.py) applied the same Fraction arithmetic to all three rows of the already saved trace. Its [`ALL_OUTPUTS_ANALYSIS.json`](ALL_OUTPUTS_ANALYSIS.json) includes exact rational numerators/denominators and the first outward binary64 remainder endpoint for each side. No native executable was run for this follow-up.

| Output | Lower conditional margin | Upper conditional margin | Smallest fixed-polynomial lower remainder shift |
| --- | ---: | ---: | ---: |
| 1 | `−3.330452308×10⁻¹⁶` | `+1.104650772×10⁻¹⁶` | 48 binary64 ULPs; actual `3.330669074×10⁻¹⁶` |
| 2 | `−5.159067004×10⁻¹⁹` | `+5.059097017×10⁻¹⁹` | 1 binary64 ULP; actual `1.734723476×10⁻¹⁸` |
| 3 | `−5.310919262×10⁻²⁰` | `+3.298200777×10⁻²⁰` | 1 binary64 ULP; actual `1.734723476×10⁻¹⁸` |

All three lower-side sufficient inequalities fail; all three upper sides pass. The shifts above are exact *what-if* adjustments to the saved final native TM remainder endpoints **with its polynomial and input TMs held fixed**. Changing the corrected JSON biases would rerun center/radius and TM construction, so these numbers are not a validated source patch. The archived [`run_first_box.sh`](run_first_box.sh) prevents a repeat in its run directory. The frozen `RESULT.json`, old RPC, original experiment, and library were not modified for the three-output analysis. Controller soundness, first-step plant behavior under any widened control, later controls, and the full-time property remain open; native octagon production stays **CLOSED**. No hash or digest verification was performed.
