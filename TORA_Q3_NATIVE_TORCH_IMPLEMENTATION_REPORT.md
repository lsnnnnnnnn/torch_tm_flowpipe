# Native Torch TORA-Q3 implementation report

## Plain-language outcome

1. **Did we implement the same TORA?** Yes.  The frozen dynamics, B48 leaf
   partition/order, held-control semantics, `h=0.1`, six variables, complete
   total-degree Q3 basis, 84-slot order, and `|x1..x4| <= 2` property match the
   Xiangru contract.  The Taylor-model algorithms are method-native and are
   not claimed to be identical.
2. **Did common-control plant replay reach T20?** Yes: 200/200 segments and
   48/48 leaves per segment validated.
3. **Did native full closed loop reach T20?** No.  It validated T1 and reached
   T4.3, then leaf 0 failed the `x3` tube property at segment 44 (T=4.4).
4. **Which result is tighter?** It depends on state and enclosure kind.  At
   T20, Torch endpoint medians are tighter for `x1` and `x2`; tube medians are
   tighter for `x2` and wider for `x1`, `x3`, and `x4`.  `u1` and the `x3/x4`
   endpoints differ only by the explicit roundoff envelope.  See the common-
   control report for exact ratios.
5. **What are steady runtime and cold wall?** The B48 one-step steady median is
   2.349 s on the V100.  Full repeated T20 statistics are recorded in
   `TORA_Q3_RUNTIME_REPORT.md` and the machine-readable runtime directory.
6. **Where is the first divergence?** In the corrected common-control lane,
   segment 1 `x1`: endpoint max difference `3.8234e-6`, tube max difference
   `1.0082e-3`, and leaf-0 remainder-width difference `4.0129e-6`.
7. **What is the present classification?** Two actual implementation bugs
   were found and fixed.  The remaining plant differences are expected
   sine/Picard/remainder/range-method differences; the native closed-loop
   failure is accumulated method-native plant/state-projection growth feeding
   a different but sound controller input box.  The historical Git lineage
   remains blocked by unknown asset authorization (Case D); the owner later
   authorized only a separate parentless clean review branch, documented in
   `CLEAN_REVIEW_PUBLICATION.md`.
8. **Is the old VDP `t=6.397...` issue solved?** No.  This task implements and
   audits TORA-Q3; it does not resolve that independent VDP issue.

## Frozen implementation contract

The native lane uses state order `[x1,x2,x3,x4,u1]` and dynamics

```text
x1' = x2
x2' = -x1 + 0.1 sin(x3)
x3' = x4
x4' = u1 - 10
u1' = 0
```

The B48 order is the Cartesian split `(8,6,1,1)` of
`[0.6,0.7] × [-0.7,-0.6] × [-0.4,-0.3] × [0.5,0.6]`.  Each segment has local
time plus five normalized spatial/control generators.  Complete Q3 therefore
has `C(6+3,3)=84` terms and fingerprint
`fa135259d41a68a73a6fc609880c4fd466bf2d53b2dddeba30298a484fa5e44d`.

The plant uses configurable K2/K3 polynomial Picard and configurable seed and
remainder rounds.  Formal runs use K2, seed 0.01, and ten componentwise
remainder rounds.  Multiplication and integration overflow, sine composition,
roundoff, endpoint substitution, and affine composition are carried in named
interval ledgers.  The plant module imports no Xiangru kernel.

## Controller boundary

`TORA_CONTROLLER_PATH` is mandatory.  The loader verifies original-controller
SHA-256 `52a50c6bc6b1b45b89319edc809cd4d3baca06c32f9fd2ddceb2e95007414418`,
reconstructs the flattened float64 Linear/ReLU graph, and uses the same
auto_LiRPA CROWN/same-slope contract and outward affine composition.  No
controller bytes are included in the new deliverables.  The nominal ONNX
float32 comparison has max error `5.1034085e-7` (declared tolerance `1e-6`),
and the initial B48 interval comparison has max error `3.5527137e-15`.

## Actual bugs found and fixed

The first implementation exported local Picard tube/endpoint bounds before
substituting the affine normalized-state parameterization.  It also evaluated
an exact-time endpoint termwise, so `z_i` and `h·z_i` coefficients could not
cancel after `tau=h`.  Pre-fix traces and hashes were retained privately.

The fixes add sound dense affine composition and sound exact-time substitution
with outward coefficient-aggregation error.  The regression tests
`test_affine_composition_materializes_local_spatial_coordinates` and
`test_exact_time_endpoint_aggregates_equal_spatial_exponents_soundly` fail on
the former behavior.  After the fix, the first full-loop `x4` endpoint max
difference fell from about `2.15e-2` to `2.20e-12`.  All plant gates were then
rerun from one leaf/one step through T20.

## Evidence boundary

Raw controller bytes, raw 200-segment per-leaf traces, unsanitized logs, and
the private Xiangru observation patch live only under the private evidence
root.  Public artifacts contain contracts, hashes, aggregate tables, tests,
and reports.  The historical authorization blocker and remediation choices
are documented in `PUBLIC_ARTIFACT_GOVERNANCE_AUDIT.md`.
