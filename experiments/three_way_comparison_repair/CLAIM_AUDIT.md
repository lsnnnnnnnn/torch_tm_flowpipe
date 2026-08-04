status: historical
valid_for_commit: unknown
superseded_by: docs/RESULTS_STATUS.md
allowed_use: provenance only

# Claim audit

| Old claim | Original evidence | Confounder | Status | Corrected wording | New artifact |
|---|---|---|---|---|---|
| Torch is tightest on Riccati | tightened Torch endpoint versus raw Flow*/DiffReach endpoints | endpoint postprocessing mismatch | invalid | Torch tightening is supplemental; compare raw endpoints | `torch_endpoint_audit.csv` |
| Flow* fails around t=0.08 | fixed order-2/candidate wrapper | constrained settings and generic failure code | invalid | that configuration fails; original Flow* reaches T=10 | `flowstar_original_parity.csv` |
| Flow* order 2 is less capable | deliberate low-order run | different legal bases/resources | invalid | order-2 fixed stress is diagnostic only | `flowstar_parameter_sensitivity.csv` |
| DiffReach is tighter on harmonic | common-box widths | bases and arithmetic remain unmatched | unresolved | report only matched raw semantics with caveats | `corrected_one_step_summary.csv` |
| DiffReach is faster | mixed build/JIT/first/steady totals | one-time and steady costs mixed | invalid | report runtime decomposition, not one scalar winner | `raw_results.csv` |
| Common-box comparison is fair | same external boxes | reset erases native dependencies | corrected | it controls carried boxes, not native method quality | `corrected_common_time_summary.csv` |
| All correctness gates passed | reinjected Flow* result | stock refined result was concealed | invalid | stock Riccati checks fail; Outcome B applies | `correctness_checks.json` |
| Flow* refinement was unvalidated | preliminary source audit | full Picard image had not been regenerated | confirmed | native remainder-only refinement fails the regenerated full-Picard self-map check | `flowstar_refinement_trace.csv` |

The old report's per-system “tightest” tables and failure-horizon rankings are
not carried forward. Candidate reinjection and endpoint-tightened Torch rows
remain visible only as labeled diagnostics/supplemental features. No claim is
made that floating-point sampled checks provide formal soundness.
