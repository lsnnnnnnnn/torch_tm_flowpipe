status: historical
valid_for_commit: 08b6f2416122cbf4220ff351e663caa1a0af13a2
superseded_by: docs/RESULTS_STATUS.md
allowed_use: provenance only

# Historical run registry

| Run | Source lineage | Current status | Allowed use |
|---|---|---|---|
| `20260723T173852Z` | first-order three-way | historical | lineage evidence only |
| `20260724T043709Z` | first-order matched-basis | historical diagnostic | reproduce under current contract before conclusions |
| `20260724T132534Z` | common contract | superseded protocol | provenance only |
| `20260728T140456Z` | correctness repair | historical repair | code/test lineage only |
| `20260730T015245Z` | deep study | provisional due to known defects | no runtime/Pareto/failure headline |
| `20260730T124958Z` | first formal freeze | failed acceptance | numerical diagnostics only; wrote outside output |
| `20260730T141302Z` | second formal freeze | rejected | Pareto incorrectly partitioned by tool |
| `20260730T153654Z` | former final formal freeze | `withdrawn_do_not_cite` | frozen provenance only; patched audit backend |
| consolidation smoke runs | pipeline checks | non-authoritative | pipeline diagnostics only |

The frozen source and artifacts remain reachable through the selected base and
existing archive tags. Current status does not edit historical bundle files.
