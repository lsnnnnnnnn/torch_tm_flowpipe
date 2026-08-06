# Pre-fix/post-fix trace field map

| Field | Pre-fix behavior | Post-fix behavior |
|---|---|---|
| right_map_input | adjacent/pre-state substitute | exact previous `tmv_right` object |
| insertion_input/output | adjacent stage labels | exact `endpoint_without_constants` / `inserted` objects |
| normalized_reset_input/output | reset aliases/substitutes | exact `inserted_for_reset` / `reset_tm` objects |
| object identity | Python-adjacent assumptions | stable full-content SHA256 plus exact in-process object checks |
| rejected attempt index | could look like an accepted step | `accepted_step_index=null`, with accepted-count-before-attempt |
| call-44 component | roots could be mislabeled x | y roots and y result are component 1; x-only aggregations component 0 |
| ancestry | implied trajectory-wide | explicitly terminal-local; cross-step completeness false |

The post-fix schema is `vdp_transition_trace_schema_v2`. Missing lifecycle objects fail closed to null with a reason; adjacent objects are never substituted.
