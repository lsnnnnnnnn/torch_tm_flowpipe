# Matched VDP order-4 execution contract

This file is generated from the stock Flow* benchmark/default sources, the Torch runner/config, and the frozen terminal checkpoint. `matched=no` is intentional and fail-closed; it is never filled by a look-alike field.

| Field | Matched | Evidence-based reason |
|---|---:|---|
| `ode` | yes | The physical x/y equations match exactly; stock adds the clock state t'=1. |
| `physical_state_order` | yes | position/velocity aliases are mapped to stock x/y. |
| `stored_state_order` | no | stock stores an extra clock state used for plotting/safety. |
| `initial_set` | yes | The physical box is identical; stock's clock starts at the point interval zero. |
| `taylor_order` | yes | Both use fixed Taylor order four. |
| `retained_degree` | yes | Both retain complete total degree <=4; the Torch raw-RHS replay uses degree 3 before tau integration. |
| `local_variable_order` | no | Stock places local time at domain index 0 and includes one normalized generator for each x/y/t state; Torch omits the constant clock generator and places tau last. |
| `local_time_domain` | yes | Domains match after the explicit index permutation in field_map.md. |
| `initial_step` | yes | Both start at h_max. |
| `h_min` | yes | Source defaults match. |
| `h_max` | yes | Source defaults match. |
| `step_shrink` | yes | Rejected attempts are halved. |
| `step_growth` | yes | Accepted steps propose 1.1*h capped by h_max. |
| `initial_candidate_remainder` | yes | The physical x/y candidate intervals match; stock also assigns the default interval to its clock component. |
| `cutoff` | yes | Absolute cutoff threshold matches. |
| `truncation` | no | The truncation support and cutoff match, but the authoritative Torch lane evaluates dropped polynomial ranges with four-leaf subdivision; stock does not. |
| `picard_validation` | no | The Torch predicate replays stock raw-remainder algebra, but iteration/control fields are not exposed as an identical public stock contract; nulls are not substituted. |
| `normalized_insertion` | no | The broad normalized-insertion structure is analogous, but the evaluated range operators and stored state dimensions are not identical. |
| `symbolic_remainder_queue` | no | This is a material cross-step representation difference, not a matched field. |
| `numeric_backend` | no | Precision is nominally 53 bits, but the interval implementations and guarantees differ. |
| `endpoint` | yes | Semantic endpoints align; no segment box is substituted for an endpoint. |
| `segment` | yes | Segment semantics align. |
| `tube` | no | Torch stores an aggregate x/y hull and segment records; stock retains the full ordered flowpipe vector. These are mapped but not equated. |

The physical ODE, initial x/y box, order, step bounds, half-on-reject/1.1-on-accept scheduler, candidate remainder radius, and cutoff match. The authoritative end-to-end executions do not form one fully matched numerical contract because Torch has no active symbolic-remainder queue, uses a different interval backend, has a different stored/local basis layout, and proactively subdivides the polynomial-truncation range.
