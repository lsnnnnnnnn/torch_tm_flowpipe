# Cross-tool field map

| Meaning | Torch field | Flow* field | Mapping status |
|---|---|---|---|
| physical position | `x`, state 0, local generator `u0` | `x`, state 0, local generator `r0` after time | exact after basis permutation |
| physical velocity | `y`, state 1, local generator `u1` | `y`, state 1, local generator `r1` after time | exact after basis permutation |
| local time | `tau`, domain index 2 | domain index 0 | exact after index permutation |
| clock state | absent | `t`, state 2, normalized generator `r2` | no Torch equivalent; use `null` |
| endpoint | `endpoint_raw_tm` at `tau=h` | composed flowpipe at local-time supremum | semantic match; never replace with segment range |
| segment | `segment.tm` over `[0,h]` | `TaylorModelFlowpipe` over domain index 0 | semantic match |
| tube | aggregate x/y hull plus ordered segment CSV | ordered `tmv_flowpipes` plus plot projection | not identical; compare ordered segment boxes, not aggregate hull |
| candidate remainder | per-x/y `[-1e-4,1e-4]` | per-x/y/t default `[-1e-4,1e-4]` | physical components match; clock field is `null` on Torch |
| Picard subfields not exported by stock trace | named Torch ledger fields | unavailable | Flow* value must remain `null`; aggregate remainder is not a substitute |
| symbolic carry | inactive (`symbolic_queue_present=false`) | `Symbolic_Remainder{J,Phi_L,scalars}`, max 100 | material mismatch |

Common monomial comparison uses `[u0,u1,tau] -> [r0,r1,time]`. The stock clock generator `r2` must have exponent zero for a physical x/y term to be comparable. Any nonzero `r2` exponent is reported as a stock-only term rather than silently projected away.
