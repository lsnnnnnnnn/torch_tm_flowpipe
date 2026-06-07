# Flow* Horner Insertion Source Map

Scope: this maps only the Flow* normal insertion path relevant to normalized composition. It is a clean-room behavioral map, not copied Flow* source.

## Local Source Files

- `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Polynomial.h`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp`
- `/srv/local/shengenli/flowstar/flowstar-toolbox/expression.h`

## Vector Dispatch

`TaylorModel.h:3378-3388` implements the scalar-order `TaylorModelVec::insert_ctrunc_normal` dispatch. It receives an outer vector `this`, inserted Taylor models `vars`, precomputed inserted polynomial ranges `varsPolyRange`, `step_exp_table`, `numVars`, scalar `order`, and `cutoff_threshold`. It clears the result vector, calls scalar insertion for each outer component, and appends each component result.

`TaylorModel.h:3419-3429` is the vector-order overload. It has the same inputs except per-component `orders`; each component uses its own order. No range logic is done at the vector layer.

## Horner Scalar Insertion

`TaylorModel.h:4213-4243` is the non-diagnostic Horner normal insertion. It receives one outer Horner form, inserted Taylor models, inserted polynomial ranges, step powers, variable count, order, and cutoff. The constant part initializes the result. The first nonconstant branch is the time branch: it recursively inserts the time coefficient, multiplies by time, scales its remainder by the step interval, applies normal truncation, and adds it to the result. State branches recursively insert the coefficient for state variable `i - 1`, multiply by `vars.tms[i - 1]`, use `varsPolyRange[i - 1]` for inserted polynomial range accounting, apply truncation/cutoff inside multiplication, and add the result.

`TaylorModel.h:4247-4290` is the diagnostic overload with `intermediate_ranges`. It receives the same insertion inputs plus a list that records intermediate range accounting. For the time branch it records the normal truncation interval. For each state branch it records three items: the inserted coefficient polynomial range, the inserted right-map polynomial range, and truncation plus cutoff uncertainty from the multiplication.

`TaylorModel.h:4294-4342` specializes the diagnostic overload for interval Horner coefficients into real Taylor models. It additionally splits interval constants into midpoint plus interval remainder and records that interval before the same time/state branch recursion.

## Multiplication Accounting

`TaylorModel.h:797-825` defines `mul_insert_ctrunc_normal` without explicit diagnostics. It receives the left Taylor model, the inserted right Taylor model, the right model polynomial range `tmPolyRange`, step powers, order, and cutoff. It multiplies polynomial parts, evaluates the left polynomial over the normal domain when the right remainder is nonzero, uses `tmPolyRange` when the left remainder is nonzero, adds the interval-times-interval term, then calls normal truncation and normal cutoff.

`TaylorModel.h:829-867` is the diagnostic variant. It exposes the left polynomial range as `tm1` and exposes truncation plus cutoff as `intTrunc`. The uncertainty components are:

- dropped high-order terms from normal truncation
- cutoff terms from normal cutoff
- left polynomial range times inserted remainder
- inserted polynomial range times left remainder
- left remainder times inserted remainder

This is the key source-guided difference from the current PyTorch direct sparse substitution: Flow* accounts for these terms at each Horner multiplication, not only after a fully expanded substitution.

## Polynomial Truncation And Cutoff

`Polynomial.h:1530-1551` defines `ctrunc_normal`. It removes terms whose total degree exceeds `order`, stores them in a temporary polynomial, and evaluates that dropped polynomial with normal-domain interval evaluation using the step power table. The result interval is returned to the Taylor model remainder path.

`Polynomial.h:1791-1810` defines `cutoff_normal`. It removes coefficients contained in the cutoff threshold, stores removed terms in a temporary polynomial, and evaluates that temporary polynomial with normal-domain interval evaluation. This interval is added to truncation/cutoff uncertainty by the Taylor model layer.

## Callers In Continuous Flowpipes

`Continuous.cpp:415-420` implements scalar-order `Flowpipe::compose_normal`. It computes `tmvPolyRange` from the current right map with `polyRangeNormal`, then inserts the preconditioned map `tmvPre` through `tmv` using `insert_ctrunc_normal`.

`Continuous.cpp:422-427` is the per-output-order overload. It uses the same `tmvPolyRange` source and vector dispatch but supplies component orders.

`Continuous.cpp:428-440` is the output-axis overload. It computes polynomial ranges and inserts only selected preconditioned components.

`Continuous.cpp:857-887` shows the normal advance path for real-valued ODE expressions. It evaluates the previous pre map at the step end, removes constants to form an origin-centered reset, computes normal polynomial ranges for the right map, and calls `insert_ctrunc_normal` to compose the next flowpipe initial map.

`Continuous.cpp:1045-1074` is the interval-expression advance overload with the same reset insertion pattern.

## Intermediate Ranges In Validation

`Continuous.cpp:963-1014` creates `intermediate_ranges`, passes it into Picard normal construction, and later reuses it to compute remainder bounds. The list is populated by insertion and expression diagnostics rather than by generic interval evaluation.

`TaylorModel.h:3616-3634` routes Horner-form ODEs through `HornerForm::insert_ctrunc_normal(..., intermediate_ranges, ...)` before time integration.

`TaylorModel.h:3638-3647` starts the remainder recomputation path by iterating over the recorded `intermediate_ranges`.

`expression.h:1540-1549` shows the same three-part intermediate range pattern for expression multiplication: left polynomial range, right polynomial range, and truncation/cutoff interval. `expression.h:1554-1569` applies the same pattern after reciprocal expansion for division.

## Behavioral Difference From Current PyTorch Direct Insertion

Current PyTorch normalized insertion expands each outer sparse term through the right map, accumulates a direct composed Taylor model at a high work order, and then performs one final truncate/cutoff step. That direct path records final truncation, cutoff, composed polynomial range, and output remainder.

Flow* normal insertion recursively processes Horner branches. The time branch is multiplied by the step variable and normally truncated immediately. Each state branch is recursively inserted, then multiplied by the corresponding inserted right map with intermediate range accounting. The right-map polynomial range comes from `tmvPolyRange`; the coefficient polynomial range is evaluated normal-domain at the multiplication stage; dropped and cutoff intervals are added immediately.

## Clean-Room Implementation Target

The PyTorch clean-room target is an opt-in diagnostic and then an optional reset mode:

1. Preserve the existing direct sparse substitution as the default.
2. Implement a separate Horner diagnostic that receives the same outer Taylor model/vector, inserted right map, order, cutoff, domain, and optional time-variable index.
3. Recursively group the outer polynomial by variable powers to emulate Horner branch order.
4. Treat the first variable as a time branch only when an explicit time variable is present; endpoint/right-map reset diagnostics normally have no local time variable after endpoint substitution.
5. At every multiplication by an inserted right-map Taylor model, compute and record:
   - kept polynomial range
   - dropped-term truncation range
   - cutoff range
   - coefficient polynomial range times inserted remainder
   - inserted polynomial range times coefficient remainder
   - remainder times remainder
6. Return both direct and Horner results, stage ranges, and top uncertainty components.
7. Use Horner output for production reset only through explicit `reset_mode=normalized_insertion_horner`.
8. Keep reports clear that this is clean-room behavior and not a verbatim Flow* port.
