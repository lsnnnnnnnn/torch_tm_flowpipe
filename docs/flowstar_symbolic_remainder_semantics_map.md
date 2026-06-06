# Flow* Symbolic Remainder Semantics Map

This map is based only on the local Flow* source under `/srv/local/shengenli/flowstar/flowstar-toolbox`.
It is a clean-room behavioral map for the PyTorch prototype; it does not copy Flow* implementation code.

## Source Locations

| Mechanism | Local source |
| --- | --- |
| Symbolic_Remainder declaration | `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:19-35` |
| Symbolic_Remainder constructors, reset, copy | `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:12-52` |
| Fixed-step real ODE symbolic advance | `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:2123-2418` |
| Adaptive-step real ODE symbolic advance | `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:2715-3040` |
| Symbolic reach loops and queue reset | `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.h:832-972` |
| Normal insertion and Picard validation helpers | `/srv/local/shengenli/flowstar/flowstar-toolbox/TaylorModel.h:3378-3488`, `3616-3736` |
| Range/safety materialization paths | `/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp:4240-4250`, `4440-4555`, `9491-9519`, `10009-10037` |

## Symbolic_Remainder State

### Flow* Behavior

`Symbolic_Remainder` stores four public fields.
`J` is a `std::vector<Matrix<Interval> >`; each entry is an interval column for a step-local symbolic remainder source.
`Phi_L` is a `std::vector<Matrix<Real> >`; each entry stores a linear propagation matrix for queued symbolic remainders.
`scalars` is a `std::vector<Real>`; it records the inverse coordinate scaling used when the endpoint map is normalized.
`max_size` is an unsigned queue limit.

The default constructor only initializes `max_size` to zero.
The constructor from an initial flowpipe initializes one scalar per state dimension to one and stores the requested queue size.
The copy and assignment paths copy `J`, `Phi_L`, `scalars`, and `max_size`.
`reset(dim)` clears `J` and `Phi_L` and recreates `dim` unit scalars; it preserves `max_size`.

### Current PyTorch Symqueue Behavior

`FlowstarSymbolicRemainderQueue` mirrors the shape with immutable `J`, `Phi_L`, `scalars`, and `max_size` fields.
The old `normalized_insertion_symqueue` mode materializes propagated old queue width into the next ordinary reset Taylor-model remainder.

### Difference

The old PyTorch mode has the right queue-shaped state, but it merges propagated symbolic width into the ordinary initial remainder channel before target-remainder validation.
Flow* stores queue entries separately and only adds them to the local initial-set range construction around the symbolic advance.

### Required Clean-Room Change

Keep queue state separate from the ordinary Picard target channel.
For PyTorch this means an opt-in split mode where propagated queue width is carried as symbolic/output width and is not used as the seed remainder for the target precheck.

## Fixed-Step Symbolic Advance

### Flow* Behavior

The fixed real-ODE overload begins by evaluating the current flowpipe at the step end and removing the constant part.
It decomposes the endpoint Taylor map into linear and other parts.
The linear part is extracted into `Phi_L_i`, and a copy named `linear_x0` is retained before scaling.
`Phi_L_i.right_scale_assign(symbolic_remainder.scalars)` applies the previous normal-coordinate scalars.

Older queue entries are propagated by multiplying existing `Phi_L` entries by the new `Phi_L_i`, pushing the new matrix, and accumulating `J_i` as the sum of propagated older `J` columns.
`J_ip1` is the current step's new interval remainder column.
When the queue is nonempty, Flow* inserts only the non-linear endpoint part through `insert_ctrunc_normal`, saves that insertion remainder as `J_ip1`, adds the unscaled linear polynomial part back to the result, and sets the result remainder to `J_ip1 + J_i` for range construction.
When the queue is empty, Flow* inserts the full endpoint map and then extracts the resulting remainder as `J_ip1`.

`symbolic_remainder.J.push_back(J_ip1)` happens before the new local Picard flowpipe is validated.
The range of the local initial set is computed with the current ordinary insertion remainder plus propagated symbolic contribution.
The scaling vector is then recomputed from that range, and `symbolic_remainder.scalars` is updated.
The normalized result Taylor model is cut off with the initial simplification threshold.

Picard validation is then performed on `new_x0 = center + S*r`.
Flow* assigns `x.tms[i].remainder = tm_setting.remainder_estimation[i]`, computes one `Picard_ctrunc_normal` image, adds polynomial-difference uncertainty, and checks whether that Picard image remainder is a subset of the assigned target remainder.
The propagated queue width is not separately checked by a seed-remainder subset test against `tm_setting.remainder_estimation`.

### Current PyTorch Symqueue Behavior

The old PyTorch mode computes an inserted endpoint, builds a fresh normalized reset TM, propagates older interval columns through a linear map, and adds the propagated width as an ordinary remainder on the next reset TM.
The following call to target validation computes `seed_remainders = base_i.remainder + candidate_i.remainder` and rejects immediately if those seed remainders are not inside `[-1e-4, 1e-4]`.

### Difference

Flow* uses `J_i` to construct and scale the local initial range while keeping the Picard target check focused on the Picard image remainder.
Old PyTorch converts `J_i` into an ordinary reset remainder, so it can fail before Picard validation with `initial or cutoff remainder exceeds target remainder`.

### Required Clean-Room Change

For `normalized_insertion_symqueue_split`, keep the normalized reset TM's ordinary remainder clean.
Queue the current insertion remainder as the new `J_ip1`, carry older `J_i` in the queue, and add the materialized symbolic width to reported output/range boxes.
Record diagnostics proving the split: ordinary-only range width, symbolic contribution width, output-materialized width, total range width with symbolic, and target-checked width.

## Adaptive Symbolic Advance

### Flow* Behavior

The adaptive real-ODE overload repeats the same symbolic endpoint decomposition and queue update as fixed-step advance.
It then optionally changes the step size through `tm_setting.setStepsize(new_stepsize, tm_setting.order)` before target validation.
If the `Picard_ctrunc_normal` image remainder is not contained in the target remainder, Flow* shrinks the step and recomputes the Picard image, polynomial differences, and intermediate ranges.
The queue update and local initial range setup happen before the first containment check in the advance call.
When a smaller step is tried inside that same call, Flow* recomputes the Picard validation data for the new step size.

In the reach loop, an accepted flowpipe is pushed to the result list, `currentFlowpipe` is replaced by the accepted flowpipe, and the queue is reset only after acceptance if `J.size() >= max_size`.

### Current PyTorch Symqueue Behavior

PyTorch adaptive stepping retries by calling `flowpipe_step_from_tm` with halved `h_try`.
For the old symqueue mode, the propagated symbolic width was already materialized into the ordinary reset TM from the previous accepted step, so every retry sees that width in the seed-remainder target precheck.

### Difference

Flow* does not have a separate PyTorch-style seed precheck that rejects propagated queue width before the Picard image is computed.
The PyTorch old mode therefore rejects a local box/step that the Flow* one-step oracle can validate.

### Required Clean-Room Change

The split mode must keep propagated symbolic width out of ordinary seed remainders for every adaptive retry.
The same symbolic contribution must remain available for output materialization and diagnostics.

## Picard Validation With Symbolic Remainder

### Flow* Behavior

The target remainder is assigned in the symbolic advance by setting each Picard candidate remainder to `tm_setting.remainder_estimation[i]`.
`Picard_ctrunc_normal` computes a temporary image with truncation and cutoff uncertainty.
The polynomial difference between the temporary image and candidate is interval-evaluated over the normal domain and added to the temporary image remainder.
The subset check compares this temporary Picard remainder against the assigned target remainder.

The symbolic queue contribution participates in the local initial-set range and scaling used to create `new_x0`; it is not a separate reason to fail the target subset check as an initial or cutoff remainder.

### Current PyTorch Symqueue Behavior

The old PyTorch target validators first combine base and candidate remainders and require that combined seed remainder to be inside the target remainder.
Because the old symqueue reset materialized propagated symbolic width into the base remainder, it can fail before Picard residual or ctrunc validation is attempted.

### Difference

The old PyTorch check is stricter than the Flow* target subset check for symbolic queue contributions.
It treats symbolic queue width as ordinary target remainder rather than as carried symbolic/range width.

### Required Clean-Room Change

Do not count propagated symbolic queue width in the ordinary target seed precheck.
Keep ordinary Picard target validation at radius `1e-4` and separately report `target_checked_width` so the run cannot silently loosen the target.

## Range Evaluation And Safety Output

### Flow* Behavior

Flow* materializes remainder-like objects when evaluating flowpipe ranges, safety constraints, and plotting/Horner output.
The relevant paths add interval remainders and, when present in other flowmap variants, zonotope or intervalized remainder contributions into range boxes.
For the symbolic remainder advance itself, `result.tmv.tms[i].remainder = J_ip1 + J_i` is used to compute the local initial-set range and scaling before the Picard flowpipe is built.

### Current PyTorch Symqueue Behavior

Old PyTorch materializes propagated symbolic width into the next ordinary reset remainder, so reported range boxes include it, but target validation also sees it as ordinary.

### Difference

Old PyTorch preserves output conservatism at the cost of an overly strict target precheck.
The missing split is between range/output materialization and ordinary Picard target validation.

### Required Clean-Room Change

For split mode, materialize propagated symbolic width onto accepted segment output boxes after validation and before report rows are written.
Keep the next reset TM ordinary-only for target validation, and keep queue state alive so propagated symbolic width is not dropped.

## Mechanism Summary

| Mechanism | Flow* behavior | Current PyTorch symqueue behavior | Difference | Required clean-room change |
| --- | --- | --- | --- | --- |
| Queue state | Stores `J`, `Phi_L`, `scalars`, and `max_size`; reset clears queue and restores unit scalars. | Stores analogous queue data. | Shape is aligned. | Preserve state shape and reset behavior. |
| Endpoint decomposition | Splits endpoint into linear and non-linear parts. | Uses inserted endpoint and extracts linear coefficients for propagation. | Approximate but aligned in intent. | Keep clean-room linear propagation; document approximation. |
| Older `J` propagation | Propagates older columns through updated `Phi_L` and accumulates `J_i`. | Propagates older columns and materializes them on reset. | Materialization channel is wrong for target checking. | Carry as symbolic/output contribution in split mode. |
| Current `J_ip1` | Comes from insertion/truncation/cutoff remainder and is pushed to queue. | Current insertion remainder is queued for future propagation. | Aligned in intent. | Keep current insertion as symbolic candidate, not ordinary target seed. |
| Target check | Checks Picard temporary image remainder against `tm_setting.remainder_estimation`. | First checks seed remainder against target. | Propagated symbolic width can fail too early. | Exclude symbolic queue width from seed precheck. |
| Range output | Includes materialized remainder contributions in range/safety output. | Includes propagated width by ordinary reset materialization. | Conservative but channel-mixed. | Add symbolic contribution to output boxes after validation. |
| Queue reset | Reset happens after accepted flowpipe when queue length reaches `max_size`. | Same queue-size threshold. | Aligned. | Preserve default `max_size` behavior. |
