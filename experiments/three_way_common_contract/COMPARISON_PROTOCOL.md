# Comparison protocol

## Scientific contract

The experiment is a tool-level comparison under common external contracts.
The tools do not expose a literal common internal order:

| Tool | Local construction | Minimum/native setting |
| --- | --- | --- |
| Torch TM | complete total degree in local time and state generators | order 1 |
| DiffReach affine flag | affine final support with transient restricted `t²` and `t*z` support | `TRUNCATE_TO_AFFINE=True` |
| Flow* | complete total degree in local time and normalized generators | fixed order 2 (order 1 is rejected) |

`benchmark_spec.yaml` is the sole benchmark definition. Each adapter evaluates
that sparse polynomial specification rather than maintaining separate
hand-written ODEs.

## Protocol A: `one_step_common_input`

Every tool receives the same ODE, state order, initial componentwise box,
fixed `h`, and horizon `[0,h]`. The adapters export the whole-segment tube and
the endpoint at `t=h`. No previous segment exists.

## Protocol B: `multi_step_common_box_carry`

At each boundary the accepted endpoint is interval-evaluated to a componentwise
axis-aligned box. The next native segment starts from precisely that box. There
is no inflation, cross-state generator carry, symbolic remainder carry,
higher-order polynomial carry, or previous local-time variable.

This isolates the local integrators under the same wrapping/reset contract.

## Protocol C: `native_low_order`

Torch carries its dependency-preserving order-1 Taylor model. DiffReach carries
its upstream normalized symbolic representation, with affine-flag and default
restricted quasi-quadratic variants reported separately. Flow* carries the
accepted fixed-order-2 Taylor-model flowpipe with native QR normalization.

This is a practical native benchmark, not a representation-controlled result.

## Validation and extraction

- Torch requires `FlowpipeSegment.status == "validated"` from its public native
  Picard-growth validator.
- DiffReach calls the saved upstream `CT_Dyn_Reach.step_once` method. Its
  returned initial Picard contraction flag must hold componentwise. A failed
  flag stops the run.
- Flow* calls public `Flowpipe::advance`. Order 1 must be rejected and order 2
  accepted by `setFixedStepsize`. After a successful advance, the adapter
  restores the configured candidate remainder that passed the initial Picard
  self-map test. This audited workaround avoids exporting the stock
  un-revalidated refinement image.

Endpoints are evaluated with local time fixed to `h`; tubes retain local time
over `[0,h]`. The collector checks that every exported endpoint is contained
by its corresponding tube.

## Correctness gates

The strict collector requires:

1. Complete configuration coverage.
2. Identical point evaluations of every canonical ODE.
3. Identical initial boxes and state ordering.
4. Distinct endpoint/tube extraction with endpoint containment.
5. Zero Riccati and harmonic one-step exact violations.
6. Zero analytic violations in any successful Riccati/harmonic row.
7. Deterministic DOP853 trajectory containment for all validated rows.
8. No failed native validation labeled as success.
9. Flow* order guard and workaround labels on every run.
10. DiffReach provenance resolving the class, Picard routine, and Taylor-model
    class to the unchanged upstream repository.

Common-time tables emit `validation_failed` when a method has no accepted
endpoint at a checkpoint. Earlier widths are never substituted. Separate
failure-horizon rows retain each method's own last valid time and width.

## Runtime contract

Flow* compilation, DiffReach JIT compilation, first execution, steady
per-step execution, and Python orchestration are distinct fields. Torch has
zero build/JIT time and reports eager first/steady execution. No ambiguous
single total-runtime ranking is generated.
