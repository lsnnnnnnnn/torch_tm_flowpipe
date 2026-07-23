# Representation and order semantics

This benchmark does not equate three similarly named flags. It audits the
retained monomials at the exact source paths used by the adapters and records
measured support in each per-run JSON file.

## Torch TM flowpipe

`order=1` is complete total degree at most one across local time and all
normalized initial generators. `Polynomial.mul_truncate` and
`Polynomial.truncate` select terms using `sum(exponents) <= order`
(`src/torch_tm_flowpipe/polynomial.py`). The Picard implementation evaluates
dropped polynomial products over the current domain and adds those bounds to
the interval remainder (`src/torch_tm_flowpipe/flowpipe.py`).

The dependency-preserving protocol passes each final Taylor model into the
next step, retaining generator dependence. The range-only supplement converts
each final model to a box and restarts, so it intentionally adds wrapping.
Both use fixed `h`, order 1, float64 CPU, zero symbolic-remainder history, no
cutoff, no adaptive step, and no rescue.

## Flow*

The installed toolbox's
`Computational_Setting::setFixedStepsize(step, order)` explicitly returns
false when `order < 2`
(`/srv/local/shengenli/flowstar/flowstar-toolbox/Continuous.cpp`). The adapter
checks that return value and records `unsupported_order` for both native and
strict order-one protocols. It never silently substitutes order 2.

For the labeled supplemental diagnostic, Flow* `Polynomial::ctrunc` compares
the `Term.d` total degree with the requested order and interval-evaluates
dropped terms into the remainder
(`/srv/local/shengenli/flowstar/flowstar-toolbox/Polynomial.h`). The diagnostic
therefore retains complete total degree at most two in local time and
normalized generators. It uses a fixed step and order, QR normalization,
symbolic-remainder size zero, cutoff `1e-15`, and remainder estimate `1e-4`.

## DiffReach

DiffReach's `QuadPoly` stores:

- `c`: constant;
- `L[..., 0]`: local time and `L[..., j>=1]`: generator-linear terms;
- `Lt[..., 0]`: local-time squared;
- `Lt[..., j>=1]`: local-time times generator.

The native `TRUNCATE_TO_AFFINE=True` path is affine only after the step's
final projection. Time integration first creates nonzero `Lt` terms, so its
transient effective degree is two. The implemented polynomial projection
bounds `Lt`, shifts its midpoint into `c`, and embeds its radius into reused
`L` generator coefficients (`/srv/local/shengenli/DiffReach/src/polynomial.py`).
`QuadTM.truncate_to_affine` keeps the existing TM remainder and calls that
projection (`/srv/local/shengenli/DiffReach/src/taylor_model.py`). This is not
the independent interval-remainder projection needed to establish strict
common-affine semantics, so the strict DiffReach protocol is recorded as
unsupported.

With `TRUNCATE_TO_AFFINE=False`, the retained form is a restricted
quasi-quadratic basis represented by `c`, `L`, and `Lt`; it is not the full
complete-total-degree-two basis. It is reported only as a supplement.

The benchmark's automated support test measures the initial model, the result
of a dynamics evaluation, the result of time integration, and the final
flowpipe segment. It asserts that the affine setting has nonzero transient Lt
support and zero final Lt support, and that the quasi-quadratic setting retains
final Lt support.

## Protocol mapping

| Protocol | Torch TM | Flow* | DiffReach |
| --- | --- | --- | --- |
| Native first-order setting | complete total degree 1, dependency preserving | requested fixed order 1; unsupported by API | affine-dynamics flag with transient Lt |
| Strict common affine | supported; same Torch run semantics | unsupported by API | unsupported; no tested independent-remainder Lt projection |
| Supplemental native representation | range-only restarts | complete total degree 2 diagnostic | restricted quasi-quadratic Lt basis |

Endpoint intervals fix local time at `h`. Tube intervals evaluate the same
stored segment over local time `[0,h]`. No endpoint is substituted for a
whole-segment tube.
