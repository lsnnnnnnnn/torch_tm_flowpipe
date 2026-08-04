status: diagnostic
valid_for_commit: unknown
superseded_by: docs/EXPERIMENT_PROTOCOL.md
allowed_use: diagnostic only

# DiffReach endpoint semantics audit

The adapter saves and calls the actual upstream
`src.reachability.CT_Dyn_Reach.step_once`. That path constructs the local
Taylor model, runs upstream Picard remainder validation, returns the final local
model and symbolic/affine parameterization, and reuses them for native carry.

The audit composes the returned local TM with the upstream parameterization:

- `diffreach_tube` evaluates time and generators over their complete domains;
- `diffreach_endpoint_raw` fixes only the time coordinate to `h`;
- the final interval remainder is the returned composed `QuadTM.R`;
- `L`, `Lt`, generator radii, and symbolic state are carried only in native
  protocols.

There is no endpoint-specific residual recomputation. Tube and endpoint are
evaluations of the same upstream segment object.

The adapter changes one numeric behavior: upstream's hard-coded float32 linear
TM constructor default is replaced with explicit `jnp.float64` while JAX x64 is
enabled. It does not replace Picard construction, remainder validation,
polynomial arithmetic, or reset semantics. A fail-fast import shim supplies
only unused optional neural-bound names when `jax_verify` is absent.
`diffreach_provenance.json` records the callable source and these facts.

For affine Riccati at `h=0.01`, the raw endpoint is
`[-2.507513055141797e-05, 0.1001001879815906]`, width
`0.10012526311214201`, with remainder width
`2.5150489001753456e-05`. The restricted quasi-quadratic raw width is
`0.10012522561219696`. Both contain the analytic endpoint in the audited case.
