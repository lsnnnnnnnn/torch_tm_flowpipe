# Plot-spec contracts

`quad_legacy_author_contract.json` is a v1 archived-author overlay. It is
explicitly unbound and is retained for faithful visualization of historical
saved artifacts.

`quad_archcomp26_shared_content.json` is a v2 content contract. It binds the
benchmark instance, horizon, coordinates, and declared official source hashes,
and requires every plotted observer file to carry a matching instance identity
through its verified sidecar. This prevents an unbound historical series from
being relabelled as the official instance. The official QUAD
dynamics/controller selection and the complete execution contract nevertheless
remain unresolved, so an instance-bound plot alone is not an ARCH-COMP 2026
result.

V2 export requires `--instance-id` and fails closed on an instance or horizon
mismatch. The current renderer accepts only continuous-time v2 specs and only
axis-aligned property regions. Discrete index sets, halfspaces, and nonlinear
coupled constraints require explicit future schema/renderer support; they must
not be approximated as boxes.

`archcomp26_status.json` accounts for all 16 non-VCAS instances without
claiming that an execution contract is resolved. It separates materialized or
extractable axis-aligned content from eight fail-closed cases where the current
renderer cannot express the property or authoritative temporal semantics still
conflict.
