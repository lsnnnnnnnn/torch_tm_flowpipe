# Plot-spec contracts

`quad_legacy_author_contract.json` is a v1 archived-author overlay. It is
explicitly unbound and is retained for faithful visualization of historical
saved artifacts.

`quad_archcomp26_shared_content.json` is a v2 content contract. It binds the
benchmark instance, horizon, coordinates, and declared official source hashes,
and requires every plotted observer file to carry matching benchmark,
instance, coordinate-order, step-size, and expected-step identities through
its verified sidecar. This prevents an unbound historical series from being
relabelled or placed on a fabricated axis. The official QUAD
dynamics/controller selection and the complete execution contract nevertheless
remain unresolved, so an instance-bound plot alone is not an ARCH-COMP 2026
result.

V2 export requires `--instance-id` and fails closed on an instance or horizon
mismatch. The current renderer accepts only continuous-time v2 specs and only
axis-aligned property regions. Linear halfspaces require the explicit v3
affine-threshold form below. The Docking radial-speed inequality uses the
single fixed v4 form described below. Discrete index sets and every other
nonlinear coupled constraint still require future schema/renderer support;
none may be approximated as boxes.

`acc_safe_distance_archcomp26_content.json` is the narrowly scoped v3
extension. V3 retains the v2 source/run binding and continuous-time rules, and
adds exactly one affine coordinate of the declared raw coordinates plus one
scalar threshold property region. ACC uses
`safe_distance_margin = x1 - x4 - 1.4*x5 - 10` and the audited all-times
condition `safe_distance_margin >= 0`. Saved raw-coordinate boxes are mapped
with outward binary64 interval arithmetic; correlations between source
coordinates remain unavailable and are disclosed in every rendered artifact.
V3 is not an arbitrary expression language and does not support nonlinear
derived coordinates or state-state projections of a derived coordinate.

`docking_constraint_archcomp26_content.json` is the narrowly scoped v4
extension. V4 adds exactly one `radial_speed_margin` coordinate with the fixed
form `offset + radial_gain*hypot(position) - hypot(velocity)`, two declared
position coordinates, and two declared velocity coordinates. Docking uses
`0.2 + 0.002054*sqrt(sx^2 + sy^2) - sqrt(sx_dot^2 + sy_dot^2)` and the audited
all-times threshold `docking_safety_margin >= 0`. Each saved lane box is mapped
independently with outward binary64 operations before lane intervals are
hulled. The result is a conservative interval extension with source-coordinate
correlations unavailable, not an exact nonlinear reachable image or a
property certificate. V4 is not a general nonlinear-expression language. The
one-second control period is not treated as the solver integration step, and
the content contract does not imply that 40 numerical frames exist. The
official requested state-1-over-time view remains the primary plot; this
margin view is only an auxiliary property plot once an identity-bound Docking
series exists.

`archcomp26_status.json` accounts for all 16 non-VCAS instances without
claiming that an execution contract is resolved. Eight source-audited,
axis-aligned continuous-time content contracts are materialized as v2 specs,
ACC is materialized as the single affine-threshold v3 spec, and Docking as the
single fixed radial-speed v4 spec. The other six stay fail-closed because the
current renderer cannot express the property or authoritative temporal
semantics still conflict.
