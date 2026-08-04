# Native reproduction standard

This branch uses *native reproduction* in a narrow, auditable sense: the source
identity, author/stock entrypoint, mathematical inputs, configuration and success
condition are fixed before execution.  A process exit code is evidence about the
process, not by itself evidence that the requested horizon, property or
certificate completed.

## Four independent conclusions

Every registry row reports these separately:

1. `reproduction_status`: whether a saved reference outcome was reproduced;
2. `completion_status`: whether the native process reached the requested horizon;
3. `property_status` and `certificate_status`: what property/certificate the
   native output actually exposes;
4. `soundness_level`: `formal`, `empirical`, or `unknown` for the result as run.

`primary_comparison_eligible` is a fifth, explicit gate.  It is false for every
partial/failed run, patched diagnostic, source-unknown result, mismatched workload,
or result whose soundness/correctness gate is unresolved.

## Reproduction statuses

The only permitted values are:

- `reproduced_exact`: selected identity and result fields are exact;
- `reproduced_with_declared_tolerance`: all selected fields pass a tolerance that
  existed in the author's artifact or code before this run;
- `reference_failure_reproduced`: a saved native failure boundary/failed
  certificate is reproduced; it is not a success;
- `native_run_completed_reference_unavailable`: the official/native run completed,
  but the upstream repository supplies no raw reference artifact;
- `native_algorithm_failed`, `native_unsupported_configuration`,
  `environment_failed`, `build_failed`, `reference_command_ambiguous`,
  `source_identity_unknown`, `patched_diagnostic_only`, and `not_attempted` have
  their literal meanings.

No status is inferred from a nearby endpoint, tube, segment, cache, generated
harness, or another backend.  Missing fields remain missing.

## Required evidence chain

Each attempted native row has a driver-produced `command.json` with literal argv,
cwd, environment overrides, UTC start/end, timeout, exit code, stdout/stderr paths
and hashes.  Each `reproduced_*` or `reference_failure_reproduced` row additionally
requires:

- exact repository SHA and `source_changed=false` when claiming exactness;
- the native entrypoint and nonempty config/input hashes;
- hashed fresh and reference artifacts;
- a hashed comparison result that names every comparison scope;
- for tolerance comparisons, both the value and its pre-existing source.

Runtime and peak memory are excluded from numerical reproduction unless the
reference declares a comparison contract.  Different devices, cold/JIT/core
boundaries, partition counts, effective bases, reached horizons, or properties are
never converted into a speedup or winner.

The machine enforcement is
`scripts/native_reproduction/validate_registry.py`.  It re-hashes command logs and
artifacts and rejects missing evidence, dirty exact claims, partial completion
claims, formal claims from empirical/unknown rows, and diagnostics in the native
table.

## What is not native reproduction

Adapters, rewritten models, basis translations, generated Flow* harnesses,
endpoint repair, post-hoc hulls and observation patches cannot establish a native
reproduction row.  A pre-existing patched trace may be re-read only in the
separate `diagnostics` section and must remain `patched_diagnostic_only`.

The only new scripts on this branch are command recording, provenance collection,
raw comparison/summarization and schema validation.  They do not implement an ODE,
Taylor-model backend, controller, basis, reset, or repair.

## Correctness is a separate gate

A successful stock Flow* benchmark does not erase the scalar-affine
under-enclosure diagnostic.  A completed DiffReach run whose returned flag covers
only initial contraction does not prove every roundoff-inclusive refinement
self-map.  Those backend-wide blockers remain visible in soundness and comparison
eligibility even when reproduction or completion succeeds.
