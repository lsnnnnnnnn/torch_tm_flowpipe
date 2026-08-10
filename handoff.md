# Handoff: compiled fixed support and structured remainder S1

Date: 2026-08-10

Branch: `codex/structured-remainder-compiled-fixed-support-closure-20260810`

Run ID: `20260810T070908Z`

## Outcome

The previous evidence-package gap is closed and claim/soundness eligibility is
separated. The fixed-support object path now has a cached immutable kernel plan
and bit-exact functional tensor core. Fullgraph Inductor completes B64 T10 but
changes arithmetic, so it is an empirical performance-only lane with outcome
`FIXED_SUPPORT_COMPILE_SEMANTICS_CHANGED`.

The separate fixed-support outward CPU reference passes its independent exact
oracle but fails before T1, yielding
`FIXED_SUPPORT_FORMAL_SOUNDNESS_NOT_CLOSED`. The K16 structured-remainder S1
primitive and typed source decomposition pass analytic/conservation tests; a
local frozen-terminal attribution closes ordinary y while containing the full
image, but no S1 prefix state exists. The exact terminal result is
`STRUCTURED_REMAINDER_LOCAL_GATE_FAILED`, and no fresh horizon was started.

NAV/DR15 is absent in the pinned DiffReach tree. The required harmonic and
scalar Riccati fallback completes 100 steps on CPU/V100 B1/B64 and contains the
analytic endpoint hull: `GENERALITY_GATE_PASSED`, plant-only scope.

## Key measurements

| lane | CPU | V100 | qualification |
|---|---:|---:|---|
| frozen object B64 T10 warm median | 75.592882 s | 127.233569 s | ordinary empirical |
| functional eager B64 T10 | 46.335458 s | 118.174358 s | object-bit-exact |
| compiled B64 T10 stable warm | 5.038308 s | 6.926640 s | arithmetic changed; raw ratios only |

Compiled core synchronization is 0 plus one final decision sync, versus 1000
object inclusion gates. V100 profiling observes 369 kernel events per logical
step and no boundary `item`/scalar/`to`/`stack`/`index` calls. The compiled CPU
is faster than V100. B1 ordinary and compiled both first fail at step 536;
B64 partitions complete.

## Evidence map

- [manifest](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/manifest.json)
- [checksums](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/SHA256SUMS)
- [claim registry](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/claim_registry.csv)
- [compiled results](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/fixed_support_compiled_results.csv)
- [outward results](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/fixed_support_outward_results.csv)
- [terminal A/B](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/structured_terminal_ab.json)
- [stopped horizon ladder](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/structured_horizon_ladder.csv)
- [second systems](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/second_system_results.csv)
- [verification environment and commands](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/verification.json)
- [final pytest log](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/final_pytest.log)

## One next action

Thread the delivered S1 state and exact eligible-source removal through every
accepted complete-O4 boundary from t=0, include all active columns in endpoint
and tube publication, serialize the prefix state, and repeat the immutable
terminal A/B. Do not start a horizon ladder until that gate passes. For the
fixed compiler line, the bounded alternative is to preserve eager reduction
order or qualify an outward eager shadow; do not call the current raw timing
ratio a same-semantics speedup.

## Verification

- Full CUDA-enabled suite: `515 passed, 2 skipped in 203.54 s`.
- Focused local previous/current package regression: `3 passed in 14.92 s`.
- `python -m compileall -q src experiments tests`: exit 0.
- `git diff --check 05ae30b4..HEAD` and staged diff check: exit 0.
- Closure package: 221 repository-root checksum entries, all valid; every
  manifest path tracked and no absolute server path in the public surface.
- Remote package commit: `3f7d77aef446133a5fa51ba5f427bae905f17806`.
- Independent remote fresh clone: 221 checksums valid and the combined package
  link/tracking/clean-copy/rebuild suite is `3 passed in 16.29 s`.

The final handoff-only commit changes documentation, not code or package
bytes. See [status](docs/STATUS.md), [verification](outputs/structured_remainder_compiled_fixed_support_20260810/20260810T070908Z/verification.json),
and the run manifest for the auditable records.
