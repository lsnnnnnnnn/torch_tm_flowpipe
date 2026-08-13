# Flow* actual path versus copied probe — 2026-08-13

Status: `STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED` for the pinned frozen contract.

The old probe is not itself the stock entry point: it executes its local
`traced_advance_adaptive_symbolic(...)` copy.  This audit therefore added a
separate clean driver that calls the real public
`ODE::reach(...) -> ODE::reach_symbolic_remainder -> Flowpipe::advance(...)`
path, plus a second isolated Flow* worktree with read-only observation hooks in
that same path.  No copied advance body is present in the stock driver.

## Three-way result

| cell | executed path | result |
|---|---|---|
| F-stock-clean | pinned unmodified Flow* source, one-shot `ode.reach` | 1000 accepted steps, T=10 |
| F-stock-instrumented | actual path, guarded read-only pre/post-reset hook | bit-for-bit identical stock CSV to clean |
| F-copied-probe | probe-local copied advance with exact audit serialization | matches the actual path at every eligible comparison |

The clean and instrumented stock CSVs have the same SHA-256,
`e2b3898560c72d273d164f0056937208d547cc748fc821a2fd6a64cd442ce53d`.
Across all 1000 steps, accepted schedule, `h`/time binary64 values, endpoint,
one-segment tube, and prefix tube match.  Fifty-five time hex strings use a
different textual spelling such as `0x1p-2` versus
`0x1.0000000000000p-2`; parsing and re-encoding shows identical binary64 bits,
so this is formatting, not a numerical difference.

The actual hook also closes the internal comparison:

- 1000/1000 pre-reset `tmvPre`/`tmv` objects equal the copied probe's retained
  term support, binary-exact coefficients, and ordinary remainders;
- 999/999 post-reset actual states equal the next copied prestate, including
  `tmvPre`, `tmv`, domain, center/scale scalars, queue `J`, `Phi_L`, and
  `max_size`;
- there is no first bitwise, semantic, or decision-relevant difference within
  this scope.

This upgrades the copied trace to a stock-equivalent observation only for
Flow* `b85a3211748cb77b736fe4ad42ee02d8d2b81148` and the frozen VDP B1,
complete-O4, fixed `h=.01`, T=10, Q100 contract.  It is not a generic proof that
the copied harness will remain equivalent for another model, overload,
setting, or Flow* revision.

## Driver qualification

The eligible actual run is one `ode.reach` call over T=10.  A supplementary
driver that invokes `ode.reach` separately for each step also reaches 1000
steps, but its per-call horizon interacts with Flow*'s `THRESHOLD_HIGH`
handling.  It is therefore retained as a diagnostic and excluded from the
equivalence proof.

The full instrumented Flow* tracked diff, compiler/link commands, `ldd`, binary
hashes, raw outputs, and the derived 1000-row comparison are in
`outputs/flowstar_torch_causal_mechanism_closure_20260813/20260813T060020Z/00_identity_provenance`
through `05_copied_probe_equivalence`.
