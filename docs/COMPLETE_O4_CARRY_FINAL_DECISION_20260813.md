# Complete-O4 carry final decision — 2026-08-13

The final decision is `NO_FIX_AUTHORIZED`.  The baseline, actual-path probe
equivalence, and lossless serialization bridge close; causal factor separation
is partial; the complete cross-operator same-prestate attribution does not
close; therefore the oracle and production candidate are not authorized.

## Ten direct answers

1. **Is the probe the stock path?** No.  It is a copied advance harness.  For
   this pinned contract it is now proven stock-equivalent over 1000 steps by a
   separate actual `ode.reach` run and exact actual-path observation.
2. **What is the plotted “zero”?** A positive local projection minimum, not
   numerical zero.  The four minima are `0.00861211181140531`,
   `0.026272600935460244`, `0.008888711363604695`, and
   `0.030888053869117083`, all well above `1e-9`.
3. **What causes the step-1 difference?** The first localized difference is
   local Picard coefficient arithmetic/grouping: 23/31 returned coefficients
   differ bitwise.  Old queue sources have not crossed a boundary yet, so the
   queue cannot be the sole cause.
4. **How much do queue and Horner contribute?** Inside Torch, Horner changes
   all published widths from step 3 and extends 632→636; at step 632 it reduces
   endpoint x/y by about 0.12459/0.31534.  The diagnostic queue changes only
   segments from step 2, adds about 0.10016/0.06520 at step 632, and changes
   neither endpoints nor horizon.  Inside Flow*, Q1/Q2/Q10 accept
   620/640/685 versus Q100's 1000.
5. **What is the first decision-relevant line/operator?** A unique causal line
   is unresolved.  The earliest candidate boundary is Flow* local Picard
   construction at `Continuous.cpp:2328-2343` versus Torch
   `dense_polynomial_picard`; the accumulated difference later reaches Torch's
   subset decision in `dense_picard_validate_step`.  Without the two cross
   same-prestate cells, naming one line as the cause would overclaim.
6. **Is the lossless bridge genuinely bidirectional?** Yes for exact schema
   import/export: 24/24 Flow* byte roundtrips and continuations, plus a
   byte-exact Torch→Flow*→schema roundtrip.  This does not mean the two full
   operators accept each other's different-dimensional state contracts.
7. **Is the source-ledger oracle closed?** No; Gate F was not authorized.
8. **Is a candidate authorized?** No.  L1/L2/L3 are all not run; diagnostic
   Horner/queue modes remain feature-gated and do not change the default.
9. **What is the evidence class?** Exact-rational micro fixtures and canonical
   byte equality are formal/discrete; MPFR-directed bounds and dyadic
   roundtrips are directed-numerical; horizons and widths are deterministic
   empirical results; Torch factorial modes are diagnostic; unique cross-tool
   source attribution is unresolved.
10. **What is the one next action?** Design a shared full operator sub-contract
    that both engines can consume without dropping `t`, `Phi_L/J`, term
    identity, or remainder categories, then rerun the two missing
    cross-operator same-prestate cells.  Only a closed Gate E may unlock the
    independent outward oracle.

## Allowed outcomes selected

- `BASELINE_CONCLUSIONS_REPRODUCED`
- `FLOWSTAR_WIDTH_MINIMUM_POSITIVE_NOT_NUMERICALLY_NEAR_ZERO`
- `STOCK_COPIED_PROBE_EQUIVALENCE_CLOSED`
- `CAUSAL_FACTOR_SPLIT_PARTIAL`
- `SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE`
- `SOURCE_MECHANISM_CANDIDATES_LOCALIZED_CAUSAL_SPLIT_OPEN`
- `SOURCE_LEDGER_ORACLE_INCOMPLETE`
- `NO_FIX_AUTHORIZED`

The feature-gated Horner and `flowstar_linear_v2` paths are retained only as
diagnostic machinery.  The legacy default is unchanged, and no L1/L2/L3
production claim is made.
