# Native all-box adaptive remainder construction — 2026-10-04

This is the isolated construction-only application of the verified
[offline endpoint plan](../native_quad_allbox_adaptive_remainder_plan_20261004_001/README.md).
It has no CROWN server, neural-network evaluation, ODE reach call or GPU work.
The native production gate remains **CLOSED** regardless of the construction
result: every row still assumes that the original real-affine CROWN bounds
are valid on its saved RPC box.

Execution status is determined only by `START.json`, the raw stage exit-code
files, `RESULT.json`, and `EXIT.json` when those receipts exist. The initial
subagent upload was blocked before execution by automatic authorization
review; [UPLOAD_REVIEW.json](UPLOAD_REVIEW.json) records that event. It is an
authorization event, not a numerical refusal or a benchmark attempt. Preparing
these sources and the plan does not establish a native result.

## Isolated change and refusal rules

[source.patch](source.patch) compares the new program with the saved
`native_quad_allbox_correlated_remainder_gate_20261003_001/allbox_control_trace.cpp`.
The source partition, input TM construction, transported slopes, conditionally
corrected biases, double center/radius operations, and output polynomial
construction are unchanged. It removes the old fixed lower pad.

Before modifying each output, the native program compares every source/domain,
input/RPC model, original output polynomial, original remainder, slope, bias,
center and radius to the saved full-precision trace. It compares canonical JSON
representations, preserving the exact hexadecimal values without conflating
JSONcpp's parsed signed/unsigned integer types with numerical changes. A first
mismatch stops the process and retains the completed trace prefix.

The new operation assigns the exact planned binary64 pair directly as
`Interval(lo,hi)` to the final control remainder. The new pair must be finite,
ordered and contain the old remainder; the saved resulting MPFR endpoints must
exactly equal the plan. No extra rounded interval addition is used.

The independent Python checker lifts the actual native endpoints and saved
coefficients to rational numbers. It compares all unchanged fields again,
checks the actual pair against both [PLAN.csv](PLAN.csv) and
[PLAN_ENDPOINTS.json](PLAN_ENDPOINTS.json), verifies the corrected real-CROWN
coefficient transfer, and evaluates the affine reference at every active
normalized-box vertex. It also proves nearest-outward endpoint minimality
under preservation of the original remainder. `CHECK_ROWS.csv` is flushed after
each passing output. Any first failure stops checking; a partial trace or
partial ledger never qualifies unexamined rows.

The copied [exact_checker_primitives.py](exact_checker_primitives.py) is the
new offline plan's exact arithmetic and schema checks. It does not import or
execute an older experiment gate. Its synthetic check tests directed rounding,
successful affine inclusion, and deliberate rejection of an insufficient
remainder.

## Execution and artifacts

[INPUT.json](INPUT.json) names the existing read-only library and trace and the
specific mutation. [run_once.sh](run_once.sh) refuses existing START, RESULT
or native trace files, records START before source preflight, and performs:

1. `/usr/bin/g++-15` with the existing recentered static archive, CPU 12,
   maximum 300 s; no library rebuild or source-library mutation.
2. One new construction-only native process, CPU 12, maximum 180 s.
3. One exact-rational native-result checker, CPU 12, maximum 300 s.

The raw `build_exit_code.txt`, `run_exit_code.txt`, `check_exit_code.txt`,
`wrapper_exit_code.txt`, stage logs and `EXIT.json` distinguish stage failures
and unattempted later stages. `ALLBOX_TM_TRACE.jsonl` preserves the full new
trace. `RESULT.json` reports only completed checked boxes and rows.

No old failed fixed-pad gate is rerun. No hashes or digest verification are
performed. Even a complete successful result only qualifies this saved
first-batch conditional native remainder construction. It cannot establish
independent NN/CROWN soundness, a plant trajectory, later controller calls,
full-period or full-horizon containment, or production octagon readiness.

## Local static review before authorized upload

An independent source-only review found a refusal-location defect in
`check_replay.py`: an input-stage failure in a later lane could retain the
previous lane's output index. The local code now resets `output = None` at the
start of every lane. Python syntax parsing passed; this was not a checker run
or a numerical experiment. The defect affected diagnostics, not acceptance.

The review examined frozen-field comparisons, hexadecimal endpoint transport,
remainder replacement, proof scope, and stage-exit handling. No other definite
source defect was found. It did not compile or execute the native program,
validate every plan row numerically, or establish a native result. An external
checker timeout may be recorded by the wrapper fallback and the completed CSV
prefix without an exact lane/output refusal coordinate.

The user subsequently authorized Huan server experiments explicitly: “huan服务器你随便用啊做实验，你别动人家github咋都行”. The same eight-file upload and single bounded replay were then approved and executed; no Huan or Xiangru GitHub repository was accessed. The earlier authorization refusals remain recorded in `UPLOAD_REVIEW.json` as historical events.

## Completed authorized replay

The raw remote receipts show a single run from `2026-10-04T08:38:31.453880+00:00`
to `2026-10-04T08:39:16.299003+00:00`, wrapper PID `176147`. Build, native run,
exact checker and wrapper each exited `0`. The checker took
`40.19774992499879` seconds and reported
`CONDITIONAL_NATIVE_ADAPTIVE_REMAINDER_CONSTRUCTION_VERIFIED`:

- 1,024 source boxes checked in full.
- 3,072 native output rows checked in full.
- 196,608 exact affine vertex comparisons.
- No first-refusal coordinate and no unchecked prefix represented as complete.

`START.json`, `RESULT.json`, `EXIT.json`, stage exit files, logs,
`ALLBOX_TM_TRACE.jsonl`, and `CHECK_ROWS.csv` are the raw server records.
[DOWNLOAD_AUDIT.json](DOWNLOAD_AUDIT.json) records direct byte equality for
all eight uploaded input files and all fifteen raw output copies against a
second read of the same server files. The local structural audit confirms
1,024 ordered trace lanes, 3,072 ordered ledger rows, 196,608 recorded vertex
checks, nonnegative exact rational margins, ledger endpoints equal to native
trace endpoints, raw RESULT equal to checker stdout, and all stage exits zero.
This audit did not rerun the numerical checker. The remote compiled executable
is retained only on the server. macOS tar added eight 163-byte AppleDouble
`._*` metadata files on upload; these were not program inputs and are not
part of the local evidence or publication.

This result proves application of the saved endpoint plan in the existing
native control-TM constructor, with the frozen input/polynomial/transport
fields unchanged and every planned remainder endpoint checked. It remains
conditional on the original real-affine CROWN bounds. It supplies no
independent NN/CROWN certificate, ODE propagation, later controller calls,
full-horizon NNCS containment, or safety verdict. **Native production gate:
CLOSED.** This diagnostic is not a new benchmark cell or an old rerun.
