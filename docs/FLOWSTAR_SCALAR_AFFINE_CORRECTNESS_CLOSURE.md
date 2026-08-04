# Flow* scalar-affine correctness closure

Task status: `COMPLETE_DIAGNOSIS`

Selected outcome: **F — clean stock Flow* core behavior**

Flow* correctness gate: **OPEN**

Primary comparison eligible: **false**

## Frozen contract and provenance

The primary contract is

```text
x' = 1 + 2*x
x(0) in [0, 0.1]
h = 0.01 (fixed)
order = effective RHS order = 4
candidate remainder = [-1e-4, 1e-4]
cutoff = 1e-15
preconditioning = Flow* native diagonal scaling
symbolic remainder = disabled
interval precision = 53 bits
containment tolerance = none
```

The launch document's claimed full starting SHA,
`438ee68fd71fa6182eb66cac17229e20dd3cb7d3f`, has 41 hexadecimal characters.
The exact remote branch tip and selected worktree parent are the valid commit
`438ee68fd71fa6182eb66cac17229e20dd3cb7d3`. The new clean branch descends from
that commit; the dirty pre-existing worktree was preserved.

The stock checkout is the detached, tracked-clean
`/srv/local/shengenli/flowstar_stock_gcc11` at
`b85a3211748cb77b736fe4ad42ee02d8d2b81148`, remote
`https://github.com/chenxin415/flowstar`. Its untracked files are documented GCC
11 build outputs only. The exact environment and identities are:

- Docker image:
  `sha256:6549fefc0ae934982bf902f6a1f6ee9a2baf0def2ee763b278f914e4bbd096bf`;
- compiler: `g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0`;
- linker: GNU ld 2.38;
- `libflowstar.a`:
  `b5ff500af66354b0518cf12e7d951f4525f435e8e2d695cf84b91821992c9d9a`;
- official Van der Pol binary:
  `266ba4edf9b905a185efcae4f72c28f2a9ca34362c5960e2995ea0e2bb35d51f`;
- generated trace binary:
  `90b53a2b52b243263fe0680e36888505bdcd2338bb7d51d4376c87aa748ef5f9`;
- official scalar binary:
  `8102947d4e0cce8545c1c0f03d45cef71a3d8a4e8a764acef93e936b7869f7bf`;
- MPFR oracle binary:
  `3b9709f28fe6ba30f2535c392d557c0417dff8f3975a56fe52e45e1defb3b28a`.

Flow* is statically selected by the sole mounted `-L
/flowstar/flowstar-toolbox -lflowstar` path. `LD_LIBRARY_PATH` and all historical
audit behavior variables are unset. Dynamic linkage was captured with `ldd`.
The dirty GCC 15 compatibility checkout `/srv/local/shengenli/flowstar` has a
different archive hash, `3a658f95...`, and is explicitly recorded as unused.
Generated binaries were deleted after their hashes and linkage were recorded;
they are not committed.

The exact reproduction command is:

```bash
conda run -n py11 python experiments/flowstar_scalar_affine_closure/run_closure.py \
  --flowstar-root /srv/local/shengenli/flowstar_stock_gcc11 \
  --run-root outputs/flowstar_scalar_affine_correctness_closure/20260804T131445Z
```

The runner records every container compile/run command in `commands.jsonl`.
Both primary generated-stock runs produced byte-identical 38,975-byte stdout
logs with SHA256
`365b25cf4d26b6eab0a48c50370a572811fae36d89ed40eebea28b419aa0ea52`;
their parsed traces are also exactly equal. No timing fields are emitted.

## Independent outward oracle

Solving the affine ODE gives

\[
x(t;x_0)=\left(x_0+\tfrac12\right)e^{2t}-\tfrac12.
\]

On the frozen domain,
`dx/dx0 = exp(2t) > 0` and
`dx/dt = (1 + 2*x0) exp(2t) > 0`. Therefore the endpoint extrema are at
`(x0,t)=(0,0.01)` and `(0.1,0.01)`, while the whole-step tube extrema are at
`(0,0)` and `(0.1,0.01)`. No sparse-time sampling is used to infer the tube.

A separate C++ program evaluates the formula at 256-bit MPFR precision using
explicit `MPFR_RNDD`/`MPFR_RNDU` at every operation and converts to binary64 in
the same directed mode:

| object | lower | upper | binary64 hexadecimal |
|---|---:|---:|---|
| endpoint | `0.010100670013377904` | `0.1121208040160535` | `0x1.4afa8fb004c89p-7`, `0x1.cb3f2f2733eafp-4` |
| full tube | `-0` | `0.1121208040160535` | `-0x0p+0`, `0x1.cb3f2f2733eafp-4` |

The 100-decimal-digit outward fallback produces the same endpoint bounds. The
previous values `0.010100670013377895` and `0.11212080401605326` are reproduced
by deterministic RK4 with 1,000 substeps, but are secondary approximations—not
the analytic oracle.

Containment is strict, with no tolerance:

```text
lower_defect = max(0, exported_lower - oracle_lower)
upper_defect = max(0, oracle_upper - exported_upper)
max_defect   = max(lower_defect, upper_defect)
```

## Primary reproduction

| object | lower | upper | lower defect | upper defect | contained |
|---|---:|---:|---:|---:|---|
| generated `endpoint_raw` | `0.010100670333333329` | `0.1121208036666667` | `3.199554250016279e-10` | `3.4938679727147814e-10` | no |
| generated `full_tube` | `-2.265795929406518e-18` | `0.1121208036666667` | `0` | `3.4938679727147814e-10` | no |
| official accepted-right endpoint | `0.010100670332313124` | `0.11212080366544246` | `3.1995542153218093e-10` | `3.4938679727147814e-10` | no |

The official row is compared to an independently rerun MPFR oracle at the
official accepted right time `0.009999999999`, not at nominal `0.01`. The
official tube also fails on its upper bound by `3.4938679727147814e-10`.

`endpoint_raw` and the separately time-substituted `endpoint_collapsed` happen
to have equal numeric bounds in this case. They remain distinct fields.
`endpoint_tightened` is not a distinct stock field, and `repaired_hull` is
explicitly unavailable and was never computed. `last_segment`/`full_tube` use
the full accepted local-time domain and are not substituted for the endpoint.

## First containment loss

The generated observer first calls unmodified `Flowpipe::advance` and freezes the
accepted object. It then performs a read-only diagnostic replay of the exact
stock operations in `Continuous.cpp:857-1040`. The confined
`protected -> public` preprocessing shim exposes `Flowpipe::tmvPre`, `tmv`, and
`domain` for observation; it does not alter the library or numerical behavior.
The replayed polynomial terms and stored remainder are exactly equal to the
already accepted native object.

The full machine table is in `first_containment_loss.json`. The decisive rows are:

| path | stage | Flow* source | endpoint-corner hull | oracle object | lower defect | upper defect | contained | first loss |
|---|---|---|---|---|---:|---:|---|---|
| generated-stock | candidate remainder seed | `Continuous.cpp:961-969` | `[0.010000670333333331, 0.11222080366666669]` | exact endpoint corners | `0` | `0` | yes | no |
| generated-stock | combined cutoff/truncation Picard image | `Continuous.cpp:966-982`; `TaylorModel.h:3707-3724` | `[0.01009866899999999, 0.11212280507333336]` | exact endpoint corners | `0` | `0` | yes | no |
| generated-stock | validated image plus roundoff interval | `Continuous.cpp:969-1005` | `[0.010098668999999989, 0.11212280507333336]` | exact endpoint corners | `0` | `0` | yes | no |
| generated-stock | refinement 0 accepted TMV | `Continuous.cpp:1013-1029` | `[0.010100630306666662, 0.11212084369480002]` | exact endpoint corners | `0` | `0` | yes | no |
| generated-stock | refinement 1 accepted TMV | `Continuous.cpp:1013-1029` | `[0.010100669532799996, 0.11212080446722936]` | exact endpoint corners | `0` | `0` | yes | no |
| generated-stock | **refinement 2 accepted TMV** | `Continuous.cpp:1013-1029`; `TaylorModel.h:3728-3743` | `[0.010100670317322662, 0.11212080368267795]` | exact endpoint corners | `3.0394475825090694e-10` | `3.3337554938839276e-10` | **no** | **yes** |
| generated-stock | accepted `tmvPre` | `Continuous.cpp:1038-1040` | `[0.010100670333333329, 0.1121208036666667]` | exact endpoint corners | `3.199554250016279e-10` | `3.4938679727147814e-10` | no | no |
| generated-stock | composed flowpipe | `Continuous.cpp:386-390` | `[0.010100670333333329, 0.1121208036666667]` | exact endpoint corners | `3.199554250016279e-10` | `3.4938679727147814e-10` | no | no |
| endpoint observer | `endpoint_raw` | observer `generated_stock_trace.cpp:413-423`; `TaylorModel.h:2987-2996` | `[0.010100670333333329, 0.1121208036666667]` | exact interval endpoint | `3.199554250016279e-10` | `3.4938679727147814e-10` | no | no |

The initial `Flowpipe` box is constructed at `Continuous.cpp:260-269` and the
normalized initial set remains `[0,0.1]` with zero stored remainder through
`Continuous.cpp:864-952`. The no-remainder Picard loop at
`Continuous.cpp:954-956` calls `TaylorModel.h:3681-3703`. Its order-4 polynomial
contains the `tau^4` constant coefficient `0.3666666666666667` but not the
`tau^4*xi` initial-dependency term. For the centered initial generator `0.05*xi`,
that missing coefficient is
`0.05 * 2^4 / 4! = 0.03333333333333333`; at `h=0.01` its magnitude is about
`3.333e-10`. This term has total degree five and is excluded from the order-four
polynomial, so the stored remainder must cover it.

The candidate `[-1e-4,1e-4]`, combined cutoff/truncation remainder (about
`[-2.001333e-6,2.001407e-6]`), and accepted refinements 0 and 1 all cover the
analytic corners. Refinement 2 replaces the stored remainder by approximately
`[-1.601067e-11,1.601125e-11]`. The stock acceptance loop tests only whether this
new remainder is a subset of the previous remainder and its width ratio; at that
point the analytic corners are no longer enclosed. Later refinement, composition,
time substitution (`TaylorModel.h:3433-3443`), interval evaluation, `%.17g`
formatting, and parsing preserve rather than create the loss.

Cutoff and truncation are returned together by stock
`Picard_ctrunc_normal`; the public stock object does not expose them as two
separate additive fields. The observer records that combined remainder, the
polynomial coefficients, and the separately computed polynomial roundoff
difference instead of inventing a decomposition.

## Official/generated parity

| claim | result |
|---|---|
| model/constants | exact parity: `1 + 2*x` |
| initial-set representation | exact parity: `Flowpipe([0,0.1])` |
| order/cutoff/candidate | exact parity: 4, `1e-15`, `[-1e-4,1e-4]` |
| preconditioning/symbolic remainder | diagonal scaling / disabled on both |
| accepted schedule | not equal: generated `0.01`; official `0.009999999999` |
| raw polynomial/stored remainder/refinement fields | unavailable through the official public API |
| endpoint/segment/tube | available on both, but compared at their own accepted times |

The official source uses only public APIs and calls `ODE<Real>::reach`. The fixed
route in `Continuous.h:590-665` invokes the same
`currentFlowpipe.advance(... expressions ...)` function localized above.
Its loop starts at `THRESHOLD_HIGH`, so a one-segment horizon is shortened by
`1e-12`; this explains schedule non-parity and is not used to excuse the failed
range. The official route completes one segment and returns safe, but its endpoint
and tube still under-enclose their accepted-time oracle. It cannot expose the
internal refinement row, while its shared core call and independent failure
confirm that the generated observer did not create the numerical behavior.

## Bounded attribution matrix

These rows are diagnostics only; none replaces or repairs the primary contract.

| variation | value | max strict defect | result |
|---|---:|---:|---|
| step | `h=0.01` | `3.4938679727147814e-10` | failed |
| step | `h=0.005` | `2.1334142785711663e-11` | failed |
| step | `h=0.0025` | `1.3177098301397905e-12` | failed |
| order | 4 | `3.4938679727147814e-10` | failed |
| order | 5 | `1.3867934578470908e-12` | failed |
| order | 6 | `4.579669976578771e-15` | failed |
| input corner | `[0,0]` | `1.3377902952083076e-11` | failed upper |
| input corner | `[0.1,0.1]` | `1.605346411359676e-11` | failed upper |
| input interval | `[0,0.1]` | `3.4938679727147814e-10` | failed both |

The observed step-defect slopes are `4.0336` and `4.0171`, consistent with the
order-four truncation/dependency attribution. Higher order and smaller step reduce
the defect but do not make the frozen primary result sound.

## Diagnosis and repository changes

Outcomes A–E are rejected by the evidence: the MPFR oracle and monotonicity are
independent; raw stdout and parsed traces are exact; the accepted native object is
already unsound before composition/export; official and generated configuration
fields match; and the intended local endpoint domain is used. Outcome F is
selected because both official-stock and generated-stock enter the same unmodified
`Flowpipe::advance` core and under-enclose, with the generated path locating the
first loss inside its accepted remainder refinement.

No external or repository numerical core was changed. Added code consists only
of the stock observer/minimal reproducer, official public-API reproducer, directed
MPFR oracle, strict analysis/runner, and regression tests. No Flow* patch,
tolerance, epsilon, candidate widening, repaired hull, adapter, parameter retuning,
or result-data edit is present.

The Flow* correctness gate therefore remains **OPEN**, and every cross-tool
primary-comparison eligibility flag remains false. The exact next separately
authorized task is the Xiangru complete-Q3 interval-soundness audit. TORA B48,
speedup/ranking work, and Torch tensorization remain unauthorized; TORA can be
considered only after both the Flow* and Q3 soundness gates close.

## Evidence

Portable committed evidence:

- `outputs/flowstar_scalar_affine_correctness_closure/20260804T131445Z/`;
- `first_containment_loss.json` for the complete row-by-row trace;
- `official_generated_parity.json` for separate configuration, schedule, range,
  and field parity;
- `analytic_oracle.json` and raw MPFR stdout;
- `step_order_corner_matrix.csv` for the seven unique required diagnostic rows;
- `backend_identity.json`, `commands.jsonl`, manifest, and SHA256 sums.

Server-local preserved failed runner attempt:
`/srv/local/shengenli/flowstar_scalar_affine_correctness_closure_failed_20260804T123840Z`.
It failed only in evidence path bookkeeping after two successful primary runs and
is excluded from committed scientific evidence; it was not overwritten or
deleted.
