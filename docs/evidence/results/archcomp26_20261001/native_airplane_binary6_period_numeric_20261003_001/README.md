# Airplane continuous: high-corner first held-control period diagnostic

**Observed: numerical refusal at substep 5.** This separate run covers only
binary-cover cell `111111`. Its first four `h=0.01 s` substeps were accepted,
giving a saved numerical prefix through `t=0.04 s`; the fifth substep failed
the native Picard self-inclusion check. The requested first `0.1 s` held-control
period and the benchmark's `T=2 s` horizon were not completed. The remote run
identity is
`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_airplane_binary6_period_numeric_20261003_001`.
No prior run directory is changed.

The original [RESULT](cells/111111/RESULT.json) is `failed/exit 2` after
3.9264 s wall, without timeout; the native result is
`4 = UNCOMPLETED_SAFE` under an **empty solver safety set**. That native status
describes numerical noncompletion and is not a safety verdict. The
[driver audit](cells/111111/AUDIT.json) and separately implemented
[independent saved-evidence audit](INDEPENDENT_AUDIT.json) both identify four
complete 408-byte range records, one controller RPC, and the first refusal at
step 5. The new initial box, controller response, and first range record match
the prior high-corner first-step receipt directly. The first four Picard groups
passed; at the fifth, coordinates 0 and 1 require approximately
`[-0.01138866,0.01142780]` and `[-0.01119116,0.01125378]` respectively,
outside their frozen `[-0.01,0.01]` estimates. The queue stopped at this first
numerical refusal; no other binary-cover cell was launched for this period
diagnostic.

Each saved tube has property status **Unknown**, not a demonstrated unsafe
trajectory. All saved tubes retain positive `cos(theta)`; the smallest saved
lower bound is `0.5350594`. These observations are confined to the four-step
prefix. They do not prove safety, failure of a physical trajectory,
independent NN/CROWN soundness, or Flow* floating-point enclosure.

The [isolated driver](../../../../../tools/archcomp26_airplane_binary6_period_numeric_20261003_001.py)
copies the saved six-bit C++ entry and applies only these numerical changes:
one `0.1 s` call with fixed `h=0.01 s` (ten substeps), an empty safety set
inside the reach call so property Unknown cannot end integration, and a
completion gate of ten saved segments. The 12→6 ONNX call occurs once before
the reach call, so its output is held over the period. The original safety
box is checked from each accepted tube after the reach call; a tube crossing
the boundary is labelled `unknown`, not unsafe. Positive `cos(theta)` is
checked separately because the ODE divides by `cos(theta)`; if the saved tube
cannot establish positivity, even ten solver segments are **not** counted as
conditional ODE coverage.

The target was **only `111111`**, with a 180-second process limit. The
driver writes `START.json`, `RESULT.json`, the original RPC/native/range/safety
files, the frozen generated C++ source, `BUILD.json`, and `AUDIT.json`. Before
interpreting later segments, its audit directly compares the new initial box,
one RPC value object and first range record with the prior high-corner saved
receipt. A mismatch stops the gate. Completion requires ten valid 408-byte
range records, ten matching tube rows, one RPC, and a completed native
numerical status. A normal native numerical refusal saves the accepted prefix
and records its first refused step. With zero saved segments, property and
denominator-domain fields are null and the old first-segment comparison is
unavailable; any returned numerical refusal is marked *uncompared*. A thrown
exception before the reach call returns cannot persist the in-memory prefix;
missing ranges, timeout, and malformed receipts are labelled
infrastructure/unknown, not zero-step Picard refusal.

The prior first-step files contain coordinate ranges but no restartable
Flowpipe/Taylor-model and symbolic-remainder state. The new ten-step run must
therefore recompute its own first substep; it does not rerun or overwrite the
old first-step job. Even a complete `0.1 s` result would be conditional on the
saved controller envelope and would not establish `T=2 s` completion,
full-time safety, or independent NN/CROWN soundness.

The local and server no-solver preflights passed before the single launch:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tools/archcomp26_airplane_binary6_period_numeric_20261003_001.py offline docs/evidence/results/archcomp26_20261001
```

The isolated source change and old corner receipt were valid. The saved
[C++ source](build/archcomp/airplane/airplane_binary6_period_numeric.cpp),
[RPC](cells/111111/controller_rpc.jsonl), [native log](cells/111111/native.log),
[ranges](cells/111111/ranges.bin), and [per-step safety table](cells/111111/safety.tsv)
are original run artifacts. Re-run the independent audit offline with
`PYTHONDONTWRITEBYTECODE=1 python3 -B
docs/evidence/results/archcomp26_20261001/native_airplane_binary6_period_numeric_20261003_001/audit_saved_independent.py`.
