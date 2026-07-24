# Validation record

The packaged smoke runner passed all gates before the full sweep:

- 8 DiffReach projection/parity tests passed;
- 5 Torch basis/reset tests passed;
- Flow* raw/transformed extraction agreed;
- every smoke adapter contained the analytic references;
- deterministic smoke trajectories were contained; and
- all eight report plots generated.

The numerical full sweep was launched in tmux session
`tm_first_order_followup`.  It completed all Torch, DiffReach, Flow*, common
collection, analytic-reference, and deterministic-trajectory gates.  The
session later terminated during a duplicate repository-wide pytest invocation,
after 51% progress and without an assertion traceback.  The exact test phase
was rerun independently to completion:

```text
275 passed, 3 skipped in 162.70s
8 DiffReach follow-up tests passed
```

The Flow* accepted-prefix metadata was then corrected without changing the
numerical enclosures.  Common collection, frozen-baseline hashing, plots, and
the report were regenerated.  Final gates:

```text
analytic endpoint checks: 24,559 passed, 0 violations
deterministic trajectory checks: 130,104 passed, 0 violations
focused Flow* audit: 5 exact + 500 sampled checks, 0 violations
frozen baseline: 1,999 files byte-for-byte unchanged during the run
```

`raw_results.csv` is the non-duplicated common table.  The full per-segment
Torch diagnostic JSON is committed as `torch_diagnostics.json.gz`; its
uncompressed local form and generated C++ executables are deliberately omitted
from the curated Git artifact.
