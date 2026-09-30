# Remote native QUAD read-only recheck — 2026-10-01

This is a read-only recheck of the original native QUAD job. No solver,
watchdog, replacement run, or GPU experiment was started.

Audit timestamp: `2026-10-01T02:28:17+08:00`. The attached structured receipt
is `docs/evidence/remote_native_quad_recheck_20261001.json`.

## Identity and terminal state

- Remote research root:
  `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`
- Original run:
  `runs/native_quad_matched_20260929/initial_affine_cover_variant/full1000_v1`
- Watch directory: the sibling `full1000_v1_watch`
- `RESULT.json`: `status=timeout`, solver timeout `21600 s`, supervisor elapsed
  `21609.3083 s`.
- Watch receipt: elapsed `21609.612654 s`; solver/server return code `-15`.
- Recorded watchdog/supervisor/server/solver PIDs `1359708`, `1359709`,
  `1359714`, and `1359789` were absent at the recheck.
- There are 30 complete control periods, numbered 0 through 29. Every period
  reports all 1024 lanes complete, 20 accepted steps, status 2, and
  `completed=true`. The complete common prefix is therefore 600 integration
  steps, not the requested 1000-step/T=5 result.
- Final symbolic-remainder size is 600 with capacity 1000; total accepted lane
  steps are 614400 (`600 * 1024`).
- No complete T=5 result or full-horizon endpoint width exists for this run.

## Byte identity against the published handoff evidence

The live remote files matched the already-published local evidence:

| File | SHA-256 |
|---|---|
| `INPUT.json` | `c2823266ae9a1c7574f9754d2e03a344fbc5a97f3c8c040d9989d27f69ab4ef4` |
| `RESULT.json` | `036943cd0b2a0e040017e30c0ac82d0b41f6eb645d924d2294dfe70e2bde97f2` |
| run `processes.json` | `ab023941166c46d6841ea04192bcbf8ddb3e27bd6ea84c98a13dcaa124759fff` |
| `periods.jsonl` | `4a0c4d4a78a0144afa535d46465908c3338ebe38e4ad78e645a3e2326a6d0a26` |
| watch `process.json` | `387dfdd8a801f7de55b4908ad548db5dc4983cbe2b39a6bf0c71918f4376071b` |

## Duplicate-run and resource check

- Sibling run directories were only the known 6-hour snapshot, original run,
  watch directory, short40 run, and short40 watch directory.
- No 24-hour replacement directory was present.
- At audit time GPU 0 was occupied by another user's VLLM process (about
  13.3 GiB); GPUs 1–3 were empty. No process or allocation was changed.
- All 47 paths listed by the frozen
  `research/gpu_verified_20260930/EXTERNAL_ASSETS.json` were still present on
  the server (0 missing). This is an existence check, not a fresh recursive
  content-hash or ABI qualification.

The PID-absence, sibling-directory, and GPU facts are transient observations
from the live read-only audit. They are not derived from, or guaranteed by,
the five file hashes above. The receipt records that limitation and the exact
check categories; it is not a durable resource-availability promise.

## Operational boundary

The experiment pause remains in force. Any future native full-horizon retry
must first repeat the read-only identity/process check and then use a new run
directory; it must not overwrite or restart `full1000_v1`.
