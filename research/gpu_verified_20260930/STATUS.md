# Current status — 2026-09-30

The user requested publication of the latest clean branch and a Markdown goal for the next conversation. This session organized existing source/evidence and made read-only server checks. It did not start a benchmark, modify numerical algorithms, or stop any server process.

## New native QUAD terminal evidence

The previous handoff's last observation, 400/1000 steps at 2026-09-29 20:19:53 China time, was a snapshot, not a final result. Read-only collection on 2026-09-30 retrieved:

- `RESULT.json`: `status=timeout`; native limit 21600 seconds; supervisor elapsed 21609.30825667875 seconds; solver and NN server return codes -15 after the supervisor's timeout cleanup.
- `watch_process.json`: process wall 21609.61265403917 seconds, return code 1; peak sampled process-tree RSS 40395374592 bytes and owned GPU 1180696576 bytes. The watchdog itself did not report a resource-guard stop.
- `periods.jsonl`: 30 records, periods 0–29; each contains 1024 lanes with `completed=true` and `accepted_this_period=20`. This establishes 600 steps in complete-period records. Any partial next period is not promoted to a completed period.

These are failure/completion records, not full-horizon performance or terminal-width data. The large `ranges.bin` was not downloaded or independently re-audited in this packaging session. No new 24-hour run was launched. The prior request to stop an active six-hour job is now historical for this naturally timed-out run; a future session should still check the actual run identity before any new launch.

The original files are copied under `evidence/native_terminal_20260930/`, with server paths and hashes in `evidence/REMOTE_ASSETS_MANIFEST.json`.

## Other results remain unchanged

- Latest strict P3 + trig: complete 1024×1000, 50 NN; watch1533.752052 seconds; complete output equivalence to previous P3 established by the archived actual CPU audit.
- Huan parity + B-only SR chunk: five complete runs; median75.250099 seconds. No new Huan run here.
- Corrected full-batch P2: common accepted prefix799, first rejection800. No rerun here.
- Airplane: existing sparse-metadata and property-checking work; no full200 or new four-way timing.
- Unicycle: existing native first-step states; GPU comparison still blocked before numerical execution by the loader environment. No loader fix or new arm executed here.
- Latest P3 is not yet qualified across all ARCH benchmarks; full end-to-end NNCS certification remains incomplete.

The English report and long handoff are preserved as dated evidence snapshots. Their old “native terminal unknown/400” statements must be read together with this update. Future reports should use this terminal classification rather than silently replacing the historical files.
