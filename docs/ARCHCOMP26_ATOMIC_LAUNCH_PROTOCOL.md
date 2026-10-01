# ARCH-COMP 2026 atomic launch protocol

Status: design contract only. The production launcher remains disabled,
`wrapper_sha256` remains null, and this document does not authorize an
experiment.

## Safety claim and boundary

The launcher may promise automatic at-most-once workload authorization for one
`invocation_id`. It cannot promise arbitrary-payload exactly-once execution
across process and host crashes. Recovery never recreates a spawn or GO
capability from disk. An ambiguous prepared or possibly-started attempt is
retained and failed closed for adoption, terminal projection, or explicit human
resolution.

The journal phase is launch authority, an independently durable phase anchor is
the rollback detector, the child supervisor's terminal marker is the execution
commit, and ledger/result/matrix files are rebuildable projections. UTC is for
audit display and freshness expiry. Ordering, elapsed time, and process identity
must use journal sequence plus Linux boot/process clocks.

The current filesystem design is crash-consistent for the trusted launcher
protocol. It is not WORM storage against a malicious process running as the same
UID. Every path used as launch authority must nevertheless be securely opened
without symlink traversal and verified against a frozen device/inode/owner/mode
identity.

## Required authority chain

The frozen campaign configuration must bind distinct pre-provisioned `0700`
journal and state directories. Directory link counts are not frozen. The state
directory contains one identity-derived attempt directory per invocation; its
name is `sha256(invocation_id)`, never raw user text. The matrix parent identity
must also be frozen if the matrix is outside the state directory.

The next schema revision must add these identities to `launch_guard`, extend
the hold scope through projection-parent fsync, and include them in
`campaign_configuration_sha256`. A persisted journal anchor has exactly:

```text
sequence, filename, sha256, event_id, event, invocation_id
```

`filename` must be the canonical name implied by `sequence`. The in-memory
`created` flag is never serialized.

An `attempt_prepared` payload must bind the cell and slot, attempt index, cell
plan SHA, fresh-audit link, immutable lock receipt, identity-bound attempt
directory, planned argv/cwd hashes, blocked-supervisor protocol, and handshake
nonce hash. Empty or event-generic payloads cannot authorize spawn.

The immutable active receipt represents the prepared phase and therefore has no
PID. It binds the campaign/configuration, host, cell/slot/attempt/invocation,
prepared timestamp, plan and command hashes, audit and lock evidence, journal
and attempt-directory identities, prepared journal anchor, spawn protocol, and
nonce hash. Publishing this receipt and committing the matrix to `running` must
finish before a spawn permit exists.

The immutable process identity must bind at least:

```text
host_boot_id, pid, proc_start_ticks,
exe_path, exe_device, exe_inode,
cmdline_nul_sha256,
cwd_path, cwd_device, cwd_inode,
planned_argv_sha256, planned_cwd_sha256,
spawned_at_utc, spawned_boottime_ns,
session_id, process_group_id, handshake_nonce_sha256
```

PID reuse is rejected by the `(host_boot_id, pid, proc_start_ticks)` tuple.
Executable, exact NUL-separated cmdline bytes, and cwd are independently
checked. A pidfd may be held at runtime but is not serialized. No signal may be
sent until identity is re-read after `pidfd_open` and all fields still match.

The attempt directory uses fixed no-clobber `spawn.json` and `terminal.json`
commit markers so recovery can find them from the active receipt without an
untrusted directory scan. The terminal marker binds the prepared/spawn anchors,
process identity when one exists, execution phase, outcome/exit status,
UTC/boottime, elapsed time when available, raw artifact hashes, and failure
detail. Valid execution phases include `never_spawned`, `spawned_blocked`,
`work_started`, and `payload_exited`.

The result and attempt-ledger revisions must bind prepared, spawned, and
terminal anchors and their immutable receipt pointers. A never-spawned attempt
has no process identity or timing sample, is never included in timing, and is
still retained in the ledger. No positive PID or elapsed time may be fabricated
to satisfy an older schema.

## Normal commit order

All steps occur while the identity-bound non-blocking campaign flock is held:

1. Recover and validate the journal against a minimum head obtained from
   separately durable active/result evidence, never from the journal itself.
2. Perform a fresh duplicate-process/original-QUAD audit. Fsync the audit file
   and parent directory, then publish an immutable lock receipt that binds the
   campaign configuration.
3. Append and fsync `attempt_prepared`.
4. Publish the active prepared receipt with `O_EXCL`, canonical bytes, `0400`,
   file fsync, no-clobber hard link, directory fsync, pending unlink, and a
   second directory fsync.
5. Commit one matrix replacement, guarded by the expected old matrix SHA, that
   sets the selected cell to `running` and installs the active pointer. Fsync
   the file and parent, reopen securely, and verify exact bytes.
6. Only the conjunction of a newly created prepared event, newly durable active
   receipt, verified matrix commit, and the still-held lock creates an
   in-memory, non-serializable, one-shot `SpawnPermit`.
7. Consume the permit before `Popen`. Start a stable supervisor blocked on a GO
   pipe. Before GO it must have no workload side effect. READY returns over a
   separate pipe.
8. Read strong `/proc` identity, publish it, append `process_spawned`, and
   durably publish the spawn anchor. Recheck audit freshness and process
   identity. Only then write GO exactly once and close the capability FD.
9. The supervisor writes a durable `work_started` marker before the workload
   side effect. It durably writes raw artifacts and `terminal.json`, including
   parent-directory fsync, even if the original parent launcher disappears.
10. Validate terminal evidence, append `attempt_terminal`, and durably publish
    the terminal anchor.
11. Fsync immutable ledger/result dependencies. One final matrix replacement
    atomically records terminal status and result pointer and clears active;
    fsync its parent and reopen/verify. Only then release the flock.

The public surface should expose one normal `run_one_attempt` path and a
separate recovery path that has no spawner or GO capability. Lower-level
writers remain internal.

## Crash recovery rules

| Last durable state | Recovery action |
|---|---|
| Prepared journal only | No spawn; retain/fail closed or project `never_spawned` after explicit policy. |
| Active prepared receipt or running matrix | Same invocation is never passed to `Popen` again, even if the crash appears pre-spawn. |
| Child exists but spawn anchor is absent | Verify strong identity and wait for its blocked/terminal protocol; never resend GO. |
| GO delivery is uncertain | Mark possibly started; adopt a matching live process, or finalize a valid terminal; absent without terminal fails closed. |
| Terminal marker exists but terminal journal/anchor is absent | Idempotently append the deterministic terminal event and publish its anchor. |
| Terminal anchor exists but projection is incomplete | Rebuild ledger/result/matrix only; no child action. |
| Matrix rename/fsync outcome is uncertain | Securely reopen and compare canonical projection; redo only the same projection if needed. |

If the parent dies before GO, pipe EOF makes the supervisor write
`spawned_blocked`/aborted terminal evidence and exit without executing the
workload. After GO, a live matching process may be monitored or adopted. A dead
process without valid terminal evidence is `incomplete_unknown`, not permission
to rerun.

## Mandatory fake-child qualification

Before binding `wrapper_sha256`, tests must inject hard process exits after
every file/parent fsync and every state transition: prepared append, active
publish, matrix rename/fsync, `Popen`, READY, identity publish, spawned append
and anchor, GO, work marker, raw output fsync, terminal publish, terminal append
and anchor, ledger/result publish, and final matrix fsync. Recovery must be run
twice and prove, per invocation, `Popen <= 1`, workload side effect `<= 1`, and
recovery spawner/GO calls `== 0`.

Additional required adversarial tests cover PID reuse, boot/start-tick mismatch,
exe/cmdline/cwd mutation, `/proc/stat` names containing spaces and `)`, exact
NUL cmdline hashing including empty arguments, audit expiry before GO, terminal
symlink/FIFO/directory/hardlink/path-swap cases, journal anchor deletion or
rewrite, matrix compare-and-swap conflicts, and a contender remaining locked out
until final projection-parent fsync.

## Enablement gate

The launcher remains unavailable until all of the following are implemented
and independently reviewed:

- event-specific journal payload validation;
- frozen journal/state/matrix-parent identities;
- immutable prepared, spawn, process-identity, and terminal publishers;
- blocked-supervisor/READY/GO protocol and strong Linux identity recovery;
- ledger/result schemas that retain never-spawned attempts and terminal anchors;
- matrix compare-and-swap with durable parent fsync;
- full crash matrix and real two-process exclusion tests;
- production wrapper source SHA bound into the campaign configuration.

Until then the CLI must continue returning a nonzero disabled status and the
matrix must keep `wrapper_sha256=null`.
