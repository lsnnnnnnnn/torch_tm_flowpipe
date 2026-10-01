"""Atomic lock and journal primitives for the future ARCH-COMP26 launcher.

This module intentionally does not spawn experiments yet.  The production
matrix keeps ``wrapper_sha256=null`` until fresh-audit, sample-journal, process,
and terminal-fsync handling are wired around this lock.

The lock file must be provisioned before the campaign and its device/inode
identity bound by the future campaign receipt.  Acquisition never creates,
unlinks, or replaces it.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Iterator, Mapping


JOURNAL_ENTRY_SCHEMA = "archcomp26-campaign-journal-entry-v1"
JOURNAL_EVENT_TYPES = {
    "attempt_prepared", "process_spawned", "attempt_terminal",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_UTC_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?Z\Z"
)
_FINAL_ENTRY_RE = re.compile(
    r"(?P<sequence>[0-9]{20})\.json\Z"
)
_PENDING_ENTRY_RE = re.compile(
    r"\.pending-(?P<sequence>[0-9]{20})-(?P<sha256>[0-9a-f]{64})-"
    r"(?P<token>[0-9a-f]{32})\.json\Z"
)
_MAX_JOURNAL_ENTRY_BYTES = 1024 * 1024


class CampaignLockError(RuntimeError):
    """The fixed campaign lock is unsafe or cannot be used."""


class CampaignLockBusy(CampaignLockError):
    """Another launcher currently owns the campaign lock."""


class CampaignJournalError(RuntimeError):
    """The append-only campaign journal is unsafe or inconsistent."""


@dataclass(frozen=True)
class CampaignJournalReceipt:
    """Identity of one durable journal event."""

    sequence: int
    filename: str
    sha256: str
    event_id: str
    event: str
    invocation_id: str
    created: bool


def _check_directory(descriptor: int, owner: int, display_path: Path) -> None:
    directory_stat = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(directory_stat.st_mode)
        or directory_stat.st_uid not in {0, owner}
        or stat.S_IMODE(directory_stat.st_mode) & 0o022
    ):
        raise CampaignLockError(
            f"campaign lock directory has unsafe identity or mode: {display_path}"
        )


def _open_verified_directory(directory: Path, owner: int) -> int:
    """Open an absolute directory without following any path-component symlink."""
    if not directory.is_absolute() or any(
        part in {os.curdir, os.pardir, ""} for part in directory.parts[1:]
    ):
        raise CampaignLockError("campaign lock directory must be a normalized absolute path")
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    except AttributeError as error:
        raise CampaignLockError("required secure POSIX open flags are unavailable") from error
    try:
        descriptor = os.open(os.sep, flags)
    except OSError as error:
        raise CampaignLockError("cannot open filesystem root for campaign lock") from error
    walked = Path(os.sep)
    try:
        _check_directory(descriptor, owner, walked)
        for part in directory.parts[1:]:
            candidate = walked / part
            try:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
            except (OSError, TypeError, NotImplementedError) as error:
                raise CampaignLockError(
                    f"cannot securely open campaign lock directory: {candidate}"
                ) from error
            try:
                _check_directory(next_descriptor, owner, candidate)
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
            walked = candidate
        if os.fstat(descriptor).st_uid != owner:
            raise CampaignLockError("campaign lock parent is not owned by expected uid")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _check_lock_file(file_stat: os.stat_result, owner: int) -> None:
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_nlink != 1
        or file_stat.st_uid != owner
        or stat.S_IMODE(file_stat.st_mode) != 0o600
    ):
        raise CampaignLockError("campaign lock file has unsafe identity or mode")


@contextmanager
def exclusive_campaign_lock(
    lock_path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_uid: int | None = None,
) -> Iterator[None]:
    """Hold the pre-provisioned, identity-bound lock without path traversal."""
    path = Path(lock_path)
    owner = os.geteuid() if expected_uid is None else expected_uid
    if (
        isinstance(owner, bool)
        or not isinstance(owner, int)
        or owner < 0
        or not isinstance(expected_identity, tuple)
        or len(expected_identity) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in expected_identity
        )
        or path.name in {"", os.curdir, os.pardir}
    ):
        raise CampaignLockError("campaign lock identity is invalid")
    parent_descriptor = _open_verified_directory(path.parent, owner)
    try:
        try:
            flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        except AttributeError as error:
            raise CampaignLockError("required secure POSIX open flags are unavailable") \
                from error
        try:
            descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise CampaignLockError(f"cannot open campaign lock: {path}") from error
        locked = False
        try:
            file_stat = os.fstat(descriptor)
            _check_lock_file(file_stat, owner)
            if (file_stat.st_dev, file_stat.st_ino) != expected_identity:
                raise CampaignLockError("campaign lock does not match frozen identity")
            if os.get_inheritable(descriptor):
                raise CampaignLockError("campaign lock descriptor is inheritable")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise CampaignLockBusy("campaign lock is already held") from error
                raise CampaignLockError("cannot acquire campaign lock") from error
            try:
                path_stat = os.stat(
                    path.name, dir_fd=parent_descriptor, follow_symlinks=False
                )
            except OSError as error:
                raise CampaignLockError(
                    "campaign lock disappeared after acquisition"
                ) from error
            _check_lock_file(path_stat, owner)
            if (path_stat.st_dev, path_stat.st_ino) != expected_identity:
                raise CampaignLockError("campaign lock path changed during acquisition")
            yield
        finally:
            path_error = None
            if locked:
                try:
                    final_stat = os.stat(
                        path.name, dir_fd=parent_descriptor, follow_symlinks=False
                    )
                    _check_lock_file(final_stat, owner)
                    if (final_stat.st_dev, final_stat.st_ino) != expected_identity:
                        raise CampaignLockError("campaign lock path changed while held")
                except (OSError, CampaignLockError) as error:
                    path_error = CampaignLockError(
                        "campaign lock path changed while held"
                    )
                    path_error.__cause__ = error
            unlock_error = None
            try:
                if locked:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError as error:
                unlock_error = CampaignLockError("cannot release campaign lock")
                unlock_error.__cause__ = error
            finally:
                os.close(descriptor)
            if path_error is not None:
                if unlock_error is not None:
                    raise path_error from unlock_error
                raise path_error
            if unlock_error is not None:
                raise unlock_error
    finally:
        os.close(parent_descriptor)


def _journal_owner(expected_uid: int | None) -> int:
    owner = os.geteuid() if expected_uid is None else expected_uid
    if isinstance(owner, bool) or not isinstance(owner, int) or owner < 0:
        raise CampaignJournalError("campaign journal owner is invalid")
    return owner


def _journal_identity(expected_identity: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(expected_identity, tuple)
        or len(expected_identity) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in expected_identity
        )
    ):
        raise CampaignJournalError("campaign journal identity is invalid")
    return expected_identity


def _open_journal_directory(
    journal_dir: Path,
    owner: int,
    expected_identity: tuple[int, int],
) -> int:
    try:
        descriptor = _open_verified_directory(Path(journal_dir), owner)
    except CampaignLockError as error:
        raise CampaignJournalError(str(error)) from error
    directory_stat = os.fstat(descriptor)
    if (
        (directory_stat.st_dev, directory_stat.st_ino) != expected_identity
        or directory_stat.st_uid != owner
        or stat.S_IMODE(directory_stat.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise CampaignJournalError(
            "campaign journal directory does not match frozen 0700 identity"
        )
    return descriptor


def _recheck_journal_path(
    journal_dir: Path,
    owner: int,
    expected_identity: tuple[int, int],
) -> None:
    descriptor = _open_journal_directory(journal_dir, owner, expected_identity)
    os.close(descriptor)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _canonical_journal_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise CampaignJournalError(
            f"campaign journal event is not canonical JSON: {error}"
        ) from error
    if len(encoded) > _MAX_JOURNAL_ENTRY_BYTES:
        raise CampaignJournalError("campaign journal entry exceeds size limit")
    return encoded


def _parse_journal_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or _UTC_RE.fullmatch(value) is None:
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None


def _validate_journal_entry(
    entry: Any,
    *,
    expected_sequence: int,
    campaign_configuration_sha256: str,
) -> datetime:
    fields = {
        "schema_version", "sequence", "previous_entry_sha256", "event_id",
        "event", "campaign_configuration_sha256", "invocation_id",
        "recorded_at_utc", "payload",
    }
    if not isinstance(entry, dict) or set(entry) != fields:
        raise CampaignJournalError("campaign journal entry has the wrong shape")
    if entry["schema_version"] != JOURNAL_ENTRY_SCHEMA:
        raise CampaignJournalError("campaign journal entry has the wrong schema")
    if type(entry["sequence"]) is not int \
            or entry["sequence"] != expected_sequence:
        raise CampaignJournalError("campaign journal sequence is not contiguous")
    previous = entry["previous_entry_sha256"]
    if previous is not None and (
        not isinstance(previous, str) or _SHA256_RE.fullmatch(previous) is None
    ):
        raise CampaignJournalError("campaign journal predecessor hash is invalid")
    for name in ("event_id", "invocation_id"):
        value = entry[name]
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise CampaignJournalError(f"campaign journal {name} is invalid")
    if not isinstance(entry["event"], str) \
            or entry["event"] not in JOURNAL_EVENT_TYPES:
        raise CampaignJournalError("campaign journal event type is invalid")
    if entry["campaign_configuration_sha256"] \
            != campaign_configuration_sha256:
        raise CampaignJournalError("campaign journal configuration identity mismatch")
    if not isinstance(entry["payload"], dict):
        raise CampaignJournalError("campaign journal payload must be an object")
    recorded = _parse_journal_time(entry["recorded_at_utc"])
    if recorded is None:
        raise CampaignJournalError("campaign journal timestamp is invalid")
    return recorded


def _check_journal_file(
    file_stat: os.stat_result,
    owner: int,
    allowed_nlinks: set[int],
) -> None:
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_uid != owner
        or stat.S_IMODE(file_stat.st_mode) != 0o400
        or file_stat.st_nlink not in allowed_nlinks
    ):
        raise CampaignJournalError(
            "campaign journal entry has unsafe identity, mode, or link count"
        )


def _read_journal_file(
    directory_descriptor: int,
    name: str,
    *,
    owner: int,
    allowed_nlinks: set[int],
    expected_sequence: int,
    expected_sha256: str | None,
    campaign_configuration_sha256: str,
) -> tuple[dict[str, Any], bytes, os.stat_result, str]:
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    except AttributeError as error:
        raise CampaignJournalError(
            "required secure POSIX open flags are unavailable"
        ) from error
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except (OSError, TypeError, NotImplementedError) as error:
        raise CampaignJournalError(
            f"cannot securely open campaign journal entry {name!r}"
        ) from error
    try:
        before = os.fstat(descriptor)
        _check_journal_file(before, owner, allowed_nlinks)
        if before.st_size <= 0 or before.st_size > _MAX_JOURNAL_ENTRY_BYTES:
            raise CampaignJournalError("campaign journal entry size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65536))
            if not chunk:
                raise CampaignJournalError("campaign journal entry is truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CampaignJournalError("campaign journal entry grew during read")
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev", "st_ino", "st_uid", "st_mode", "st_nlink",
            "st_size", "st_mtime_ns",
        )
        if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
            raise CampaignJournalError("campaign journal entry changed during read")
        _check_journal_file(after, owner, allowed_nlinks)
        try:
            path_stat = os.stat(
                name, dir_fd=directory_descriptor, follow_symlinks=False
            )
        except (OSError, TypeError, NotImplementedError) as error:
            raise CampaignJournalError(
                "campaign journal entry path changed during read"
            ) from error
        if (path_stat.st_dev, path_stat.st_ino) != (after.st_dev, after.st_ino):
            raise CampaignJournalError(
                "campaign journal entry path changed during read"
            )
        raw = b"".join(chunks)
        try:
            entry = json.loads(
                raw.decode("utf-8"), parse_constant=_reject_nonfinite_json
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            raise CampaignJournalError(
                f"campaign journal entry is not valid JSON: {error}"
            ) from error
        if not isinstance(entry, dict) or _canonical_journal_bytes(entry) != raw:
            raise CampaignJournalError("campaign journal entry is not canonical JSON")
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise CampaignJournalError("campaign journal filename digest mismatch")
        _validate_journal_entry(
            entry,
            expected_sequence=expected_sequence,
            campaign_configuration_sha256=campaign_configuration_sha256,
        )
        return entry, raw, after, digest
    finally:
        os.close(descriptor)


def _journal_names(directory_descriptor: int) -> list[str]:
    try:
        names = os.listdir(directory_descriptor)
    except (OSError, TypeError, NotImplementedError) as error:
        raise CampaignJournalError("cannot list campaign journal directory") from error
    if not all(isinstance(name, str) for name in names):
        raise CampaignJournalError("campaign journal contains an invalid filename")
    return sorted(names)


def _classified_journal_names(
    directory_descriptor: int,
    *,
    names: list[str] | None = None,
) -> tuple[dict[int, str], list[tuple[str, int, str]]]:
    finals: dict[int, str] = {}
    pending: list[tuple[str, int, str]] = []
    snapshot = _journal_names(directory_descriptor) if names is None else names
    for name in snapshot:
        final_match = _FINAL_ENTRY_RE.fullmatch(name)
        if final_match is not None:
            sequence = int(final_match.group("sequence"))
            if sequence in finals:
                raise CampaignJournalError(
                    "campaign journal contains a duplicate sequence"
                )
            finals[sequence] = name
            continue
        pending_match = _PENDING_ENTRY_RE.fullmatch(name)
        if pending_match is not None:
            pending.append((
                name,
                int(pending_match.group("sequence")),
                pending_match.group("sha256"),
            ))
            continue
        raise CampaignJournalError(
            f"campaign journal contains unknown file {name!r}"
        )
    if len(pending) > 1:
        raise CampaignJournalError(
            "campaign journal contains multiple pending publications"
        )
    return finals, pending


def _check_discardable_pending(
    directory_descriptor: int,
    pending_name: str,
    *,
    owner: int,
) -> None:
    try:
        pending_stat = os.stat(
            pending_name, dir_fd=directory_descriptor, follow_symlinks=False
        )
    except (OSError, TypeError, NotImplementedError) as error:
        raise CampaignJournalError(
            "cannot inspect uncommitted journal pending file"
        ) from error
    mode = stat.S_IMODE(pending_stat.st_mode)
    if (
        not stat.S_ISREG(pending_stat.st_mode)
        or pending_stat.st_uid != owner
        or pending_stat.st_nlink != 1
        or mode & ~0o600
    ):
        raise CampaignJournalError(
            "uncommitted journal pending file is unsafe to discard"
        )


def _recover_pending_publication(
    directory_descriptor: int,
    journal_dir: Path,
    *,
    owner: int,
    expected_identity: tuple[int, int],
    campaign_configuration_sha256: str,
    expected_anchored_head: tuple[int, str] | None,
) -> None:
    finals, pending = _classified_journal_names(directory_descriptor)
    if not pending:
        return
    pending_name, sequence, digest = pending[0]
    existing = finals.get(sequence)
    if existing is None:
        if expected_anchored_head is not None \
                and sequence <= expected_anchored_head[0]:
            raise CampaignJournalError(
                "anchored journal entry exists only as a pending file"
            )
        _check_discardable_pending(
            directory_descriptor, pending_name, owner=owner
        )
    else:
        pending_entry, pending_raw, pending_stat, _ = _read_journal_file(
            directory_descriptor,
            pending_name,
            owner=owner,
            allowed_nlinks={2},
            expected_sequence=sequence,
            expected_sha256=digest,
            campaign_configuration_sha256=campaign_configuration_sha256,
        )
        del pending_entry
        final_name = existing
        _, final_raw, final_stat, _ = _read_journal_file(
            directory_descriptor,
            final_name,
            owner=owner,
            allowed_nlinks={2},
            expected_sequence=sequence,
            expected_sha256=digest,
            campaign_configuration_sha256=campaign_configuration_sha256,
        )
        if (
            (pending_stat.st_dev, pending_stat.st_ino)
            != (final_stat.st_dev, final_stat.st_ino)
            or pending_raw != final_raw
        ):
            raise CampaignJournalError(
                "pending and final journal entries are different objects"
            )
    _recheck_journal_path(journal_dir, owner, expected_identity)
    try:
        os.fsync(directory_descriptor)
        os.unlink(pending_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
    except (OSError, TypeError, NotImplementedError) as error:
        raise CampaignJournalError(
            "cannot durably recover campaign journal publication"
        ) from error


def _validate_journal_state(entries: list[dict[str, Any]]) -> None:
    seen_event_ids: set[str] = set()
    seen_invocation_ids: set[str] = set()
    active_invocation: str | None = None
    active_phase: str | None = None
    previous_time: datetime | None = None
    for entry in entries:
        event_id = entry["event_id"]
        if event_id in seen_event_ids:
            raise CampaignJournalError("campaign journal event_id is not unique")
        seen_event_ids.add(event_id)
        recorded = _parse_journal_time(entry["recorded_at_utc"])
        if recorded is None or (
            previous_time is not None and recorded < previous_time
        ):
            raise CampaignJournalError("campaign journal timestamps run backwards")
        previous_time = recorded
        event = entry["event"]
        invocation = entry["invocation_id"]
        if event == "attempt_prepared":
            if active_invocation is not None:
                raise CampaignJournalError(
                    "campaign journal starts an attempt before the prior one is terminal"
                )
            if invocation in seen_invocation_ids:
                raise CampaignJournalError(
                    "campaign journal invocation_id is not unique"
                )
            seen_invocation_ids.add(invocation)
            active_invocation = invocation
            active_phase = event
        elif event == "process_spawned":
            if active_invocation != invocation or active_phase != "attempt_prepared":
                raise CampaignJournalError(
                    "campaign journal spawned event lacks matching prepared attempt"
                )
            active_phase = event
        elif event == "attempt_terminal":
            if active_invocation != invocation or active_phase not in {
                "attempt_prepared", "process_spawned",
            }:
                raise CampaignJournalError(
                    "campaign journal terminal event lacks an open attempt"
                )
            active_invocation = None
            active_phase = None


def _scan_journal(
    directory_descriptor: int,
    *,
    owner: int,
    campaign_configuration_sha256: str,
) -> list[tuple[dict[str, Any], CampaignJournalReceipt]]:
    initial_names = _journal_names(directory_descriptor)
    finals, pending = _classified_journal_names(
        directory_descriptor, names=initial_names
    )
    if pending:
        raise CampaignJournalError("campaign journal recovery left a pending file")
    sequences = sorted(finals)
    if sequences != list(range(len(sequences))):
        raise CampaignJournalError("campaign journal sequence has a gap")
    records: list[tuple[dict[str, Any], CampaignJournalReceipt]] = []
    previous_sha256: str | None = None
    for sequence in sequences:
        name = finals[sequence]
        entry, _, _, digest = _read_journal_file(
            directory_descriptor,
            name,
            owner=owner,
            allowed_nlinks={1},
            expected_sequence=sequence,
            expected_sha256=None,
            campaign_configuration_sha256=campaign_configuration_sha256,
        )
        if entry["previous_entry_sha256"] != previous_sha256:
            raise CampaignJournalError(
                "campaign journal hash-chain predecessor mismatch"
            )
        receipt = CampaignJournalReceipt(
            sequence=sequence,
            filename=name,
            sha256=digest,
            event_id=entry["event_id"],
            event=entry["event"],
            invocation_id=entry["invocation_id"],
            created=False,
        )
        records.append((entry, receipt))
        previous_sha256 = digest
    _validate_journal_state([entry for entry, _ in records])
    if _journal_names(directory_descriptor) != initial_names:
        raise CampaignJournalError("campaign journal changed during scan")
    return records


def _anchored_journal_head(
    expected_anchored_head: tuple[int, str] | None,
) -> tuple[int, str] | None:
    if expected_anchored_head is None:
        return None
    if (
        not isinstance(expected_anchored_head, tuple)
        or len(expected_anchored_head) != 2
        or isinstance(expected_anchored_head[0], bool)
        or not isinstance(expected_anchored_head[0], int)
        or expected_anchored_head[0] < 0
        or not isinstance(expected_anchored_head[1], str)
        or _SHA256_RE.fullmatch(expected_anchored_head[1]) is None
    ):
        raise CampaignJournalError("expected anchored journal head is invalid")
    return expected_anchored_head


def _require_anchored_journal_head(
    records: list[tuple[dict[str, Any], CampaignJournalReceipt]],
    expected_anchored_head: tuple[int, str] | None,
) -> None:
    if expected_anchored_head is None:
        return
    sequence, digest = expected_anchored_head
    if sequence >= len(records) or records[sequence][1].sha256 != digest:
        raise CampaignJournalError(
            "campaign journal does not contain the expected anchored head"
        )


def _recover_and_scan_journal(
    directory_descriptor: int,
    journal_dir: Path,
    *,
    owner: int,
    expected_identity: tuple[int, int],
    campaign_configuration_sha256: str,
    expected_anchored_head: tuple[int, str] | None,
) -> list[tuple[dict[str, Any], CampaignJournalReceipt]]:
    _recover_pending_publication(
        directory_descriptor,
        journal_dir,
        owner=owner,
        expected_identity=expected_identity,
        campaign_configuration_sha256=campaign_configuration_sha256,
        expected_anchored_head=expected_anchored_head,
    )
    records = _scan_journal(
        directory_descriptor,
        owner=owner,
        campaign_configuration_sha256=campaign_configuration_sha256,
    )
    _require_anchored_journal_head(records, expected_anchored_head)
    _recheck_journal_path(journal_dir, owner, expected_identity)
    return records


def recover_campaign_journal(
    journal_dir: Path,
    *,
    expected_identity: tuple[int, int],
    campaign_configuration_sha256: str,
    expected_anchored_head: tuple[int, str] | None = None,
    expected_uid: int | None = None,
) -> tuple[CampaignJournalReceipt, ...]:
    """Recover residue under lock and verify any externally persisted head."""
    owner = _journal_owner(expected_uid)
    identity = _journal_identity(expected_identity)
    anchored_head = _anchored_journal_head(expected_anchored_head)
    if not isinstance(campaign_configuration_sha256, str) \
            or _SHA256_RE.fullmatch(campaign_configuration_sha256) is None:
        raise CampaignJournalError("campaign configuration SHA-256 is invalid")
    descriptor = _open_journal_directory(Path(journal_dir), owner, identity)
    try:
        records = _recover_and_scan_journal(
            descriptor,
            Path(journal_dir),
            owner=owner,
            expected_identity=identity,
            campaign_configuration_sha256=campaign_configuration_sha256,
            expected_anchored_head=anchored_head,
        )
        return tuple(receipt for _, receipt in records)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise CampaignJournalError("campaign journal write made no progress")
        view = view[written:]


def _publish_journal_entry(
    directory_descriptor: int,
    journal_dir: Path,
    entry: Mapping[str, Any],
    *,
    owner: int,
    expected_identity: tuple[int, int],
) -> tuple[str, str]:
    raw = _canonical_journal_bytes(entry)
    digest = hashlib.sha256(raw).hexdigest()
    sequence = entry["sequence"]
    final_name = f"{sequence:020d}.json"
    pending_name = (
        f".pending-{sequence:020d}-{digest}-{secrets.token_hex(16)}.json"
    )
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    except AttributeError as error:
        raise CampaignJournalError(
            "required secure POSIX open flags are unavailable"
        ) from error
    try:
        descriptor = os.open(
            pending_name, flags, 0o600, dir_fd=directory_descriptor
        )
    except (OSError, TypeError, NotImplementedError) as error:
        raise CampaignJournalError("cannot create pending journal entry") from error
    try:
        if os.get_inheritable(descriptor):
            raise CampaignJournalError("campaign journal descriptor is inheritable")
        _write_all(descriptor, raw)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        pending_stat = os.fstat(descriptor)
        _check_journal_file(pending_stat, owner, {1})
        _recheck_journal_path(journal_dir, owner, expected_identity)
        try:
            os.link(
                pending_name,
                final_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except (OSError, TypeError, NotImplementedError) as error:
            raise CampaignJournalError(
                "cannot atomically publish journal entry without overwrite"
            ) from error
        linked_stat = os.fstat(descriptor)
        _check_journal_file(linked_stat, owner, {2})
        final_stat = os.stat(
            final_name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (final_stat.st_dev, final_stat.st_ino) != (
            linked_stat.st_dev, linked_stat.st_ino
        ):
            raise CampaignJournalError("published journal entry identity mismatch")
        os.fsync(directory_descriptor)
        os.unlink(pending_name, dir_fd=directory_descriptor)
        os.fsync(directory_descriptor)
        final_stat = os.stat(
            final_name, dir_fd=directory_descriptor, follow_symlinks=False
        )
        _check_journal_file(final_stat, owner, {1})
        if (final_stat.st_dev, final_stat.st_ino) != (
            linked_stat.st_dev, linked_stat.st_ino
        ):
            raise CampaignJournalError("journal entry changed after publication")
        _recheck_journal_path(journal_dir, owner, expected_identity)
        return final_name, digest
    except CampaignJournalError:
        raise
    except (OSError, TypeError, NotImplementedError) as error:
        raise CampaignJournalError(
            "campaign journal publication failed in an uncertain state"
        ) from error
    finally:
        os.close(descriptor)


def append_campaign_journal_event(
    journal_dir: Path,
    *,
    expected_identity: tuple[int, int],
    campaign_configuration_sha256: str,
    event_id: str,
    event: str,
    invocation_id: str,
    recorded_at_utc: str,
    payload: Mapping[str, Any],
    expected_anchored_head: tuple[int, str] | None = None,
    expected_uid: int | None = None,
) -> CampaignJournalReceipt:
    """Append under lock and report whether this journal event is new.

    Only a new ``attempt_prepared`` event whose receipt is subsequently fsynced
    into external active-run evidence may authorize spawn.  ``created`` on the
    spawned or terminal events never authorizes repeating their side effects.
    """
    owner = _journal_owner(expected_uid)
    identity = _journal_identity(expected_identity)
    anchored_head = _anchored_journal_head(expected_anchored_head)
    if not isinstance(campaign_configuration_sha256, str) \
            or _SHA256_RE.fullmatch(campaign_configuration_sha256) is None:
        raise CampaignJournalError("campaign configuration SHA-256 is invalid")
    descriptor = _open_journal_directory(Path(journal_dir), owner, identity)
    try:
        records = _recover_and_scan_journal(
            descriptor,
            Path(journal_dir),
            owner=owner,
            expected_identity=identity,
            campaign_configuration_sha256=campaign_configuration_sha256,
            expected_anchored_head=anchored_head,
        )
        proposed_body = {
            "schema_version": JOURNAL_ENTRY_SCHEMA,
            "event_id": event_id,
            "event": event,
            "campaign_configuration_sha256": campaign_configuration_sha256,
            "invocation_id": invocation_id,
            "recorded_at_utc": recorded_at_utc,
            "payload": dict(payload) if isinstance(payload, Mapping) else payload,
        }
        for existing, receipt in records:
            existing_body = {
                name: existing[name]
                for name in proposed_body
            }
            if existing["event_id"] == event_id:
                if _canonical_journal_bytes(existing_body) \
                        != _canonical_journal_bytes(proposed_body):
                    raise CampaignJournalError(
                        "campaign journal event_id conflicts with existing event"
                    )
                return CampaignJournalReceipt(
                    sequence=receipt.sequence,
                    filename=receipt.filename,
                    sha256=receipt.sha256,
                    event_id=receipt.event_id,
                    event=receipt.event,
                    invocation_id=receipt.invocation_id,
                    created=False,
                )
        sequence = len(records)
        previous = records[-1][1].sha256 if records else None
        entry = {
            **proposed_body,
            "sequence": sequence,
            "previous_entry_sha256": previous,
        }
        _validate_journal_entry(
            entry,
            expected_sequence=sequence,
            campaign_configuration_sha256=campaign_configuration_sha256,
        )
        _validate_journal_state([item for item, _ in records] + [entry])
        filename, digest = _publish_journal_entry(
            descriptor,
            Path(journal_dir),
            entry,
            owner=owner,
            expected_identity=identity,
        )
        verified = _recover_and_scan_journal(
            descriptor,
            Path(journal_dir),
            owner=owner,
            expected_identity=identity,
            campaign_configuration_sha256=campaign_configuration_sha256,
            expected_anchored_head=anchored_head,
        )
        if len(verified) != sequence + 1 \
                or verified[-1][1].filename != filename \
                or verified[-1][1].sha256 != digest:
            raise CampaignJournalError(
                "campaign journal append was not durably observable"
            )
        return CampaignJournalReceipt(
            sequence=sequence,
            filename=filename,
            sha256=digest,
            event_id=event_id,
            event=event,
            invocation_id=invocation_id,
            created=True,
        )
    finally:
        os.close(descriptor)


def main() -> int:
    print(
        "ARCH-COMP26 launch remains disabled: atomic audit/spawn/journal/finalize "
        "integration is not implemented.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
