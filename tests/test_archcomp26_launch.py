import json
import hashlib
import multiprocessing
import os
from pathlib import Path
import stat
import threading

import pytest

from torch_tm_flowpipe.archcomp26_launch import (
    CampaignJournalError,
    CampaignLockBusy,
    CampaignLockError,
    append_campaign_journal_event,
    exclusive_campaign_lock,
    main,
    recover_campaign_journal,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the campaign launcher requires POSIX flock"
)

_TIMEOUT_S = 30.0
_CAMPAIGN_CONFIGURATION_SHA256 = "a" * 64


def _provision_lock(lock_path):
    lock_path.touch(mode=0o600, exist_ok=False)
    lock_path.chmod(0o600)
    lock_stat = lock_path.stat()
    return lock_stat.st_dev, lock_stat.st_ino


def _provision_journal(journal_path):
    journal_path.mkdir(mode=0o700)
    journal_path.chmod(0o700)
    journal_stat = journal_path.stat()
    return journal_stat.st_dev, journal_stat.st_ino


def _append_event(
    journal_path,
    journal_identity,
    *,
    event_id,
    event,
    invocation_id="invocation-1",
    recorded_at_utc="2026-01-01T00:00:00Z",
    payload=None,
    expected_anchored_head=None,
):
    return append_campaign_journal_event(
        journal_path,
        expected_identity=journal_identity,
        campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
        event_id=event_id,
        event=event,
        invocation_id=invocation_id,
        recorded_at_utc=recorded_at_utc,
        payload={} if payload is None else payload,
        expected_anchored_head=expected_anchored_head,
        expected_uid=os.getuid(),
    )


def _hold_lock(lock_path, expected_identity, acquired, release):
    with exclusive_campaign_lock(
        Path(lock_path),
        expected_identity=expected_identity,
        expected_uid=os.getuid(),
    ):
        acquired.set()
        if not release.wait(_TIMEOUT_S):
            raise TimeoutError("test did not release the campaign lock")


def _try_launch(lock_path, expected_identity, busy, audit_ran, spawn_ran):
    try:
        with exclusive_campaign_lock(
            Path(lock_path),
            expected_identity=expected_identity,
            expected_uid=os.getuid(),
        ):
            audit_ran.set()
            spawn_ran.set()
    except CampaignLockBusy:
        busy.set()


def _try_recover_journal(journal_path, expected_identity, rejected):
    try:
        recover_campaign_journal(
            Path(journal_path),
            expected_identity=expected_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )
    except CampaignJournalError:
        rejected.set()


def _hold_through_terminal_fsync(
    lock_path,
    expected_identity,
    terminal_path,
    child_exited,
    allow_file_fsync,
    file_fsynced,
    allow_directory_fsync,
    directory_fsynced,
):
    terminal_path = Path(terminal_path)
    temporary_path = terminal_path.with_suffix(".tmp")

    with exclusive_campaign_lock(
        Path(lock_path),
        expected_identity=expected_identity,
        expected_uid=os.getuid(),
    ):
        child_exited.set()
        if not allow_file_fsync.wait(_TIMEOUT_S):
            raise TimeoutError("test did not allow terminal-file fsync")

        with temporary_path.open("wb") as stream:
            stream.write(b'{"status":"completed"}\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, terminal_path)
        file_fsynced.set()

        if not allow_directory_fsync.wait(_TIMEOUT_S):
            raise TimeoutError("test did not allow terminal-directory fsync")
        directory_fd = os.open(
            terminal_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        directory_fsynced.set()


def _join(process):
    process.join(_TIMEOUT_S)
    assert not process.is_alive(), f"process {process.name} did not finish"
    assert process.exitcode == 0


def _stop(process):
    if process is None:
        return
    if process.is_alive():
        process.terminate()
    process.join(5)


def _assert_launch_is_busy(context, lock_path, expected_identity):
    busy = context.Event()
    audit_ran = context.Event()
    spawn_ran = context.Event()
    contender = context.Process(
        target=_try_launch,
        args=(str(lock_path), expected_identity, busy, audit_ran, spawn_ran),
    )
    try:
        contender.start()
        _join(contender)
        assert busy.is_set()
        assert not audit_ran.is_set()
        assert not spawn_ran.is_set()
    finally:
        _stop(contender)


def test_second_process_fails_closed_before_audit_or_spawn(tmp_path):
    context = multiprocessing.get_context("spawn")
    lock_path = tmp_path / "campaign.lock"
    expected_identity = _provision_lock(lock_path)
    acquired = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_lock,
        args=(str(lock_path), expected_identity, acquired, release),
    )

    try:
        holder.start()
        assert acquired.wait(_TIMEOUT_S)
        locked_inode = lock_path.stat().st_ino

        _assert_launch_is_busy(context, lock_path, expected_identity)

        release.set()
        _join(holder)
        assert lock_path.stat().st_ino == locked_inode
        with exclusive_campaign_lock(
            lock_path,
            expected_identity=expected_identity,
            expected_uid=os.getuid(),
        ) as lock_handle:
            assert lock_handle is None
    finally:
        release.set()
        _stop(holder)


def test_context_never_exposes_a_descriptor_that_can_release_the_lock(tmp_path):
    context = multiprocessing.get_context("spawn")
    lock_path = tmp_path / "campaign.lock"
    expected_identity = _provision_lock(lock_path)

    with exclusive_campaign_lock(
        lock_path,
        expected_identity=expected_identity,
        expected_uid=os.getuid(),
    ) as lock_handle:
        assert lock_handle is None
        with pytest.raises(TypeError):
            os.close(lock_handle)
        _assert_launch_is_busy(context, lock_path, expected_identity)


def test_lock_covers_child_exit_through_terminal_file_and_directory_fsync(
    tmp_path,
):
    context = multiprocessing.get_context("spawn")
    lock_path = tmp_path / "campaign.lock"
    expected_identity = _provision_lock(lock_path)
    terminal_path = tmp_path / "terminal.json"
    child_exited = context.Event()
    allow_file_fsync = context.Event()
    file_fsynced = context.Event()
    allow_directory_fsync = context.Event()
    directory_fsynced = context.Event()
    holder = context.Process(
        target=_hold_through_terminal_fsync,
        args=(
            str(lock_path),
            expected_identity,
            str(terminal_path),
            child_exited,
            allow_file_fsync,
            file_fsynced,
            allow_directory_fsync,
            directory_fsynced,
        ),
    )

    try:
        holder.start()
        assert child_exited.wait(_TIMEOUT_S)
        _assert_launch_is_busy(context, lock_path, expected_identity)

        allow_file_fsync.set()
        assert file_fsynced.wait(_TIMEOUT_S)
        _assert_launch_is_busy(context, lock_path, expected_identity)

        allow_directory_fsync.set()
        assert directory_fsynced.wait(_TIMEOUT_S)
        _join(holder)

        assert json.loads(terminal_path.read_text()) == {"status": "completed"}
        assert lock_path.exists()
        with exclusive_campaign_lock(
            lock_path,
            expected_identity=expected_identity,
            expected_uid=os.getuid(),
        ):
            pass
    finally:
        allow_file_fsync.set()
        allow_directory_fsync.set()
        _stop(holder)


def test_replaced_lock_path_cannot_create_a_second_lock(tmp_path):
    lock_path = tmp_path / "campaign.lock"
    expected_identity = _provision_lock(lock_path)

    with pytest.raises(CampaignLockError, match="changed while held"):
        with exclusive_campaign_lock(
            lock_path,
            expected_identity=expected_identity,
            expected_uid=os.getuid(),
        ):
            lock_path.unlink()
            replacement_identity = _provision_lock(lock_path)
            assert replacement_identity != expected_identity
            with pytest.raises(CampaignLockError, match="frozen identity"):
                with exclusive_campaign_lock(
                    lock_path,
                    expected_identity=expected_identity,
                    expected_uid=os.getuid(),
                ):
                    pytest.fail("replacement inode must never become the campaign lock")


def test_symlinked_ancestor_is_rejected(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    lock_path = real_parent / "campaign.lock"
    expected_identity = _provision_lock(lock_path)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(CampaignLockError, match="securely open"):
        with exclusive_campaign_lock(
            alias / lock_path.name,
            expected_identity=expected_identity,
            expected_uid=os.getuid(),
        ):
            pytest.fail("a symlinked path component must never be accepted")


def test_group_writable_parent_and_non_0600_lock_are_rejected(tmp_path):
    group_parent = tmp_path / "group-writable"
    group_parent.mkdir(mode=0o700)
    group_lock = group_parent / "campaign.lock"
    group_identity = _provision_lock(group_lock)
    group_parent.chmod(0o770)
    with pytest.raises(CampaignLockError, match="unsafe identity or mode"):
        with exclusive_campaign_lock(
            group_lock,
            expected_identity=group_identity,
            expected_uid=os.getuid(),
        ):
            pytest.fail("a group-writable parent must never be accepted")

    mode_lock = tmp_path / "wrong-mode.lock"
    mode_identity = _provision_lock(mode_lock)
    mode_lock.chmod(0o700)
    with pytest.raises(CampaignLockError, match="unsafe identity or mode"):
        with exclusive_campaign_lock(
            mode_lock,
            expected_identity=mode_identity,
            expected_uid=os.getuid(),
        ):
                pytest.fail("a non-0600 lock must never be accepted")


def test_journal_append_is_canonical_chained_and_idempotent(tmp_path):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    prepared = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
        payload={"audit": {"path": "audit.json", "sha256": "b" * 64}},
    )
    assert prepared.created is True
    retried = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
        payload={"audit": {"path": "audit.json", "sha256": "b" * 64}},
    )
    assert retried == type(retried)(
        sequence=prepared.sequence,
        filename=prepared.filename,
        sha256=prepared.sha256,
        event_id=prepared.event_id,
        event=prepared.event,
        invocation_id=prepared.invocation_id,
        created=False,
    )
    spawned = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:spawned",
        event="process_spawned",
        recorded_at_utc="2026-01-01T00:00:01Z",
        payload={"pid": True},
    )
    assert spawned.sequence == 1 and spawned.created is True

    paths = sorted(journal_path.iterdir())
    assert [path.name for path in paths] == [prepared.filename, spawned.filename]
    entries = []
    for receipt, path in zip((prepared, spawned), paths):
        raw = path.read_bytes()
        entry = json.loads(raw)
        assert raw == (
            json.dumps(
                entry,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8") + b"\n"
        )
        assert hashlib.sha256(raw).hexdigest() == receipt.sha256
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
        assert path.stat().st_nlink == 1
        entries.append(entry)
    assert entries[0]["previous_entry_sha256"] is None
    assert entries[1]["previous_entry_sha256"] == prepared.sha256
    recovered = recover_campaign_journal(
        journal_path,
        expected_identity=journal_identity,
        campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
        expected_uid=os.getuid(),
    )
    assert [receipt.sha256 for receipt in recovered] == [
        prepared.sha256, spawned.sha256,
    ]
    assert all(receipt.created is False for receipt in recovered)
    with pytest.raises(CampaignJournalError, match="event_id conflicts"):
        _append_event(
            journal_path,
            journal_identity,
            event_id="invocation-1:prepared",
            event="attempt_prepared",
            payload={"changed": True},
        )
    with pytest.raises(CampaignJournalError, match="event_id conflicts"):
        _append_event(
            journal_path,
            journal_identity,
            event_id="invocation-1:spawned",
            event="process_spawned",
            recorded_at_utc="2026-01-01T00:00:01Z",
            payload={"pid": 1},
        )


def test_journal_state_machine_blocks_a_second_unresolved_attempt(tmp_path):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    with pytest.raises(CampaignJournalError, match="prior one is terminal"):
        _append_event(
            journal_path,
            journal_identity,
            event_id="invocation-2:prepared",
            event="attempt_prepared",
            invocation_id="invocation-2",
            recorded_at_utc="2026-01-01T00:00:01Z",
        )
    _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:terminal",
        event="attempt_terminal",
        recorded_at_utc="2026-01-01T00:00:01Z",
        payload={"outcome": "process_error"},
    )
    second = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-2:prepared",
        event="attempt_prepared",
        invocation_id="invocation-2",
        recorded_at_utc="2026-01-01T00:00:02Z",
    )
    assert second.created is True


def test_journal_state_machine_rejects_reused_invocation_id(tmp_path):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:terminal",
        event="attempt_terminal",
        recorded_at_utc="2026-01-01T00:00:01Z",
    )
    with pytest.raises(CampaignJournalError, match="invocation_id is not unique"):
        _append_event(
            journal_path,
            journal_identity,
            event_id="invocation-1:prepared-again",
            event="attempt_prepared",
            recorded_at_utc="2026-01-01T00:00:02Z",
        )


def test_journal_anchor_allows_suffix_and_rejects_tail_loss_or_rewrite(tmp_path):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    prepared = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    anchor = (prepared.sequence, prepared.sha256)
    spawned = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:spawned",
        event="process_spawned",
        recorded_at_utc="2026-01-01T00:00:01Z",
        expected_anchored_head=anchor,
    )
    recovered = recover_campaign_journal(
        journal_path,
        expected_identity=journal_identity,
        campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
        expected_anchored_head=anchor,
        expected_uid=os.getuid(),
    )
    assert [receipt.sha256 for receipt in recovered] == [
        prepared.sha256, spawned.sha256,
    ]

    (journal_path / spawned.filename).unlink()
    with pytest.raises(CampaignJournalError, match="expected anchored head"):
        recover_campaign_journal(
            journal_path,
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_anchored_head=(spawned.sequence, spawned.sha256),
            expected_uid=os.getuid(),
        )

    prepared_path = journal_path / prepared.filename
    entry = json.loads(prepared_path.read_bytes())
    entry["payload"] = {"tampered": True}
    prepared_path.chmod(0o600)
    prepared_path.write_bytes(
        json.dumps(
            entry,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    )
    prepared_path.chmod(0o400)
    with pytest.raises(CampaignJournalError, match="expected anchored head"):
        recover_campaign_journal(
            journal_path,
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_anchored_head=anchor,
            expected_uid=os.getuid(),
        )


def test_journal_preserves_pending_only_copy_of_anchored_entry(tmp_path):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    receipt = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    final_path = journal_path / receipt.filename
    pending_path = journal_path / (
        f".pending-{receipt.sequence:020d}-{receipt.sha256}-{'e' * 32}.json"
    )
    os.link(final_path, pending_path)
    final_path.unlink()
    assert pending_path.stat().st_nlink == 1

    with pytest.raises(CampaignJournalError, match="only as a pending file"):
        recover_campaign_journal(
            journal_path,
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_anchored_head=(receipt.sequence, receipt.sha256),
            expected_uid=os.getuid(),
        )
    assert pending_path.exists()
    assert pending_path.read_bytes()


@pytest.mark.parametrize(
    "anchor",
    [
        (False, "a" * 64),
        (0.0, "a" * 64),
        (-1, "a" * 64),
        (0, "not-a-sha256"),
    ],
)
def test_journal_rejects_invalid_external_anchor(tmp_path, anchor):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    with pytest.raises(CampaignJournalError, match="anchored journal head is invalid"):
        recover_campaign_journal(
            journal_path,
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_anchored_head=anchor,
            expected_uid=os.getuid(),
        )


def test_journal_publish_fsyncs_file_then_both_directory_transitions(
    tmp_path, monkeypatch,
):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    operations = []
    original_fsync = os.fsync
    original_link = os.link
    original_unlink = os.unlink

    def recording_fsync(descriptor):
        kind = "dir_fsync" if stat.S_ISDIR(os.fstat(descriptor).st_mode) \
            else "file_fsync"
        operations.append(kind)
        return original_fsync(descriptor)

    def recording_link(*args, **kwargs):
        operations.append("link")
        return original_link(*args, **kwargs)

    def recording_unlink(*args, **kwargs):
        operations.append("unlink")
        return original_unlink(*args, **kwargs)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(os, "link", recording_link)
    monkeypatch.setattr(os, "unlink", recording_unlink)
    _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    assert operations == [
        "file_fsync", "link", "dir_fsync", "unlink", "dir_fsync",
    ]


def test_journal_recovers_linked_pending_publication(tmp_path, monkeypatch):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    original_unlink = os.unlink

    def fail_pending_unlink(path, *args, **kwargs):
        if str(path).startswith(".pending-"):
            raise OSError("injected crash before pending cleanup")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(os, "unlink", fail_pending_unlink)
        with pytest.raises(CampaignJournalError, match="uncertain state"):
            _append_event(
                journal_path,
                journal_identity,
                event_id="invocation-1:prepared",
                event="attempt_prepared",
            )
    assert len(list(journal_path.iterdir())) == 2
    retried = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    assert retried.created is False
    assert len(list(journal_path.iterdir())) == 1
    receipts = recover_campaign_journal(
        journal_path,
        expected_identity=journal_identity,
        campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
        expected_uid=os.getuid(),
    )
    assert len(receipts) == 1
    assert len(list(journal_path.iterdir())) == 1
    assert next(journal_path.iterdir()).stat().st_nlink == 1


def test_journal_path_replacement_during_publication_fails_closed(
    tmp_path, monkeypatch,
):
    journal_path = tmp_path / "journal"
    moved_path = tmp_path / "moved-journal"
    journal_identity = _provision_journal(journal_path)
    original_link = os.link

    def replace_path_then_link(*args, **kwargs):
        journal_path.rename(moved_path)
        _provision_journal(journal_path)
        return original_link(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(os, "link", replace_path_then_link)
        with pytest.raises(CampaignJournalError, match="frozen 0700 identity"):
            _append_event(
                journal_path,
                journal_identity,
                event_id="invocation-1:prepared",
                event="attempt_prepared",
            )

    assert list(journal_path.iterdir()) == []
    receipts = recover_campaign_journal(
        moved_path,
        expected_identity=journal_identity,
        campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
        expected_uid=os.getuid(),
    )
    assert len(receipts) == 1
    assert receipts[0].event_id == "invocation-1:prepared"


def test_journal_discards_complete_uncommitted_pending_file(tmp_path, monkeypatch):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)

    def fail_link(*args, **kwargs):
        raise OSError("injected crash before publication")

    with monkeypatch.context() as context:
        context.setattr(os, "link", fail_link)
        with pytest.raises(CampaignJournalError, match="without overwrite"):
            _append_event(
                journal_path,
                journal_identity,
                event_id="invocation-1:prepared",
                event="attempt_prepared",
            )
    assert len(list(journal_path.iterdir())) == 1
    assert recover_campaign_journal(
        journal_path,
        expected_identity=journal_identity,
        campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
        expected_uid=os.getuid(),
    ) == ()
    assert list(journal_path.iterdir()) == []


def test_journal_discards_partial_uncommitted_pending_file(tmp_path):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    pending_path = journal_path / (
        f".pending-{0:020d}-{'c' * 64}-{'d' * 32}.json"
    )
    pending_path.write_bytes(b"{partial")
    pending_path.chmod(0o600)
    assert recover_campaign_journal(
        journal_path,
        expected_identity=journal_identity,
        campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
        expected_uid=os.getuid(),
    ) == ()
    assert list(journal_path.iterdir()) == []


def test_journal_fixed_sequence_name_prevents_concurrent_fork(
    tmp_path, monkeypatch,
):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    barrier = threading.Barrier(2)
    original_link = os.link
    errors = []

    def synchronized_link(*args, **kwargs):
        barrier.wait(timeout=_TIMEOUT_S)
        return original_link(*args, **kwargs)

    def append(invocation_id):
        try:
            _append_event(
                journal_path,
                journal_identity,
                event_id=f"{invocation_id}:prepared",
                event="attempt_prepared",
                invocation_id=invocation_id,
                payload={"invocation": invocation_id},
            )
        except CampaignJournalError as error:
            errors.append(error)

    monkeypatch.setattr(os, "link", synchronized_link)
    threads = [
        threading.Thread(target=append, args=(f"invocation-{index}",))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_TIMEOUT_S)
        assert not thread.is_alive()
    assert len(errors) == 2
    assert [path.name for path in journal_path.glob("[0-9]*.json")] == [
        f"{0:020d}.json"
    ]
    with pytest.raises(CampaignJournalError):
        recover_campaign_journal(
            journal_path,
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )


def test_journal_rejects_conflicting_pending_inode(tmp_path):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    receipt = _append_event(
        journal_path,
        journal_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    final_path = journal_path / receipt.filename
    pending_path = journal_path / (
        f".pending-{receipt.sequence:020d}-{receipt.sha256}-{'f' * 32}.json"
    )
    pending_path.write_bytes(final_path.read_bytes())
    pending_path.chmod(0o400)
    with pytest.raises(CampaignJournalError, match="link count"):
        recover_campaign_journal(
            journal_path,
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )
    assert final_path.read_bytes() == pending_path.read_bytes()


def test_journal_rejects_unknown_noncanonical_and_external_hardlink(tmp_path):
    unknown_path = tmp_path / "unknown-journal"
    unknown_identity = _provision_journal(unknown_path)
    (unknown_path / ".DS_Store").write_bytes(b"unexpected")
    with pytest.raises(CampaignJournalError, match="unknown file"):
        recover_campaign_journal(
            unknown_path,
            expected_identity=unknown_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )

    noncanonical_path = tmp_path / "noncanonical-journal"
    noncanonical_identity = _provision_journal(noncanonical_path)
    entry = {
        "schema_version": "archcomp26-campaign-journal-entry-v1",
        "sequence": 0,
        "previous_entry_sha256": None,
        "event_id": "invocation-1:prepared",
        "event": "attempt_prepared",
        "campaign_configuration_sha256": _CAMPAIGN_CONFIGURATION_SHA256,
        "invocation_id": "invocation-1",
        "recorded_at_utc": "2026-01-01T00:00:00Z",
        "payload": {},
    }
    raw = (json.dumps(entry, indent=2) + "\n").encode()
    entry_path = noncanonical_path / f"{0:020d}.json"
    entry_path.write_bytes(raw)
    entry_path.chmod(0o400)
    with pytest.raises(CampaignJournalError, match="not canonical"):
        recover_campaign_journal(
            noncanonical_path,
            expected_identity=noncanonical_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )

    linked_path = tmp_path / "linked-journal"
    linked_identity = _provision_journal(linked_path)
    receipt = _append_event(
        linked_path,
        linked_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    external_link = tmp_path / "external-link.json"
    os.link(linked_path / receipt.filename, external_link)
    with pytest.raises(CampaignJournalError, match="link count"):
        recover_campaign_journal(
            linked_path,
            expected_identity=linked_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )


def test_journal_rejects_fifo_entry_without_blocking(tmp_path):
    context = multiprocessing.get_context("spawn")
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    os.mkfifo(journal_path / f"{0:020d}.json", mode=0o400)
    rejected = context.Event()
    process = context.Process(
        target=_try_recover_journal,
        args=(str(journal_path), journal_identity, rejected),
    )
    try:
        process.start()
        _join(process)
        assert rejected.is_set()
    finally:
        _stop(process)


@pytest.mark.parametrize("invalid_sequence", [False, 0.0])
def test_journal_rejects_non_integer_sequence(tmp_path, invalid_sequence):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    entry = {
        "schema_version": "archcomp26-campaign-journal-entry-v1",
        "sequence": invalid_sequence,
        "previous_entry_sha256": None,
        "event_id": "invocation-1:prepared",
        "event": "attempt_prepared",
        "campaign_configuration_sha256": _CAMPAIGN_CONFIGURATION_SHA256,
        "invocation_id": "invocation-1",
        "recorded_at_utc": "2026-01-01T00:00:00Z",
        "payload": {},
    }
    path = journal_path / f"{0:020d}.json"
    path.write_bytes(
        json.dumps(
            entry,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    )
    path.chmod(0o400)
    with pytest.raises(CampaignJournalError, match="sequence is not contiguous"):
        recover_campaign_journal(
            journal_path,
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )


@pytest.mark.parametrize("invalid_event", [[], {}])
def test_journal_rejects_unhashable_event_as_schema_error(tmp_path, invalid_event):
    journal_path = tmp_path / "journal"
    journal_identity = _provision_journal(journal_path)
    entry = {
        "schema_version": "archcomp26-campaign-journal-entry-v1",
        "sequence": 0,
        "previous_entry_sha256": None,
        "event_id": "invocation-1:prepared",
        "event": invalid_event,
        "campaign_configuration_sha256": _CAMPAIGN_CONFIGURATION_SHA256,
        "invocation_id": "invocation-1",
        "recorded_at_utc": "2026-01-01T00:00:00Z",
        "payload": {},
    }
    path = journal_path / f"{0:020d}.json"
    path.write_bytes(
        json.dumps(
            entry,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    )
    path.chmod(0o400)
    with pytest.raises(CampaignJournalError, match="event type is invalid"):
        recover_campaign_journal(
            journal_path,
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )


def test_journal_rejects_sequence_gap_and_predecessor_tamper(tmp_path):
    gap_path = tmp_path / "gap-journal"
    gap_identity = _provision_journal(gap_path)
    first = _append_event(
        gap_path,
        gap_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    _append_event(
        gap_path,
        gap_identity,
        event_id="invocation-1:spawned",
        event="process_spawned",
        recorded_at_utc="2026-01-01T00:00:01Z",
    )
    (gap_path / first.filename).unlink()
    with pytest.raises(CampaignJournalError, match="sequence has a gap"):
        recover_campaign_journal(
            gap_path,
            expected_identity=gap_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )

    chain_path = tmp_path / "chain-journal"
    chain_identity = _provision_journal(chain_path)
    _append_event(
        chain_path,
        chain_identity,
        event_id="invocation-1:prepared",
        event="attempt_prepared",
    )
    second = _append_event(
        chain_path,
        chain_identity,
        event_id="invocation-1:spawned",
        event="process_spawned",
        recorded_at_utc="2026-01-01T00:00:01Z",
    )
    second_path = chain_path / second.filename
    entry = json.loads(second_path.read_bytes())
    entry["previous_entry_sha256"] = "f" * 64
    second_path.chmod(0o600)
    second_path.write_bytes(
        json.dumps(
            entry,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
    )
    second_path.chmod(0o400)
    with pytest.raises(CampaignJournalError, match="predecessor mismatch"):
        recover_campaign_journal(
            chain_path,
            expected_identity=chain_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )


def test_journal_rejects_symlinked_ancestor(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    journal_path = real_parent / "journal"
    journal_identity = _provision_journal(journal_path)
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(CampaignJournalError, match="securely open"):
        recover_campaign_journal(
            alias / "journal",
            expected_identity=journal_identity,
            campaign_configuration_sha256=_CAMPAIGN_CONFIGURATION_SHA256,
            expected_uid=os.getuid(),
        )


def test_cli_remains_fail_closed(capsys):
    assert main() == 2
    assert "launch remains disabled" in capsys.readouterr().err
