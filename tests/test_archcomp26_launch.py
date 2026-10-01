import json
import multiprocessing
import os
from pathlib import Path

import pytest

from torch_tm_flowpipe.archcomp26_launch import (
    CampaignLockBusy,
    CampaignLockError,
    exclusive_campaign_lock,
    main,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="the campaign launcher requires POSIX flock"
)

_TIMEOUT_S = 30.0


def _provision_lock(lock_path):
    lock_path.touch(mode=0o600, exist_ok=False)
    lock_path.chmod(0o600)
    lock_stat = lock_path.stat()
    return lock_stat.st_dev, lock_stat.st_ino


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
        ):
            pass
    finally:
        release.set()
        _stop(holder)


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


def test_cli_remains_fail_closed(capsys):
    assert main() == 2
    assert "launch remains disabled" in capsys.readouterr().err
