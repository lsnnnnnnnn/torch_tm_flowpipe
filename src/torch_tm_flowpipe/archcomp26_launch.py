"""Atomic lock primitive for the future ARCH-COMP26 sample launcher.

This module intentionally does not spawn experiments yet.  The production
matrix keeps ``wrapper_sha256=null`` until fresh-audit, sample-journal, process,
and terminal-fsync handling are wired around this lock.

The lock file must be provisioned before the campaign and its device/inode
identity bound by the future campaign receipt.  Acquisition never creates,
unlinks, or replaces it.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import os
from pathlib import Path
import stat
import sys
from typing import Iterator


class CampaignLockError(RuntimeError):
    """The fixed campaign lock is unsafe or cannot be used."""


class CampaignLockBusy(CampaignLockError):
    """Another launcher currently owns the campaign lock."""


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
) -> Iterator[int]:
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
            flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
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
            yield descriptor
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


def main() -> int:
    print(
        "ARCH-COMP26 launch remains disabled: atomic audit/spawn/journal/finalize "
        "integration is not implemented.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
