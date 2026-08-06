from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


AUDIT_BEHAVIOR_ENV_VARS = (
    "FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION",
    "FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT",
)


class BackendIdentityError(RuntimeError):
    """Raised before execution when backend identity is unsafe or ambiguous."""


def _run_git(root: Path, arguments: Sequence[str]) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise BackendIdentityError(
            process.stderr.strip()
            or f"git {' '.join(arguments)} failed for {root}"
        )
    return process.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_behavior_enabled(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def enabled_audit_behavior_variables(
    environment: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    values = os.environ if environment is None else environment
    return tuple(
        name
        for name in AUDIT_BEHAVIOR_ENV_VARS
        if audit_behavior_enabled(values.get(name))
    )


def _is_audit_named_root(root: Path) -> bool:
    name = root.name.lower().replace("_", "-")
    return "flowstar-audit" in name or name.endswith("-audit")


def _is_gcc15_derivative_compatibility_change(
    changed_paths: tuple[str, ...],
    tracked_diff: str,
) -> bool:
    if changed_paths != ("flowstar-toolbox/TaylorModel.h",):
        return False
    removed = [
        line[1:].strip()
        for line in tracked_diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    added = [
        line[1:].strip()
        for line in tracked_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return removed == ["remainder = 0;"] and added == [
        "result.remainder = 0;"
    ]


@dataclass(frozen=True)
class FlowstarBackendIdentity:
    canonical_root: str
    repository_sha: str
    remote: str
    dirty: bool
    status_porcelain_v2: str
    tracked_changed_paths: tuple[str, ...]
    tracked_patch_sha256: str
    backend_class: str
    execution_route: str
    primary_eligible: bool
    gcc15_derivative_compatibility_change: bool
    audit_behavior_variables_enabled: tuple[str, ...]
    patch_manifest: str | None
    patch_manifest_sha256: str | None
    library_path: str | None
    library_sha256: str | None

    def to_record(self) -> dict[str, object]:
        return asdict(self)


def classify_flowstar_backend(
    root: str | Path,
    *,
    environment: Mapping[str, object] | None = None,
    execution_route: str = "generated-stock",
) -> FlowstarBackendIdentity:
    """Record facts without asserting that an ambiguous backend is primary-safe."""
    canonical = Path(root).expanduser().resolve(strict=True)
    sha = _run_git(canonical, ("rev-parse", "HEAD"))
    remote = _run_git(canonical, ("remote", "get-url", "origin"))
    status = _run_git(canonical, ("status", "--porcelain=v2", "--branch"))
    changed_paths = tuple(
        path
        for path in _run_git(
            canonical, ("diff", "HEAD", "--name-only")
        ).splitlines()
        if path
    )
    tracked_diff = _run_git(
        canonical, ("diff", "HEAD", "--no-ext-diff")
    )
    binary_diff = subprocess.run(
        ["git", "-C", str(canonical), "diff", "HEAD", "--binary"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if binary_diff.returncode:
        raise BackendIdentityError(
            binary_diff.stderr.decode("utf-8", errors="replace").strip()
        )
    gcc15_compat = _is_gcc15_derivative_compatibility_change(
        changed_paths, tracked_diff
    )
    if _is_audit_named_root(canonical):
        backend_class = "patched-audit"
        primary_eligible = False
    elif not changed_paths:
        backend_class = "unmodified-stock"
        primary_eligible = True
    elif gcc15_compat:
        backend_class = "stock-plus-gcc15-compat"
        primary_eligible = True
    else:
        backend_class = "unknown-dirty"
        primary_eligible = False
    library = canonical / "flowstar-toolbox" / "libflowstar.a"
    enabled = enabled_audit_behavior_variables(environment)
    dirty_lines = [
        line
        for line in status.splitlines()
        if line and not line.startswith("# branch.")
    ]
    return FlowstarBackendIdentity(
        canonical_root=str(canonical),
        repository_sha=sha,
        remote=remote,
        dirty=bool(dirty_lines),
        status_porcelain_v2=status,
        tracked_changed_paths=changed_paths,
        tracked_patch_sha256=_sha256_bytes(binary_diff.stdout),
        backend_class=backend_class,
        execution_route=execution_route,
        primary_eligible=primary_eligible and not enabled,
        gcc15_derivative_compatibility_change=gcc15_compat,
        audit_behavior_variables_enabled=enabled,
        patch_manifest=None,
        patch_manifest_sha256=None,
        library_path=str(library) if library.is_file() else None,
        library_sha256=_sha256_file(library) if library.is_file() else None,
    )


def inspect_primary_flowstar_backend(
    root: str | Path,
    *,
    environment: Mapping[str, object] | None = None,
    execution_route: str = "generated-stock",
) -> FlowstarBackendIdentity:
    identity = classify_flowstar_backend(
        root,
        environment=environment,
        execution_route=execution_route,
    )
    if _is_audit_named_root(Path(identity.canonical_root)):
        raise BackendIdentityError(
            f"primary Flowstar root is audit-named: {identity.canonical_root}"
        )
    if identity.audit_behavior_variables_enabled:
        raise BackendIdentityError(
            "primary Flowstar environment enables audit behavior: "
            + ", ".join(identity.audit_behavior_variables_enabled)
        )
    if identity.backend_class == "unknown-dirty":
        raise BackendIdentityError(
            "primary Flowstar backend has unknown tracked modifications"
        )
    if identity.library_path is None:
        raise BackendIdentityError(
            "primary Flowstar library is missing: "
            f"{identity.canonical_root}/flowstar-toolbox/libflowstar.a"
        )
    return identity


def inspect_diagnostic_flowstar_backend(
    root: str | Path,
    *,
    patch_manifest: str | Path,
    environment: Mapping[str, object] | None = None,
) -> FlowstarBackendIdentity:
    manifest = Path(patch_manifest).expanduser().resolve(strict=True)
    identity = classify_flowstar_backend(
        root,
        environment=environment,
        execution_route="patched-audit",
    )
    record = identity.to_record()
    record.update(
        {
            "backend_class": "patched-audit",
            "execution_route": "patched-audit",
            "primary_eligible": False,
            "patch_manifest": str(manifest),
            "patch_manifest_sha256": _sha256_file(manifest),
        }
    )
    return FlowstarBackendIdentity(**record)
