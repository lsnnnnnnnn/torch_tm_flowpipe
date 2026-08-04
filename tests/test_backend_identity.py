from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from torch_tm_flowpipe.protocol.backend_identity import (
    BackendIdentityError,
    classify_flowstar_backend,
    inspect_diagnostic_flowstar_backend,
    inspect_primary_flowstar_backend,
)


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _flowstar_repo(root: Path) -> Path:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "backend-test@example.invalid")
    _git(root, "config", "user.name", "Backend Test")
    _git(root, "remote", "add", "origin", "https://example.invalid/flowstar.git")
    toolbox = root / "flowstar-toolbox"
    toolbox.mkdir()
    (toolbox / "TaylorModel.h").write_text(
        "void derivative() {\n\tremainder = 0;\n}\n",
        encoding="utf-8",
    )
    (toolbox / "libflowstar.a").write_bytes(b"test archive")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "initial stock fixture")
    return root


@pytest.mark.unit
@pytest.mark.protocol
def test_clean_stock_backend_is_primary_eligible(tmp_path: Path) -> None:
    root = _flowstar_repo(tmp_path / "flowstar")
    identity = inspect_primary_flowstar_backend(root, environment={})
    assert identity.backend_class == "unmodified-stock"
    assert identity.execution_route == "generated-stock"
    assert identity.primary_eligible is True


@pytest.mark.unit
@pytest.mark.protocol
def test_gcc15_derivative_compatibility_has_factual_label(
    tmp_path: Path,
) -> None:
    root = _flowstar_repo(tmp_path / "flowstar")
    header = root / "flowstar-toolbox" / "TaylorModel.h"
    header.write_text(
        "void derivative() {\n\tresult.remainder = 0;\n}\n",
        encoding="utf-8",
    )
    identity = inspect_primary_flowstar_backend(root, environment={})
    assert identity.backend_class == "stock-plus-gcc15-compat"
    assert identity.gcc15_derivative_compatibility_change is True
    assert identity.dirty is True


@pytest.mark.unit
@pytest.mark.protocol
def test_primary_rejects_audit_named_root(tmp_path: Path) -> None:
    root = _flowstar_repo(tmp_path / "flowstar-audit")
    with pytest.raises(BackendIdentityError, match="audit-named"):
        inspect_primary_flowstar_backend(root, environment={})


@pytest.mark.unit
@pytest.mark.protocol
@pytest.mark.parametrize(
    "variable",
    [
        "FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION",
        "FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT",
    ],
)
def test_primary_rejects_audit_environment_contamination(
    tmp_path: Path,
    variable: str,
) -> None:
    root = _flowstar_repo(tmp_path / "flowstar")
    with pytest.raises(BackendIdentityError, match="audit behavior"):
        inspect_primary_flowstar_backend(root, environment={variable: "1"})


@pytest.mark.unit
@pytest.mark.protocol
def test_unknown_tracked_change_is_labeled_and_rejected(
    tmp_path: Path,
) -> None:
    root = _flowstar_repo(tmp_path / "flowstar")
    (root / "README.md").write_text("local algorithm change\n", encoding="utf-8")
    _git(root, "add", "README.md")
    identity = classify_flowstar_backend(root, environment={})
    assert identity.backend_class == "unknown-dirty"
    assert identity.primary_eligible is False
    with pytest.raises(BackendIdentityError, match="unknown tracked"):
        inspect_primary_flowstar_backend(root, environment={})


@pytest.mark.unit
@pytest.mark.protocol
def test_diagnostic_backend_requires_patch_manifest_and_is_never_primary(
    tmp_path: Path,
) -> None:
    root = _flowstar_repo(tmp_path / "flowstar-diagnostic")
    manifest = tmp_path / "patch-manifest.txt"
    manifest.write_text("0001-diagnostic.patch sha256=fixture\n", encoding="utf-8")
    identity = inspect_diagnostic_flowstar_backend(
        root,
        patch_manifest=manifest,
        environment={"FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION": "1"},
    )
    assert identity.backend_class == "patched-audit"
    assert identity.execution_route == "patched-audit"
    assert identity.primary_eligible is False
    assert identity.patch_manifest == str(manifest.resolve())
    assert identity.patch_manifest_sha256
