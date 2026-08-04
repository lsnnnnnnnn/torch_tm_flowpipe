from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest

from torch_tm_flowpipe.protocol.backend_identity import (
    inspect_primary_flowstar_backend,
)


@pytest.mark.flowstar
@pytest.mark.integration
def test_configured_flowstar_backend_is_primary_safe() -> None:
    configured = os.environ.get("FLOWSTAR_ROOT")
    if not configured:
        pytest.skip("FLOWSTAR_ROOT is not configured for the optional backend test")
    root = Path(configured).expanduser()
    if not (root / "flowstar-toolbox" / "libflowstar.a").is_file():
        pytest.skip("FLOWSTAR_ROOT has no built flowstar-toolbox/libflowstar.a")

    identity = inspect_primary_flowstar_backend(root, environment=os.environ)

    assert identity.backend_class in {
        "unmodified-stock",
        "stock-plus-gcc15-compat",
    }
    assert identity.primary_eligible is True
    assert identity.repository_sha
    assert identity.library_sha256


@pytest.mark.diffreach
@pytest.mark.integration
def test_configured_diffreach_checkout_and_environment_import() -> None:
    configured = os.environ.get("DIFFREACH_ROOT")
    python = os.environ.get("DIFFREACH_PYTHON")
    if not configured:
        pytest.skip("DIFFREACH_ROOT is not configured for the optional backend test")
    if not python:
        pytest.skip("DIFFREACH_PYTHON is not configured for the optional backend test")
    root = Path(configured).expanduser().resolve()
    interpreter = Path(python).expanduser().resolve()
    if not (root / ".git").exists():
        pytest.skip("DIFFREACH_ROOT is not a Git checkout")
    if not interpreter.is_file():
        pytest.skip("DIFFREACH_PYTHON does not exist")

    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    process = subprocess.run(
        [
            str(interpreter),
            "-c",
            (
                "import pathlib,sys; "
                f"sys.path.insert(0, {str(root)!r}); "
                "import jax, src.taylor_model as taylor_model; "
                "print(jax.__version__, pathlib.Path(taylor_model.__file__).resolve())"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert sha
    assert process.returncode == 0, process.stderr
    assert str(root) in process.stdout
