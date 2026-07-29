from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO_ROOT = EXPERIMENT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (EXPERIMENT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def pytest_addoption(parser: pytest.Parser) -> None:
    for name in ("torch", "diffreach", "flowstar"):
        parser.addoption(f"--{name}-segment", default="")
