"""Audit-only pytest plugin mapping Huan's absolute config path to local data."""

from __future__ import annotations

import os
from pathlib import Path


def pytest_collection_modifyitems(items):
    root = Path(os.environ["HUAN_CONFIG_ROOT"])
    for item in items:
        module = getattr(item, "module", None)
        if module is not None and hasattr(module, "CONFIGS"):
            current = getattr(module, "CONFIGS")
            if str(current) == "/home/huan/projects/CROWN-Reach/src/configs":
                module.CONFIGS = root
