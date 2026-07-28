from __future__ import annotations

import sys
from pathlib import Path

REPAIR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = REPAIR_DIR.parents[1]

for path in (REPAIR_DIR, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
