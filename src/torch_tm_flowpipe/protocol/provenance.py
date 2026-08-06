from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .schema import IDENTITY_FIELDS


def canonical_config_identity(row: Mapping[str, Any]) -> str:
    payload = {field: row.get(field, "") for field in IDENTITY_FIELDS}
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prepare_output_directory(
    output: Path,
    *,
    resume: bool = False,
    expected_resume_manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Create a fresh directory or validate an exact manifest-bound resume."""
    output = output.resolve()
    if not output.exists():
        output.mkdir(parents=True)
        return output
    contents = list(output.iterdir())
    if not contents:
        return output
    if not resume:
        raise FileExistsError(f"refusing non-empty output directory: {output}")

    manifest_path = output / "RUN_MANIFEST.json"
    if expected_resume_manifest is None or not manifest_path.is_file():
        raise ValueError("resume requires an existing and expected RUN_MANIFEST.json")
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing != dict(expected_resume_manifest):
        raise ValueError("resume manifest does not match commit/config provenance")
    return output
