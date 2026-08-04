from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping


REQUIRED_CROSS_TOOL_GATES = (
    "stock_backend_identity",
    "official_parser_generated_stock_field_parity",
    "endpoint_segment_tube_exporter_semantics",
    "raw_tightened_separation",
    "order_basis_contract",
    "runtime_boundary_parity",
    "completion_validation_fail_closed",
    "patched_rows_excluded_from_primary",
)


@dataclass(frozen=True)
class GateManifestDecision:
    passed: bool
    errors: tuple[str, ...]
    pending: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cross_tool_gate_manifest(
    manifest: Mapping[str, Any], *, repo_root: Path
) -> GateManifestDecision:
    """Validate gate evidence, never just a truthy YAML status.

    A verified gate is accepted only when its machine-readable evidence and
    human-readable report exist, the recorded checksum matches, and a concrete
    automated-test node id and applicability list are present.
    """
    errors: list[str] = []
    pending: list[str] = []
    gates = manifest.get("gates")
    if not isinstance(gates, Mapping):
        return GateManifestDecision(False, ("gates must be a mapping",), ())
    names = tuple(gates)
    missing = sorted(set(REQUIRED_CROSS_TOOL_GATES) - set(names))
    extra = sorted(set(names) - set(REQUIRED_CROSS_TOOL_GATES))
    if missing:
        errors.append("missing gates: " + ", ".join(missing))
    if extra:
        errors.append("unknown gates: " + ", ".join(extra))

    for name in REQUIRED_CROSS_TOOL_GATES:
        record = gates.get(name)
        if not isinstance(record, Mapping):
            continue
        verified = record.get("verified")
        if type(verified) is not bool:
            errors.append(f"{name}: verified must be a boolean")
            continue
        if verified is False:
            pending.append(name)
            if not str(record.get("blocker", "")).strip():
                errors.append(f"{name}: pending gate requires blocker")
            continue
        for key in ("evidence", "evidence_sha256", "report", "test"):
            if not str(record.get(key, "")).strip():
                errors.append(f"{name}: verified gate missing {key}")
        applies_to = record.get("applies_to")
        if not isinstance(applies_to, list) or not applies_to:
            errors.append(f"{name}: verified gate missing applies_to list")
        evidence_text = str(record.get("evidence", ""))
        report_text = str(record.get("report", ""))
        if evidence_text:
            evidence = (repo_root / evidence_text).resolve()
            if not evidence.is_relative_to(repo_root.resolve()):
                errors.append(f"{name}: evidence escapes repository")
            elif not evidence.is_file():
                errors.append(f"{name}: evidence does not exist")
            elif sha256_file(evidence) != record.get("evidence_sha256"):
                errors.append(f"{name}: evidence checksum mismatch")
        if report_text:
            report = (repo_root / report_text).resolve()
            if not report.is_relative_to(repo_root.resolve()):
                errors.append(f"{name}: report escapes repository")
            elif not report.is_file():
                errors.append(f"{name}: report does not exist")
    return GateManifestDecision(not errors and not pending, tuple(errors), tuple(pending))

