"""Source-derived verification claims for reproducible evidence packages.

The helpers in this module deliberately use a closed-world rule: a claim is
never considered successful merely because a packager was asked to write it.
Every ``pass`` or ``qualified`` claim must name immutable source files and the
SHA256 digest of each file.  Missing evidence remains ``not_run``/``unknown``;
an expected-source digest mismatch is a failure.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


VERIFICATION_SCHEMA = "torch_tm_flowpipe_source_derived_verification_v1"
DERIVATION_VERSION = 1
VALID_STATUSES = frozenset({"pass", "fail", "not_run", "unknown", "qualified"})


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _relative(path: Path, root: Path | None) -> str:
    path = Path(path)
    if root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


@dataclass(frozen=True)
class VerificationClaim:
    claim_id: str
    status: str
    source_paths: tuple[str, ...]
    source_sha256: tuple[str, ...]
    command: str | None
    exit_code: int | None
    started_at: str | None
    finished_at: str | None
    derived_by: str
    derivation_version: int
    scope: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("verification claim_id must be nonempty")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid verification status: {self.status}")
        if len(self.source_paths) != len(self.source_sha256):
            raise ValueError("verification source paths and SHA256 values disagree")
        if self.status in {"pass", "qualified"} and not self.source_paths:
            raise ValueError(
                f"verification claim {self.claim_id!r} cannot be {self.status} "
                "without source evidence"
            )
        if self.derivation_version <= 0:
            raise ValueError("derivation_version must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "status": self.status,
            "source_paths": list(self.source_paths),
            "source_sha256": list(self.source_sha256),
            "command": self.command,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "derived_by": self.derived_by,
            "derivation_version": self.derivation_version,
            "scope": self.scope,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VerificationClaim":
        return cls(
            claim_id=str(value.get("claim_id", "")),
            status=str(value.get("status", "")),
            source_paths=tuple(str(item) for item in value.get("source_paths", ())),
            source_sha256=tuple(
                str(item) for item in value.get("source_sha256", ())
            ),
            command=(
                None if value.get("command") is None else str(value["command"])
            ),
            exit_code=(
                None if value.get("exit_code") is None else int(value["exit_code"])
            ),
            started_at=(
                None if value.get("started_at") is None else str(value["started_at"])
            ),
            finished_at=(
                None if value.get("finished_at") is None else str(value["finished_at"])
            ),
            derived_by=str(value.get("derived_by", "")),
            derivation_version=int(value.get("derivation_version", 0)),
            scope=str(value.get("scope", "")),
            limitations=tuple(str(item) for item in value.get("limitations", ())),
        )


def unavailable_claim(
    claim_id: str,
    *,
    status: str = "not_run",
    scope: str,
    limitation: str,
    derived_by: str = "torch_tm_flowpipe.evidence_verification.unavailable_claim",
) -> VerificationClaim:
    if status not in {"not_run", "unknown", "fail"}:
        raise ValueError("an unavailable claim cannot be successful")
    return VerificationClaim(
        claim_id=claim_id,
        status=status,
        source_paths=(),
        source_sha256=(),
        command=None,
        exit_code=None,
        started_at=None,
        finished_at=None,
        derived_by=derived_by,
        derivation_version=DERIVATION_VERSION,
        scope=scope,
        limitations=(limitation,),
    )


def derive_command_claim(
    claim_id: str,
    source_dir: Path,
    *,
    scope: str,
    repository_root: Path | None = None,
    expected_source_sha256: Mapping[str, str] | None = None,
    evaluator: Callable[[str, str, int], tuple[str, Sequence[str]]] | None = None,
    derived_by: str = "torch_tm_flowpipe.evidence_verification.derive_command_claim",
) -> VerificationClaim:
    """Derive one claim from a runner-owned command-evidence directory.

    The directory contract is six UTF-8 files: ``command.txt``, ``stdout.log``,
    ``stderr.log``, ``exit_code.txt``, ``started_at.txt`` and
    ``finished_at.txt``.  A runner may additionally provide expected digests;
    any mismatch fails closed before command semantics are evaluated.
    """

    source_dir = Path(source_dir)
    names = (
        "command.txt",
        "stdout.log",
        "stderr.log",
        "exit_code.txt",
        "started_at.txt",
        "finished_at.txt",
    )
    paths = tuple(source_dir / name for name in names)
    if not source_dir.exists():
        return unavailable_claim(
            claim_id,
            status="not_run",
            scope=scope,
            limitation=f"command evidence directory is absent: {_relative(source_dir, repository_root)}",
            derived_by=derived_by,
        )
    missing = [path for path in paths if not path.is_file()]
    existing = tuple(path for path in paths if path.is_file())
    existing_names = tuple(_relative(path, repository_root) for path in existing)
    existing_hashes = tuple(sha256_file(path) for path in existing)
    if missing:
        return VerificationClaim(
            claim_id=claim_id,
            status="unknown",
            source_paths=existing_names,
            source_sha256=existing_hashes,
            command=None,
            exit_code=None,
            started_at=None,
            finished_at=None,
            derived_by=derived_by,
            derivation_version=DERIVATION_VERSION,
            scope=scope,
            limitations=(
                "incomplete command evidence: "
                + ", ".join(path.name for path in missing),
            ),
        )

    relative_paths = tuple(_relative(path, repository_root) for path in paths)
    hashes = tuple(sha256_file(path) for path in paths)
    if expected_source_sha256 is not None:
        mismatches: list[str] = []
        for path, relative, actual in zip(paths, relative_paths, hashes):
            expected = expected_source_sha256.get(path.name)
            if expected is None:
                expected = expected_source_sha256.get(relative)
            if expected is None or str(expected) != actual:
                mismatches.append(path.name)
        if mismatches:
            return VerificationClaim(
                claim_id=claim_id,
                status="fail",
                source_paths=relative_paths,
                source_sha256=hashes,
                command=paths[0].read_text(encoding="utf-8").rstrip("\n"),
                exit_code=None,
                started_at=paths[4].read_text(encoding="utf-8").strip() or None,
                finished_at=paths[5].read_text(encoding="utf-8").strip() or None,
                derived_by=derived_by,
                derivation_version=DERIVATION_VERSION,
                scope=scope,
                limitations=(
                    "source SHA256 mismatch: " + ", ".join(mismatches),
                ),
            )

    command = paths[0].read_text(encoding="utf-8").rstrip("\n")
    stdout = paths[1].read_text(encoding="utf-8")
    stderr = paths[2].read_text(encoding="utf-8")
    try:
        exit_code = int(paths[3].read_text(encoding="utf-8").strip())
    except ValueError:
        return VerificationClaim(
            claim_id=claim_id,
            status="fail",
            source_paths=relative_paths,
            source_sha256=hashes,
            command=command,
            exit_code=None,
            started_at=paths[4].read_text(encoding="utf-8").strip() or None,
            finished_at=paths[5].read_text(encoding="utf-8").strip() or None,
            derived_by=derived_by,
            derivation_version=DERIVATION_VERSION,
            scope=scope,
            limitations=("exit_code.txt is not an integer",),
        )
    status = "pass" if exit_code == 0 else "fail"
    limitations: Sequence[str] = ()
    if evaluator is not None:
        status, limitations = evaluator(stdout, stderr, exit_code)
        if status not in VALID_STATUSES:
            raise ValueError("claim evaluator returned an invalid status")
    return VerificationClaim(
        claim_id=claim_id,
        status=status,
        source_paths=relative_paths,
        source_sha256=hashes,
        command=command,
        exit_code=exit_code,
        started_at=paths[4].read_text(encoding="utf-8").strip() or None,
        finished_at=paths[5].read_text(encoding="utf-8").strip() or None,
        derived_by=derived_by,
        derivation_version=DERIVATION_VERSION,
        scope=scope,
        limitations=tuple(str(item) for item in limitations),
    )


def verification_document(claims: Iterable[VerificationClaim]) -> dict[str, Any]:
    values = list(claims)
    identifiers = [claim.claim_id for claim in values]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("verification claim identifiers must be unique")
    return {
        "schema": VERIFICATION_SCHEMA,
        "claims": [claim.as_dict() for claim in values],
    }


def validate_verification_document(
    value: Mapping[str, Any],
    *,
    source_root: Path | None = None,
) -> tuple[VerificationClaim, ...]:
    if value.get("schema") != VERIFICATION_SCHEMA:
        raise ValueError("verification document schema mismatch")
    raw_claims = value.get("claims")
    if not isinstance(raw_claims, list):
        raise ValueError("verification document claims must be a list")
    claims = tuple(VerificationClaim.from_mapping(item) for item in raw_claims)
    if len({claim.claim_id for claim in claims}) != len(claims):
        raise ValueError("verification claim identifiers must be unique")
    if source_root is not None:
        root = Path(source_root).resolve()
        for claim in claims:
            for relative, expected in zip(claim.source_paths, claim.source_sha256):
                path = Path(relative)
                if path.is_absolute():
                    raise ValueError("verification source path must be root-relative")
                resolved = (root / path).resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError("verification source path escapes source root")
                if not resolved.is_file():
                    raise ValueError(f"verification source is missing: {relative}")
                if sha256_file(resolved) != expected:
                    raise ValueError(f"verification source SHA256 mismatch: {relative}")
    return claims


def load_verification_document(
    path: Path,
    *,
    source_root: Path | None = None,
) -> tuple[VerificationClaim, ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("verification document must be a JSON object")
    return validate_verification_document(value, source_root=source_root)


def classify_private_path_matches(
    paths: Iterable[Path],
    *,
    scan_root: Path,
    private_prefix: str,
    provenance_only: Iterable[str] = (),
    sanitizer_fixtures: Iterable[str] = (),
) -> dict[str, Any]:
    """Classify literal private-path matches without calling presence a pass."""

    root = Path(scan_root).resolve()
    provenance = set(provenance_only)
    fixtures = set(sanitizer_fixtures)
    matches: list[dict[str, Any]] = []
    for path in sorted(Path(item) for item in paths):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = _relative(path, root)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if private_prefix not in line:
                continue
            category = (
                "provenance_only"
                if relative in provenance
                else "sanitizer_fixture"
                if relative in fixtures
                else "unclassified"
            )
            matches.append(
                {
                    "path": relative,
                    "line": line_number,
                    "category": category,
                }
            )
    remaining = [row for row in matches if row["category"] == "unclassified"]
    return {
        "private_path_present": bool(matches),
        "runtime_hidden_dependency": bool(remaining),
        "public_replay_dependency": bool(remaining),
        "ignored_categories": ["provenance_only", "sanitizer_fixture"],
        "remaining_matches": remaining,
        "matches": matches,
        "status": "qualified" if matches and not remaining else "pass" if not matches else "fail",
    }


__all__ = [
    "DERIVATION_VERSION",
    "VALID_STATUSES",
    "VERIFICATION_SCHEMA",
    "VerificationClaim",
    "classify_private_path_matches",
    "derive_command_claim",
    "load_verification_document",
    "sha256_file",
    "unavailable_claim",
    "validate_verification_document",
    "verification_document",
]
