"""Deterministic validation and checksums for a mainline result package."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, TextIO

REQUIRED_MACHINE_FILES = (
    "manifest.json",
    "native_baselines.json",
    "matched_contract.json",
    "operator_equivalence.json",
    "short_horizon.csv",
    "full_horizon.csv",
    "failure_attribution.json",
    "batch_scaling.csv",
    "timing.csv",
    "soundness_matrix.csv",
    "claim_registry.csv",
)

REQUIRED_FIGURES = (
    "flowstar_style_t_x_overlay.png",
    "flowstar_style_t_y_overlay.png",
    "phase_tube_overlay.png",
    "endpoint_tube_width_vs_time.png",
    "polynomial_range_vs_remainder.png",
    "validated_horizon.png",
    "runtime_vs_batch.png",
    "eligible_precision_throughput.png",
)

REQUIRED_RAW_DIRECTORIES = (
    "01_native_baselines",
    "02_fixed_support",
    "03_flowstar_causal_divergence",
    "04_generic_carry_candidate",
    "05_batch_scaling",
    "06_final_baseline_ladder",
    "07_fixed_support_ladder",
    "08_fresh_process_timing",
)

REQUIRED_RAW_FILES = (
    "00_provenance/evidence_recovery.json",
    "01_native_baselines/native_baselines.json",
    "02_fixed_support/fixed_support_equivalence.json",
    "03_flowstar_causal_divergence/common_basis_final/summary.json",
    "03_flowstar_causal_divergence/common_basis_final/counterfactuals.json",
    "04_generic_carry_candidate/one_step_grid_final/summary.json",
    "04_generic_carry_candidate/final_da21a9e_t10_fresh/summary.json",
    "05_batch_scaling/fixed_support/cpu_b64/summary.json",
    "06_final_baseline_ladder/torch_complete_o4_adaptive_t10_fresh/summary.json",
    "07_fixed_support_ladder/t7p5_gpu3/summary.json",
    "08_fresh_process_timing/fresh_process_timing.json",
)

CANONICAL_RUN_RELATIVE = Path(
    "outputs/mainline_realignment_20260810/20260810T025910Z"
)
ALLOWED_NUMERICAL_SOUNDNESS_CLASSES = frozenset(
    {
        "formally outward by construction",
        "safeguarded outward under declared IEEE/backend assumptions",
        "independently outward replayed for exact benchmark workload",
        "empirically sampled only",
        "unsound/ineligible on a demonstrated counterexample",
        "unknown",
    }
)
ALLOWED_NUMERICAL_SOUNDNESS_SCOPES = frozenset(
    {"primitive", "one step", "fixed workload", "multi-step lane", "native build"}
)
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_BACKTICK_CANONICAL_PATH = re.compile(
    r"`(outputs/mainline_realignment_20260810/20260810T025910Z/[^`]+)`"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stored_artifact_path(path: Path) -> Path:
    """Resolve an artifact, accepting a deterministic ``.gz`` stored form."""
    if path.is_file():
        return path
    compressed = path.with_name(path.name + ".gz")
    if compressed.is_file():
        return compressed
    return path


def open_text_artifact(path: Path) -> TextIO:
    stored = stored_artifact_path(path)
    if stored.suffix == ".gz":
        return gzip.open(stored, "rt", encoding="utf-8")
    return stored.open("r", encoding="utf-8")


def load_json_artifact(path: Path) -> Any:
    with open_text_artifact(path) as handle:
        value = json.load(handle)
    reject_nonfinite(value, label=str(path))
    return value


def reject_nonfinite(value: Any, *, label: str = "value") -> None:
    """Reject NaN/Inf recursively while allowing explicit UNAVAILABLE strings."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label}: nonfinite numeric value")
    if isinstance(value, dict):
        for key, child in value.items():
            reject_nonfinite(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_nonfinite(child, label=f"{label}[{index}]")


def validate_required_package(root: Path) -> None:
    missing = [
        name
        for name in (*REQUIRED_MACHINE_FILES, *(f"figures/{name}" for name in REQUIRED_FIGURES))
        if not (root / name).is_file()
    ]
    if missing:
        raise ValueError(f"required result artifacts missing: {missing}")


def validate_raw_evidence(root: Path) -> None:
    missing_directories = [
        name for name in REQUIRED_RAW_DIRECTORIES if not (root / name).is_dir()
    ]
    missing_files = [
        name
        for name in REQUIRED_RAW_FILES
        if not stored_artifact_path(root / name).is_file()
    ]
    if missing_directories or missing_files:
        raise ValueError(
            "required raw evidence missing: "
            f"directories={missing_directories}, files={missing_files}"
        )


def verify_recovery_inventory(root: Path) -> tuple[bool, list[str]]:
    inventory_path = root / "00_provenance/evidence_recovery.json"
    errors: list[str] = []
    try:
        inventory = load_json_artifact(inventory_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, [f"{inventory_path}: {exc}"]
    rows = inventory.get("files") if isinstance(inventory, dict) else None
    if not isinstance(rows, list) or len(rows) != inventory.get("source_file_count"):
        return False, ["recovery inventory file count mismatch"]
    if str(inventory.get("source_label", "")).startswith("/"):
        errors.append("recovery inventory exposes an absolute source path")
    for row in rows:
        required = {
            "source_relative_path",
            "source_size",
            "source_sha256",
            "stored_relative_path",
            "stored_size",
            "stored_sha256",
            "storage",
        }
        if not isinstance(row, dict) or not required <= row.keys():
            errors.append("recovery inventory has an incomplete row")
            continue
        stored = root / str(row["stored_relative_path"])
        if not stored.is_file():
            errors.append(f"{row['stored_relative_path']}: recovered file missing")
            continue
        regenerated = row["storage"] == "derived-regenerated"
        if not regenerated and stored.stat().st_size != int(row["stored_size"]):
            errors.append(f"{row['stored_relative_path']}: stored size mismatch")
        if not regenerated and sha256_file(stored) != row["stored_sha256"]:
            errors.append(f"{row['stored_relative_path']}: stored digest mismatch")
        if row["storage"] == "identity":
            if (
                row["stored_size"] != row["source_size"]
                or row["stored_sha256"] != row["source_sha256"]
            ):
                errors.append(f"{row['stored_relative_path']}: identity record differs")
        elif row["storage"] == "gzip-mtime-zero":
            digest = hashlib.sha256()
            size = 0
            try:
                with gzip.open(stored, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
            except (OSError, EOFError) as exc:
                errors.append(f"{row['stored_relative_path']}: gzip failure {exc}")
                continue
            if size != int(row["source_size"]):
                errors.append(f"{row['stored_relative_path']}: source size mismatch")
            if digest.hexdigest() != row["source_sha256"]:
                errors.append(f"{row['stored_relative_path']}: source digest mismatch")
        elif row["storage"] != "derived-regenerated":
            errors.append(f"{row['stored_relative_path']}: unknown storage mode")
    return not errors, errors


def verify_artifact_manifests(root: Path) -> tuple[bool, list[str]]:
    """Verify every nested artifact manifest and local SHA256SUMS file."""
    errors: list[str] = []
    for manifest_path in sorted(root.rglob("artifact_manifest.json")):
        try:
            manifest = load_json_artifact(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{manifest_path.relative_to(root)}: {exc}")
            continue
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(files, list) or not files:
            errors.append(f"{manifest_path.relative_to(root)}: missing files list")
            continue
        seen: set[str] = set()
        for row in files:
            if not isinstance(row, dict) or not {"path", "bytes", "sha256"} <= row.keys():
                errors.append(f"{manifest_path.relative_to(root)}: incomplete file row")
                continue
            relative = str(row["path"])
            if relative in seen:
                errors.append(f"{manifest_path.relative_to(root)}: duplicate {relative}")
                continue
            seen.add(relative)
            target = stored_artifact_path(manifest_path.parent / relative)
            if not target.is_file():
                errors.append(f"{target.relative_to(root)}: missing")
            elif target.name.endswith(".gz"):
                errors.append(
                    f"{target.relative_to(root)}: nested manifests require identity storage"
                )
            else:
                if target.stat().st_size != int(row["bytes"]):
                    errors.append(f"{target.relative_to(root)}: byte count mismatch")
                if sha256_file(target) != row["sha256"]:
                    errors.append(f"{target.relative_to(root)}: digest mismatch")

    root_sums = (root / "SHA256SUMS").resolve()
    for sums_path in sorted(root.rglob("SHA256SUMS")):
        if sums_path.resolve() == root_sums:
            continue
        for line_number, line in enumerate(
            sums_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line:
                continue
            try:
                expected, relative = line.split("  ", 1)
            except ValueError:
                errors.append(f"{sums_path.relative_to(root)}:{line_number}: malformed")
                continue
            target = sums_path.parent / relative
            if not target.is_file():
                errors.append(f"{target.relative_to(root)}: missing")
            elif sha256_file(target) != expected:
                errors.append(f"{target.relative_to(root)}: digest mismatch")
    return not errors, errors


def iter_canonical_report_references(
    repository_root: Path,
    markdown_paths: Iterable[Path],
) -> Iterable[tuple[Path, str]]:
    canonical = CANONICAL_RUN_RELATIVE.as_posix()
    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8")
        references: set[str] = set()
        for match in _MARKDOWN_LINK.finditer(text):
            raw = match.group(1).split("#", 1)[0]
            if canonical in raw:
                if raw.startswith("../"):
                    resolved = (markdown_path.parent / raw).resolve()
                    try:
                        raw = resolved.relative_to(repository_root.resolve()).as_posix()
                    except ValueError:
                        pass
                references.add(raw.rstrip("/"))
        references.update(
            match.group(1).rstrip("/")
            for match in _BACKTICK_CANONICAL_PATH.finditer(text)
        )
        for reference in sorted(references):
            yield markdown_path, reference


def validate_report_artifact_references(
    repository_root: Path,
    markdown_paths: Iterable[Path],
    *,
    require_tracked: bool,
) -> None:
    errors: list[str] = []
    tracked_paths: set[str] | None = None
    if require_tracked:
        tracked_paths = set(
            subprocess.run(
                ["git", "ls-files"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        )
    for markdown_path, reference in iter_canonical_report_references(
        repository_root, markdown_paths
    ):
        target = repository_root / reference
        if not target.exists():
            errors.append(f"{markdown_path.relative_to(repository_root)}: missing {reference}")
            continue
        if require_tracked:
            candidates = [target] if target.is_file() else [
                path for path in target.rglob("*") if path.is_file()
            ]
            for candidate in candidates:
                relative = candidate.relative_to(repository_root).as_posix()
                if tracked_paths is not None and relative not in tracked_paths:
                    errors.append(
                        f"{markdown_path.relative_to(repository_root)}: untracked {relative}"
                    )
    if errors:
        raise ValueError(f"report artifact reference failures: {errors[:20]}")


def reject_public_absolute_paths(root: Path) -> None:
    errors: list[str] = []
    for name in REQUIRED_MACHINE_FILES:
        path = root / name
        if path.is_file() and b"/srv/local/" in path.read_bytes():
            errors.append(name)
    if errors:
        raise ValueError(f"absolute server paths in public machine files: {errors}")


def write_sha256sums(
    root: Path,
    destination: Path | None = None,
    *,
    path_prefix: str | None = None,
) -> int:
    destination = destination or root / "SHA256SUMS"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != destination.resolve()
    )
    prefix = path_prefix.strip("/") + "/" if path_prefix else ""
    lines = [
        f"{sha256_file(path)}  {prefix}{path.relative_to(root).as_posix()}"
        for path in files
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def verify_sha256sums(
    root: Path,
    sums_path: Path | None = None,
    *,
    path_prefix: str | None = None,
) -> tuple[bool, list[str]]:
    sums_path = sums_path or root / "SHA256SUMS"
    errors: list[str] = []
    for line_number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: malformed")
            continue
        prefix = path_prefix.strip("/") + "/" if path_prefix else ""
        if prefix:
            if not relative.startswith(prefix):
                errors.append(f"{relative}: expected prefix {prefix}")
                continue
            relative = relative[len(prefix) :]
        target = root / relative
        if not target.is_file():
            errors.append(f"{relative}: missing")
        elif sha256_file(target) != expected:
            errors.append(f"{relative}: digest mismatch")
    return not errors, errors
