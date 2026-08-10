"""Deterministic validation and checksums for a mainline result package."""
from __future__ import annotations

import hashlib
from pathlib import Path

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_required_package(root: Path) -> None:
    missing = [
        name
        for name in (*REQUIRED_MACHINE_FILES, *(f"figures/{name}" for name in REQUIRED_FIGURES))
        if not (root / name).is_file()
    ]
    if missing:
        raise ValueError(f"required result artifacts missing: {missing}")


def write_sha256sums(root: Path, destination: Path | None = None) -> int:
    destination = destination or root / "SHA256SUMS"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != destination.resolve()
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(files)


def verify_sha256sums(root: Path, sums_path: Path | None = None) -> tuple[bool, list[str]]:
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
        target = root / relative
        if not target.is_file():
            errors.append(f"{relative}: missing")
        elif sha256_file(target) != expected:
            errors.append(f"{relative}: digest mismatch")
    return not errors, errors

