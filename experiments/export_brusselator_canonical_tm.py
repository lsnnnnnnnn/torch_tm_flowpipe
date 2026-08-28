#!/usr/bin/env python3
"""Export lossless C4 Brusselator checkpoints for same-object range replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import FlowstarNormalFlowpipeState, load_terminal_checkpoint  # noqa: E402
from torch_tm_flowpipe.brusselator_canonical_exchange import (  # noqa: E402
    FLOWSTAR_VARIABLE_ORDER,
    ORDER,
    SCHEMA,
    TAU_INDEX,
    VARIABLE_ORDER,
    build_exchange_records,
    write_records,
)


FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
INITIAL_DECIMAL = (("1.48", "1.52"), ("2.98", "3.02"))
SOURCE_PATHS = (
    "src/torch_tm_flowpipe/accepted_boundary_sr.py",
    "src/torch_tm_flowpipe/batched_dense_tm.py",
    "src/torch_tm_flowpipe/brusselator_canonical_exchange.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/polynomial.py",
    "src/torch_tm_flowpipe/symbolic_remainder.py",
    "experiments/export_brusselator_canonical_tm.py",
    "experiments/flowstar_probe/flowstar_brusselator_canonical_range.cpp",
    "experiments/flowstar_probe/flowstar_canonical_range_access.patch",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required canonical exchange source is missing: {path}")
        hashes[relative.replace("/", "__").replace(".", "_")] = _sha256(path)
    return hashes


def _checkpoint_map(baseline: Path, summary: dict[str, Any]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for row in summary.get("accepted_checkpoint_records", []):
        step = int(row["accepted_step"])
        path = baseline / row["relative_directory"]
        manifest = json.loads((path / "terminal_state_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("full_checkpoint_sha256") != row.get("full_checkpoint_sha256"):
            raise ValueError(f"checkpoint manifest/index mismatch at step {step}")
        result[step] = path
    return result


def _selected_steps(summary: dict[str, Any]) -> list[int]:
    accepted = int(summary["accepted_steps"])
    requested = {1, 2, 3, 100, 200, 300}
    material = summary.get("first_persistent_material_stock_bound_difference_step")
    if material is not None:
        requested.add(int(material))
    requested.update(range(max(1, accepted - 4), accepted + 1))
    return sorted(step for step in requested if step <= accepted)


def _schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Brusselator live-range canonical exchange object",
        "schema_id": SCHEMA,
        "physical_encoding": "ordered newline-terminated UTF-8 key=value records",
        "numeric_encoding": "canonical finite Python/C hexadecimal binary64",
        "state_dimension": 2,
        "order": ORDER,
        "torch_variable_order": list(VARIABLE_ORDER),
        "flowstar_harness_variable_order": list(FLOWSTAR_VARIABLE_ORDER),
        "tau_index": TAU_INDEX,
        "required_payloads": [
            "complete monomial exponent and coefficient tables",
            "ordinary remainder endpoints",
            "SR propagated history and current owner",
            "full SR queue history and generation metadata",
            "pre/post normalization center and scale",
            "domains and local-time power tables",
            "cutoff and source hashes",
        ],
        "live_labels": [
            "boundary_normalization",
            "right_map_construction",
            "composition_truncation",
            "cutoff_payment",
            "Picard_validation",
            "next_step_initialization",
        ],
        "reporting_labels": ["reporting_endpoint", "reporting_tube"],
        "unknown_field_policy": "reject in Python tests; Flow* harness consumes required prefixes and hashes full input",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    baseline = args.baseline_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_path = baseline / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("validation_mode") != "flowstar_raw_remainder_compat_refined":
        raise ValueError("canonical exporter requires the frozen C4 baseline")
    if not summary.get("certificate_checks_passed"):
        raise ValueError("canonical exporter refuses an uncertified C4 baseline")
    if not summary.get("c5_checkpoint_capture_enabled"):
        raise ValueError("C4 baseline did not enable canonical checkpoint capture")
    checkpoints = _checkpoint_map(baseline, summary)
    selected = _selected_steps(summary)
    missing = [
        step
        for step in selected
        if step not in checkpoints or (step > 1 and step - 1 not in checkpoints)
    ]
    if missing:
        raise ValueError(f"C4 baseline lacks exact pre/post checkpoints for: {missing}")
    sources = _source_hashes()
    initial = FlowstarNormalFlowpipeState.from_exact_decimal_box(INITIAL_DECIMAL, ORDER)
    objects: list[dict[str, Any]] = []
    for step in selected:
        post_checkpoint = load_terminal_checkpoint(
            checkpoints[step], expected_order=ORDER, expected_dtype="float64"
        )
        pre_state = (
            initial
            if step == 1
            else load_terminal_checkpoint(
                checkpoints[step - 1], expected_order=ORDER, expected_dtype="float64"
            ).normal_state
        )
        checkpoint_sha = str(post_checkpoint.manifest["full_checkpoint_sha256"])
        built = build_exchange_records(
            pre_state=pre_state,
            post_state=post_checkpoint.normal_state,
            accepted_step=step,
            checkpoint_sha256=checkpoint_sha,
            torch_solver_commit=str(summary["commit"]),
            flowstar_commit=FLOWSTAR_SHA,
            source_hashes=sources,
        )
        filename = f"accepted_step_{step:04d}.canonical"
        path = output / filename
        sha = write_records(path, built.records)
        objects.append(
            {
                "accepted_step": step,
                "filename": filename,
                "sha256": sha,
                "checkpoint_sha256": checkpoint_sha,
                "composition_branch": built.prepared.composition_branch,
                "queue_before_size": len(built.prepared.queue_before.J),
                "queue_after_sha256": built.queue_after_sha256,
                "reconstructed_post_right_bitwise": True,
                "reconstructed_post_queue_bitwise": True,
            }
        )
    index = {
        "schema": "torch_tm_flowpipe.brusselator_live_range_exchange_index/1",
        "exchange_schema": SCHEMA,
        "baseline_summary_sha256": _sha256(summary_path),
        "baseline_solver_commit": summary["commit"],
        "flowstar_commit": FLOWSTAR_SHA,
        "selected_steps": selected,
        "objects": objects,
        "source_hashes": sources,
    }
    _write_json(output / "index.json", index)
    _write_json(output / "CANONICAL_OBJECT_SCHEMA.json", _schema())
    return index


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
