#!/usr/bin/env python3
"""Build the machine tables, required figures, manifest, and checksums."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.artifact_package import (
    ALLOWED_NUMERICAL_SOUNDNESS_CLASSES,
    ALLOWED_NUMERICAL_SOUNDNESS_SCOPES,
    REQUIRED_FIGURES,
    REQUIRED_MACHINE_FILES,
    load_json_artifact,
    reject_nonfinite,
    reject_public_absolute_paths,
    sha256_file,
    validate_raw_evidence,
    validate_required_package,
    verify_artifact_manifests,
    verify_recovery_inventory,
    verify_sha256sums,
    write_sha256sums,
)

RUN_ID = "20260810T025910Z"


def _json(path: Path) -> Any:
    return load_json_artifact(path)


def _public_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("/srv/local/shengenli/", "<server-workspace>/")
    if isinstance(value, dict):
        return {key: _public_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_public_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_public_value(child) for child in value)
    return value


def _write_json(path: Path, value: Any) -> None:
    value = _public_value(value)
    reject_nonfinite(value, label=path.name)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty machine table: {path}")
    rows = _public_value(list(rows))
    reject_nonfinite(rows, label=path.name)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key, "")
                    for key in fields
                }
            )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _require_fields(value: Mapping[str, Any], fields: Sequence[str], *, label: str) -> None:
    missing = [field for field in fields if field not in value or value[field] is None]
    if missing:
        raise ValueError(f"{label}: incomplete required fields {missing}")


def _validated_horizon(summary: Mapping[str, Any]) -> float:
    completed = bool(summary.get("completed_requested_horizon"))
    value = summary.get("completed_horizon")
    if value is None and completed:
        value = summary.get("requested_horizon")
    if value is None or not math.isfinite(float(value)):
        raise ValueError("summary lacks a finite validated/completed horizon")
    return float(value)


def _candidate_decision(
    baseline_summary: Mapping[str, Any], candidate_summary: Mapping[str, Any]
) -> str:
    baseline_horizon = _validated_horizon(baseline_summary)
    candidate_horizon = _validated_horizon(candidate_summary)
    candidate_completed = bool(candidate_summary.get("completed_requested_horizon"))
    return (
        "CANDIDATE_PROMOTED"
        if candidate_completed or candidate_horizon > baseline_horizon
        else "CANDIDATE_REJECTED"
    )


def _eligibility_fields(
    *,
    mathematical_contract_known: bool,
    requested_horizon_completed: bool,
    certificate_semantics_passed: bool,
    finite_outputs: bool,
    numerical_soundness_class: str,
    numerical_soundness_scope: str,
    formal_claim_eligible: bool,
    performance_measurement_eligible: bool,
    cross_tool_ranking_eligible: bool,
) -> dict[str, Any]:
    if numerical_soundness_class not in ALLOWED_NUMERICAL_SOUNDNESS_CLASSES:
        raise ValueError(f"unsupported numerical soundness class: {numerical_soundness_class}")
    if numerical_soundness_scope not in ALLOWED_NUMERICAL_SOUNDNESS_SCOPES:
        raise ValueError(f"unsupported numerical soundness scope: {numerical_soundness_scope}")
    if formal_claim_eligible and not all(
        (
            mathematical_contract_known,
            certificate_semantics_passed,
            finite_outputs,
        )
    ):
        raise ValueError("formal claim eligibility contradicts prerequisite fields")
    if cross_tool_ranking_eligible and not all(
        (
            mathematical_contract_known,
            requested_horizon_completed,
            certificate_semantics_passed,
            finite_outputs,
            formal_claim_eligible,
            performance_measurement_eligible,
        )
    ):
        raise ValueError("cross-tool ranking eligibility contradicts prerequisites")
    return {
        "mathematical_contract_known": mathematical_contract_known,
        "requested_horizon_completed": requested_horizon_completed,
        "certificate_semantics_passed": certificate_semantics_passed,
        "finite_outputs": finite_outputs,
        "numerical_soundness_class": numerical_soundness_class,
        "numerical_soundness_scope": numerical_soundness_scope,
        "formal_claim_eligible": formal_claim_eligible,
        "performance_measurement_eligible": performance_measurement_eligible,
        "cross_tool_ranking_eligible": cross_tool_ranking_eligible,
    }


def _normalized_soundness_class(value: str) -> str:
    aliases = {
        "unsound/ineligible": "unsound/ineligible on a demonstrated counterexample",
    }
    normalized = aliases.get(value, value)
    if normalized not in ALLOWED_NUMERICAL_SOUNDNESS_CLASSES:
        raise ValueError(f"unsupported source soundness class: {value}")
    return normalized


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_worktree_status() -> str:
    canonical = f"outputs/mainline_realignment_20260810/{RUN_ID}"
    return _git(
        "status",
        "--short",
        "--untracked-files=all",
        "--",
        ".",
        f":(exclude){canonical}",
    )


def _package_tracking_status(run_root: Path) -> dict[str, Any]:
    canonical = ROOT / "outputs/mainline_realignment_20260810" / RUN_ID
    inspected = canonical if run_root.resolve() != canonical.resolve() else run_root
    tracked = set(_git("ls-files", "--", canonical.relative_to(ROOT).as_posix()).splitlines())
    files = sorted(path for path in inspected.rglob("*") if path.is_file())
    relative_files = {
        (canonical.relative_to(ROOT) / path.relative_to(inspected)).as_posix()
        for path in files
    }
    untracked = sorted(relative_files - tracked)
    return {
        "stored_file_count": len(relative_files),
        "tracked_file_count": len(relative_files & tracked),
        "all_stored_files_tracked": not untracked,
        "untracked_files": untracked,
    }


def _normalized_flowstar_stdout(path: Path) -> bytes:
    lines = [
        line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not line.startswith("time cost:")
    ]
    return ("\n".join(lines) + "\n").encode()


def _flowstar_rectangles(path: Path) -> list[tuple[float, float, float, float]]:
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("plot "):
            started = True
            continue
        if not started:
            continue
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        fields = line.split()
        if len(fields) == 2:
            try:
                current.append((float(fields[0]), float(fields[1])))
            except ValueError:
                pass
    if current:
        blocks.append(current)
    return [
        (min(x for x, _ in block), max(x for x, _ in block), min(y for _, y in block), max(y for _, y in block))
        for block in blocks
        if len(block) >= 4
    ]


def _fill_rectangles(
    ax: Any,
    rectangles: Iterable[tuple[float, float, float, float]],
    *,
    label: str,
    color: str,
    alpha: float,
) -> None:
    first = True
    for x0, x1, y0, y1 in rectangles:
        ax.fill_between(
            [x0, x1], [y0, y0], [y1, y1], color=color, alpha=alpha,
            linewidth=0, label=label if first else None,
        )
        first = False


def _torch_rectangles(rows: Sequence[Mapping[str, Any]], state: str, kind: str) -> list[tuple[float, float, float, float]]:
    out = []
    for row in rows:
        if row.get("status") != "accepted":
            continue
        values = (
            _float(row, "t_lo"), _float(row, "t_hi"),
            _float(row, f"{kind}_{state}_lo"), _float(row, f"{kind}_{state}_hi"),
        )
        if all(value is not None for value in values):
            out.append(tuple(values))
    return out


def _plot_required_figures(run_root: Path) -> None:
    figures = run_root / "figures"
    figures.mkdir(exist_ok=True)
    native = run_root / "01_native_baselines"
    baseline_dir = native / "torch_complete_o4_authoritative_t6p5"
    baseline_rows = _csv_rows(baseline_dir / "segments.csv")
    candidate_rows = _csv_rows(
        run_root / "04_generic_carry_candidate/final_da21a9e_t10_fresh/segments.csv"
    )
    flow_x = _flowstar_rectangles(native / "flowstar_stock_artifacts/vanderpol_t_x.plt")
    flow_y = _flowstar_rectangles(native / "flowstar_stock_artifacts/vanderpol_t_y.plt")

    for state, flow, filename in (
        ("x", flow_x, "flowstar_style_t_x_overlay.png"),
        ("y", flow_y, "flowstar_style_t_y_overlay.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        _fill_rectangles(ax, flow, label="stock Flow* full-segment tube", color="#1f77b4", alpha=0.15)
        _fill_rectangles(
            ax, _torch_rectangles(baseline_rows, state, "segment"),
            label="Torch complete-O4 segment tube (partial)", color="#ff7f0e", alpha=0.2,
        )
        _fill_rectangles(
            ax, _torch_rectangles(candidate_rows, state, "segment"),
            label="complete-carry candidate (partial)", color="#d62728", alpha=0.35,
        )
        ax.set(xlabel="physical time", ylabel=state, title=f"Van der Pol t-{state}: explicit tube semantics")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for index, ((_, _, x0, x1), (_, _, y0, y1)) in enumerate(zip(flow_x, flow_y)):
        if index % 3 == 0:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#1f77b4", alpha=0.08, edgecolor="none"))
    bx = _torch_rectangles(baseline_rows, "x", "segment")
    by = _torch_rectangles(baseline_rows, "y", "segment")
    for index, (xrow, yrow) in enumerate(zip(bx, by)):
        if index % 3 == 0:
            ax.add_patch(Rectangle((xrow[2], yrow[2]), xrow[3] - xrow[2], yrow[3] - yrow[2], facecolor="#ff7f0e", alpha=0.10, edgecolor="none"))
    ax.plot([], [], color="#1f77b4", linewidth=6, alpha=0.25, label="stock Flow* tube")
    ax.plot([], [], color="#ff7f0e", linewidth=6, alpha=0.3, label="Torch complete-O4 prefix")
    ax.set(xlabel="x", ylabel="y", title="Phase tube overlay (failed prefixes remain explicit)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "phase_tube_overlay.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    times = [_float(row, "t_hi") for row in baseline_rows if row.get("status") == "accepted"]
    for state, color in (("x", "#1f77b4"), ("y", "#ff7f0e")):
        endpoints = [_float(row, f"endpoint_{state}_width") for row in baseline_rows if row.get("status") == "accepted"]
        tubes = [_float(row, f"segment_{state}_width") for row in baseline_rows if row.get("status") == "accepted"]
        axes[0].plot(times, endpoints, color=color, label=f"endpoint {state}")
        axes[1].plot(times, tubes, color=color, label=f"tube {state}")
    axes[0].set(ylabel="endpoint width", title="Torch complete-O4 validated prefix widths")
    axes[1].set(xlabel="physical time", ylabel="segment-tube width")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "endpoint_tube_width_vs_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    times2, poly, remainder = [], [], []
    for row in baseline_rows:
        if row.get("status") != "accepted":
            continue
        t = _float(row, "t_hi")
        p = _float(row, "carry_composed_poly_range_width_sum") or _float(row, "carry_composed_poly_range_width")
        r = _float(row, "carry_output_remainder_width_sum") or _float(row, "carry_output_remainder_width")
        if t is not None and p is not None and r is not None:
            times2.append(t); poly.append(p); remainder.append(r)
    ax.plot(times2, poly, label="retained polynomial range width", color="#2ca02c")
    ax.plot(times2, remainder, label="ordinary remainder width", color="#d62728")
    ax.set(xlabel="physical time", ylabel="width sum", title="Polynomial-range versus remainder contribution")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "polynomial_range_vs_remainder.png", dpi=180)
    plt.close(fig)

    horizon_rows = _csv_rows(run_root / "full_horizon.csv")
    lane_labels = {
        "stock_flowstar_native": "stock Flow*",
        "stock_diffreach_native": "stock DiffReach",
        "torch_fixed_dr7_b64": "Torch fixed B64",
        "torch_complete_o4_frozen_natural": "Torch complete O4",
        "torch_complete_o4_complete_carry": "complete carry",
    }
    horizons: dict[str, float] = {}
    for row in horizon_rows:
        lane = row.get("lane", "")
        validated = _float(row, "validated_horizon")
        if lane in lane_labels and validated is not None:
            label = lane_labels[lane]
            horizons[label] = max(validated, horizons.get(label, -math.inf))
    if set(horizons) != set(lane_labels.values()):
        raise ValueError(f"validated-horizon figure lacks lanes: {horizons}")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#999999", "#9467bd", "#2ca02c", "#ff7f0e", "#d62728"]
    ax.bar(list(horizons), list(horizons.values()), color=colors)
    requested = [
        value
        for row in horizon_rows
        if (value := _float(row, "requested_horizon")) is not None
    ]
    ax.axhline(max(requested), color="black", linewidth=0.8, linestyle="--")
    ax.set(ylabel="highest validated horizon", title="Completion is not cross-contract ranking")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "validated_horizon.png", dpi=180)
    plt.close(fig)

    scaling = _collect_scaling(run_root)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for lane in ("fixed_support", "complete_carry"):
        for device in ("cpu", "cuda_v100"):
            rows = [row for row in scaling if row["lane"] == lane and row["device_group"] == device]
            rows.sort(key=lambda row: row["batch"])
            ax.plot([row["batch"] for row in rows], [row["warm_min_s"] for row in rows], marker="o", label=f"{lane} {device} warm")
            ax.plot([row["batch"] for row in rows], [row["cold_s"] for row in rows], marker="x", linestyle="--", alpha=0.65, label=f"{lane} {device} cold")
    ax.set(xscale="log", yscale="log", xlabel="batch (actual independent partitions)", ylabel="seconds", title="Cold and warm runtime by batch")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "runtime_vs_batch.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    eligible = [
        row
        for row in scaling
        if row["lane"] == "fixed_support"
        and row["performance_measurement_eligible"]
        and row["requested_horizon_completed"]
    ]
    for device, marker in (("cpu", "o"), ("cuda_v100", "s")):
        rows = [row for row in eligible if row["device_group"] == device]
        throughput = [row["batch"] * 10 / row["warm_min_s"] for row in rows]
        widths = [row["endpoint_width_sum"] for row in rows]
        ax.scatter(throughput, widths, marker=marker, label=f"fixed-support {device}")
        for row, x, y in zip(rows, throughput, widths):
            ax.annotate(f"B{row['batch']}", (x, y), fontsize=7)
    ax.set(xlabel="eligible short-run batch-steps/s", ylabel="aggregate endpoint width", title="Eligible precision/throughput rows only")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "eligible_precision_throughput.png", dpi=180)
    plt.close(fig)


def _collect_scaling(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in ("fixed_support", "complete_carry"):
        for device_group in ("cpu", "cuda_v100"):
            for batch in (1, 8, 64, 256, 512):
                path = run_root / f"05_batch_scaling/{lane}/{device_group}_b{batch}/summary.json"
                if not path.is_file():
                    raise ValueError(f"referenced scaling summary absent: {path}")
                value = _json(path)
                _require_fields(
                    value,
                    ("dtype_device", "execution_kind"),
                    label=path.relative_to(run_root).as_posix(),
                )
                if value.get("status") is None and value.get("completion_status") is None:
                    raise ValueError(f"{path}: missing status/completion_status")
                timing = value.get("cold_warm_core_process_runtime", value.get("timing", {}))
                widths = value.get("endpoint_tube_polynomial_remainder_widths", {})
                endpoint_width = widths.get("endpoint_width", [])
                completed = bool(
                    value.get(
                        "eligible_full_horizon",
                        value.get("accepted_all_batches", False),
                    )
                )
                soundness_class = _normalized_soundness_class(
                    value.get("soundness_classification", "unknown")
                )
                rows.append(
                    {
                        "track": "F",
                        "lane": lane,
                        "device_group": device_group,
                        "device": value["dtype_device"]["device"],
                        "device_name": value["dtype_device"]["device_name"],
                        "batch": batch,
                        "actual_independent_inputs": value.get("actual_independent_inputs", True),
                        "scope": value["execution_kind"],
                        "requested_horizon": value.get("requested_horizon", value.get("h")),
                        "validated_horizon": value.get("validated_horizon", value.get("h") if value.get("accepted_all_batches") else 0),
                        "completion": value.get("completion_status", value.get("status")),
                        "cold_s": timing.get("cold_s", timing.get("cold_first_call_s")),
                        "warm_min_s": timing.get("warm_min_s"),
                        "warm_median_s": timing.get("warm_median_s"),
                        "warm_max_s": max(timing.get("warm_s", [])) if timing.get("warm_s") else timing.get("warm_max_s"),
                        "peak_memory_bytes": value.get("peak_memory_bytes"),
                        "endpoint_width_sum": sum(endpoint_width) if endpoint_width else sum(
                            hi - lo for lo, hi in value.get("endpoint_hull", [])
                        ),
                        "soundness_classification": value.get("soundness_classification"),
                        "host_synchronizations": value.get("host_synchronizations", "aggregate inclusion gates and trace extraction"),
                        "source_sha": value.get("source_sha"),
                        "artifact": path.relative_to(run_root).as_posix(),
                        **_eligibility_fields(
                            mathematical_contract_known=True,
                            requested_horizon_completed=completed,
                            certificate_semantics_passed=completed,
                            finite_outputs=True,
                            numerical_soundness_class=soundness_class,
                            numerical_soundness_scope=(
                                "multi-step lane"
                                if lane == "fixed_support"
                                else "primitive"
                            ),
                            formal_claim_eligible=False,
                            performance_measurement_eligible=True,
                            cross_tool_ranking_eligible=False,
                        ),
                    }
                )
    return rows


def _horizon_row(
    lane: str,
    summary: Mapping[str, Any],
    requested: float,
    artifact: str,
    *,
    numerical_soundness_class: str,
) -> dict[str, Any]:
    _require_fields(
        summary,
        ("completed_requested_horizon", "requested_horizon", "accepted_steps", "runtime_s"),
        label=artifact,
    )
    recorded_requested = float(summary["requested_horizon"])
    if recorded_requested != float(requested):
        raise ValueError(
            f"{artifact}: requested horizon {recorded_requested} != indexed {requested}"
        )
    completed = bool(summary.get("completed_requested_horizon"))
    return {
        "track": "F",
        "lane": lane,
        "requested_horizon": recorded_requested,
        "validated_horizon": _validated_horizon(summary),
        "completion_status": "completed" if completed else "PARTIAL_HORIZON",
        "certificate_status": "passed" if completed else "failed_validation",
        "first_failure_reason": summary.get("failure_type", "") if not completed else "",
        "accepted_steps": summary.get("accepted_steps"),
        "rejected_attempts": summary.get("rejected_attempts"),
        "runtime_s": summary.get("runtime_s"),
        "raw_endpoint": summary.get("raw_endpoint"),
        "last_segment_tube": summary.get("last_segment"),
        "fallback_count": summary.get("fallback_count"),
        "repair_used": summary.get("endpoint_repair_used"),
        "artifact": artifact,
        **_eligibility_fields(
            mathematical_contract_known=True,
            requested_horizon_completed=completed,
            certificate_semantics_passed=completed,
            finite_outputs=True,
            numerical_soundness_class=_normalized_soundness_class(
                numerical_soundness_class
            ),
            numerical_soundness_scope="multi-step lane",
            formal_claim_eligible=completed and numerical_soundness_class
            in {
                "formally outward by construction",
                "safeguarded outward under declared IEEE/backend assumptions",
            },
            performance_measurement_eligible=True,
            cross_tool_ranking_eligible=False,
        ),
    }


def build(run_root: Path) -> None:
    # Provenance identifies the evidence-capture event, not whichever later
    # commit happens to regenerate the derived tables.  Preserve it on rebuild
    # so a clean copy remains byte reproducible after the branch advances.
    manifest_path = run_root / "manifest.json"
    frozen_manifest = _json(manifest_path) if manifest_path.is_file() else {}
    validate_raw_evidence(run_root)
    recovery_valid, recovery_errors = verify_recovery_inventory(run_root)
    if not recovery_valid:
        raise ValueError(f"recovered evidence failures: {recovery_errors[:20]}")
    manifests_valid, manifest_errors = verify_artifact_manifests(run_root)
    if not manifests_valid:
        raise ValueError(f"raw artifact manifest failures: {manifest_errors[:20]}")
    required_fixed_dir = run_root / "02_torch_diffreach_equivalence"
    required_fixed_dir.mkdir(exist_ok=True)
    fixed_sources = [
        "02_fixed_support/fixed_support_equivalence.json",
        "02_fixed_support/fraction_replay_cpu.json",
        "02_fixed_support/fraction_replay_cuda.json",
        "02_fixed_support/cpu_b64_t10/summary.json",
        "02_fixed_support/cuda_b64_t0p1/summary.json",
    ]
    _write_json(
        required_fixed_dir / "evidence_index.json",
        {
            "schema": "torch_tm_fixed_support_required_path_index_v1",
            "canonical_raw_directory": "02_fixed_support",
            "files": [
                {"path": relative, "sha256": sha256_file(run_root / relative)}
                for relative in fixed_sources
            ],
            "note": "This required-name directory is an index; raw files remain unmodified in their canonical directory.",
        },
    )
    native_source = run_root / "01_native_baselines/native_baselines.json"
    native = _json(native_source)
    _require_fields(native, ("lanes",), label=native_source.as_posix())
    lanes = native["lanes"]
    _require_fields(
        lanes,
        ("flowstar_stock", "diffreach_stock", "torch_authoritative_complete_o4"),
        label="native lanes",
    )
    native_flow = lanes["flowstar_stock"]
    native_flow["soundness_classification"] = _normalized_soundness_class(
        native_flow["soundness_classification"]
    )
    native_flow["track"] = "N"
    native_flow.update(
        _eligibility_fields(
            mathematical_contract_known=True,
            requested_horizon_completed=native_flow["status"] == "completed",
            certificate_semantics_passed=bool(native_flow["reported_property"]),
            finite_outputs=True,
            numerical_soundness_class=_normalized_soundness_class(
                native_flow["soundness_classification"]
            ),
            numerical_soundness_scope="native build",
            formal_claim_eligible=False,
            performance_measurement_eligible=True,
            cross_tool_ranking_eligible=False,
        )
    )
    native_diff = lanes["diffreach_stock"]
    native_diff["soundness_classification"] = _normalized_soundness_class(
        native_diff["soundness_classification"]
    )
    native_diff["track"] = "N"
    native_diff.update(
        _eligibility_fields(
            mathematical_contract_known=True,
            requested_horizon_completed=native_diff["status"] == "completed",
            certificate_semantics_passed=False,
            finite_outputs=True,
            numerical_soundness_class=_normalized_soundness_class(
                native_diff["soundness_classification"]
            ),
            numerical_soundness_scope="native build",
            formal_claim_eligible=False,
            performance_measurement_eligible=True,
            cross_tool_ranking_eligible=False,
        )
    )
    native_complete = lanes["torch_authoritative_complete_o4"]
    native_complete["soundness_classification"] = _normalized_soundness_class(
        native_complete["soundness_classification"]
    )
    native_complete["track"] = "N"
    native_complete.update(
        _eligibility_fields(
            mathematical_contract_known=True,
            requested_horizon_completed=False,
            certificate_semantics_passed=False,
            finite_outputs=True,
            numerical_soundness_class=_normalized_soundness_class(
                native_complete["soundness_classification"]
            ),
            numerical_soundness_scope="multi-step lane",
            formal_claim_eligible=False,
            performance_measurement_eligible=True,
            cross_tool_ranking_eligible=False,
        )
    )
    native["derived_root_copy"] = {"source": native_source.relative_to(run_root).as_posix(), "source_sha256": sha256_file(native_source)}
    _write_json(run_root / "native_baselines.json", native)

    matched_yaml = ROOT / "benchmarks/vdp_three_lane_contract_20260810.yaml"
    matched = yaml.safe_load(matched_yaml.read_text(encoding="utf-8"))
    _write_json(
        run_root / "matched_contract.json",
        {
            "schema": "torch_tm_matched_contract_result_v1",
            "track": "M",
            "contract_source": str(matched_yaml.relative_to(ROOT)),
            "contract_sha256": sha256_file(matched_yaml),
            "contract": matched,
            "expressibility": {
                "flowstar_stock": "native official VDP expressible; tubes only; formal stock numerical gate fails",
                "diffreach_stock": "native official VDP expressible with B64/fixed h; stock tube and later masks unavailable",
                "torch_complete": (
                    "expressible; partial at "
                    f"{lanes['torch_authoritative_complete_o4']['highest_validated_horizon']}"
                ),
                "torch_fixed": "expressible in framework; native-like fixed support, B64, h=.01",
            },
            "ranked": False,
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=False,
                certificate_semantics_passed=False,
                finite_outputs=True,
                numerical_soundness_class="unknown",
                numerical_soundness_scope="fixed workload",
                formal_claim_eligible=False,
                performance_measurement_eligible=False,
                cross_tool_ranking_eligible=False,
            ),
        },
    )

    fixed_eq = _json(run_root / "02_fixed_support/fixed_support_equivalence.json")
    common_summary = _json(run_root / "03_flowstar_causal_divergence/common_basis_final/summary.json")
    one_step = _json(run_root / "04_generic_carry_candidate/one_step_grid_final/summary.json")
    fixed_qualified = not bool(fixed_eq.get("implementation_mismatch", True))
    common_qualified = bool(common_summary.get("native_torch_replay_bit_exact"))
    carry_qualified = bool(one_step.get("all_decisions_match")) and bool(
        one_step.get("all_endpoint_coefficients_match")
    )
    _write_json(
        run_root / "operator_equivalence.json",
        {
            "schema": "torch_tm_operator_equivalence_v1",
            "track": "M",
            "fixed_support": fixed_eq,
            "common_basis": common_summary,
            "complete_carry_one_step": one_step,
            "decisions": {
                "fixed_support": "qualified" if fixed_qualified else "mismatch",
                "common_basis": "qualified" if common_qualified else "mismatch",
                "complete_carry": "zero one-step mismatch" if carry_qualified else "mismatch",
            },
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=False,
                certificate_semantics_passed=(
                    fixed_qualified and common_qualified and carry_qualified
                ),
                finite_outputs=True,
                numerical_soundness_class="empirically sampled only",
                numerical_soundness_scope="one step",
                formal_claim_eligible=False,
                performance_measurement_eligible=False,
                cross_tool_ranking_eligible=False,
            ),
        },
    )

    observer_root = run_root / "03_flowstar_causal_divergence/flowstar_observer"
    logged = observer_root / "final_logged"
    unlogged = observer_root / "final_unlogged"
    observer_equivalence = {
        "schema": "flowstar_causal_observer_equivalence_v1",
        "stock_source_sha": lanes["flowstar_stock"]["source_sha"],
        "logged_observer_sha256": sha256_file(logged / "observer.jsonl"),
        "official_outputs": {
            name: {
                "logged_sha256": sha256_file(logged / name),
                "unlogged_sha256": sha256_file(unlogged / name),
                "byte_identical": (logged / name).read_bytes() == (unlogged / name).read_bytes(),
            }
            for name in ("vanderpol_t_x.plt", "vanderpol_t_y.plt")
        },
        "normalized_stdout_sha256": {
            "logged": hashlib.sha256(_normalized_flowstar_stdout(logged / "stdout.log")).hexdigest(),
            "unlogged": hashlib.sha256(_normalized_flowstar_stdout(unlogged / "stdout.log")).hexdigest(),
        },
        "stderr_byte_identical": (logged / "stderr.log").read_bytes() == (unlogged / "stderr.log").read_bytes(),
        "accepted_schedule_rows": len(
            _flowstar_rectangles(logged / "vanderpol_t_x.plt")
        ),
        "completion_match": (
            (logged / "vanderpol_t_x.plt").read_bytes()
            == (unlogged / "vanderpol_t_x.plt").read_bytes()
        ),
        "exit_status_match": (
            _json(logged / "command.json").get("exit_status")
            == _json(unlogged / "command.json").get("exit_status")
        ) if (logged / "command.json").is_file() and (unlogged / "command.json").is_file() else "UNAVAILABLE",
    }
    observer_equivalence["normalized_stdout_identical"] = (
        observer_equivalence["normalized_stdout_sha256"]["logged"]
        == observer_equivalence["normalized_stdout_sha256"]["unlogged"]
    )
    _write_json(
        run_root / "03_flowstar_causal_divergence/observer_equivalence.json",
        observer_equivalence,
    )

    baseline_dirs = {
        0.1: "torch_complete_o4_t0p1", 0.5: "torch_complete_o4_t0p5", 1.0: "torch_complete_o4_t1p0",
        4.0: "torch_complete_o4_t4p0", 6.0: "torch_complete_o4_t6p0", 6.5: "torch_complete_o4_t6p5",
        7.5: "torch_complete_o4_t7p5", 10.0: "torch_complete_o4_t10p0",
    }
    candidate_dirs = {
        0.1: "final_da21a9e_t0p1", 0.5: "final_da21a9e_t0p5", 1.0: "final_da21a9e_t1p0",
        4.0: "final_da21a9e_t4", 6.0: "final_da21a9e_t6", 6.5: "final_da21a9e_t6p5",
        7.5: "final_da21a9e_t7p5", 10.0: "final_da21a9e_t10_fresh",
    }
    complete_soundness = lanes["torch_authoritative_complete_o4"][
        "soundness_classification"
    ]
    horizon_rows = []
    for horizon, directory in baseline_dirs.items():
        relative = f"01_native_baselines/{directory}/summary.json"
        horizon_rows.append(
            _horizon_row(
                "torch_complete_o4_frozen_natural",
                _json(run_root / relative),
                horizon,
                relative,
                numerical_soundness_class=complete_soundness,
            )
        )
    for horizon, directory in candidate_dirs.items():
        relative = f"04_generic_carry_candidate/{directory}/summary.json"
        horizon_rows.append(
            _horizon_row(
                "torch_complete_o4_complete_carry",
                _json(run_root / relative),
                horizon,
                relative,
                numerical_soundness_class=complete_soundness,
            )
        )
    fixed_ladder_paths = {
        0.1: "05_batch_scaling/fixed_support/cpu_b64/summary.json",
        0.5: "07_fixed_support_ladder/t0p5_gpu0/summary.json",
        1.0: "07_fixed_support_ladder/t1p0_gpu2/summary.json",
        4.0: "07_fixed_support_ladder/t4p0_gpu3/summary.json",
        6.0: "07_fixed_support_ladder/t6p0_gpu0/summary.json",
        6.5: "07_fixed_support_ladder/t6p5_gpu2/summary.json",
        7.5: "07_fixed_support_ladder/t7p5_gpu3/summary.json",
    }
    for horizon, relative in fixed_ladder_paths.items():
        path = run_root / relative
        if not path.is_file():
            raise ValueError(f"referenced fixed-support ladder summary absent: {path}")
        summary = _json(path)
        fixed_completed = summary["completion_status"] == "completed"
        fixed_certificate_passed = summary["certificate_status"] == "passed"
        horizon_rows.append(
            {
                "track": "F",
                "lane": "torch_fixed_dr7_b64",
                "requested_horizon": horizon,
                "validated_horizon": summary["validated_horizon"],
                "completion_status": summary["completion_status"],
                "certificate_status": summary["certificate_status"],
                "first_failure_reason": summary["first_failure_time_reason"],
                "accepted_steps": summary["accepted_rejected_steps"]["accepted"],
                "rejected_attempts": summary["accepted_rejected_steps"]["rejected_initial_inclusion"],
                "runtime_s": summary["cold_warm_core_process_runtime"]["cold_s"],
                "raw_endpoint": summary["endpoint_tube_polynomial_remainder_widths"]["raw_endpoint"],
                "last_segment_tube": summary["endpoint_tube_polynomial_remainder_widths"]["last_full_segment_tube"],
                "fallback_count": 0,
                "repair_used": False,
                "artifact": relative,
                **_eligibility_fields(
                    mathematical_contract_known=True,
                    requested_horizon_completed=fixed_completed,
                    certificate_semantics_passed=fixed_certificate_passed,
                    finite_outputs=True,
                    numerical_soundness_class=_normalized_soundness_class(
                        summary["soundness_classification"]
                    ),
                    numerical_soundness_scope="multi-step lane",
                    formal_claim_eligible=False,
                    performance_measurement_eligible=True,
                    cross_tool_ranking_eligible=False,
                ),
            }
        )
    _write_csv(run_root / "short_horizon.csv", [row for row in horizon_rows if row["requested_horizon"] <= 1.0])

    full_rows = [row for row in horizon_rows if row["requested_horizon"] >= 4.0]
    requested_horizon = float(matched["system"]["requested_horizon"])
    flow_lane = lanes["flowstar_stock"]
    diff_lane = lanes["diffreach_stock"]
    full_rows.extend(
        [
            {
                "track": "N",
                "lane": "stock_flowstar_native",
                "requested_horizon": requested_horizon,
                "validated_horizon": flow_lane["validated_horizon"],
                "completion_status": flow_lane["status"],
                "certificate_status": (
                    "reported_safe_but_formal_gate_failed"
                    if not flow_lane["primary_formal_comparison_eligible"]
                    else "reported_safe"
                ),
                "accepted_steps": flow_lane["accepted_segments"],
                "runtime_s": flow_lane["core_runtime_s"],
                "soundness": flow_lane["soundness_classification"],
                "artifact": "01_native_baselines/native_baselines.json",
                **_eligibility_fields(
                    mathematical_contract_known=True,
                    requested_horizon_completed=(
                        float(flow_lane["validated_horizon"]) >= requested_horizon
                    ),
                    certificate_semantics_passed=bool(flow_lane["reported_property"]),
                    finite_outputs=True,
                    numerical_soundness_class=_normalized_soundness_class(
                        flow_lane["soundness_classification"]
                    ),
                    numerical_soundness_scope="native build",
                    formal_claim_eligible=False,
                    performance_measurement_eligible=True,
                    cross_tool_ranking_eligible=False,
                ),
            },
            {
                "track": "N",
                "lane": "stock_diffreach_native",
                "requested_horizon": requested_horizon,
                "validated_horizon": diff_lane["validated_horizon"],
                "completion_status": diff_lane["status"],
                "certificate_status": (
                    "all_returned_initial_masks_pass"
                    if diff_lane["returned_initial_inclusion"]["passed"]
                    == diff_lane["returned_initial_inclusion"]["total"]
                    else "initial_mask_failure"
                ),
                "accepted_steps": diff_lane["step_policy"]["steps"],
                "runtime_s": diff_lane["timing_s"]["verification_after_jit"],
                "soundness": diff_lane["soundness_classification"],
                "artifact": "01_native_baselines/native_baselines.json",
                **_eligibility_fields(
                    mathematical_contract_known=True,
                    requested_horizon_completed=(
                        float(diff_lane["validated_horizon"]) >= requested_horizon
                    ),
                    certificate_semantics_passed=(
                        "masks_not_returned"
                        not in diff_lane["later_round_failure_semantics"]
                    ),
                    finite_outputs=True,
                    numerical_soundness_class=_normalized_soundness_class(
                        diff_lane["soundness_classification"]
                    ),
                    numerical_soundness_scope="native build",
                    formal_claim_eligible=False,
                    performance_measurement_eligible=True,
                    cross_tool_ranking_eligible=False,
                ),
            },
        ]
    )
    fixed_t10_path = run_root / "02_fixed_support/cpu_b64_t10/summary.json"
    fixed_t10 = _json(fixed_t10_path)
    fixed_t10_completed = fixed_t10["completion_status"] == "completed"
    fixed_t10_certificate = fixed_t10["certificate_status"] == "passed"
    full_rows.append(
        {
            "track": "F",
            "lane": "torch_fixed_dr7_b64",
            "requested_horizon": fixed_t10["requested_horizon"],
            "validated_horizon": fixed_t10["validated_horizon"],
            "completion_status": fixed_t10["completion_status"],
            "certificate_status": fixed_t10["certificate_status"],
            "accepted_steps": fixed_t10["accepted_rejected_steps"]["accepted"],
            "runtime_s": fixed_t10["cold_warm_core_process_runtime"]["cold_s"],
            "soundness": fixed_t10["soundness_classification"],
            "artifact": fixed_t10_path.relative_to(run_root).as_posix(),
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=fixed_t10_completed,
                certificate_semantics_passed=fixed_t10_certificate,
                finite_outputs=True,
                numerical_soundness_class=_normalized_soundness_class(
                    fixed_t10["soundness_classification"]
                ),
                numerical_soundness_scope="multi-step lane",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        }
    )
    final_baseline = run_root / "06_final_baseline_ladder/torch_complete_o4_adaptive_t10_fresh/summary.json"
    if not final_baseline.is_file():
        raise ValueError(f"referenced final baseline absent: {final_baseline}")
    full_rows.append(
        _horizon_row(
            "torch_complete_o4_adaptive_final",
            _json(final_baseline),
            10.0,
            final_baseline.relative_to(run_root).as_posix(),
            numerical_soundness_class=complete_soundness,
        )
    )
    _write_csv(run_root / "full_horizon.csv", full_rows)

    causal = _json(run_root / "03_flowstar_causal_divergence/common_basis_final/counterfactuals.json")
    candidate_failure = _json(run_root / "04_generic_carry_candidate/final_da21a9e_t10_fresh/summary.json")
    baseline_failure = _json(
        run_root / "06_final_baseline_ladder/torch_complete_o4_adaptive_t10_fresh/summary.json"
    )
    candidate_decision = _candidate_decision(baseline_failure, candidate_failure)
    _write_json(
        run_root / "failure_attribution.json",
        {"schema": "torch_tm_failure_attribution_v1", "observer_equivalence": observer_equivalence, "first_native_divergence": causal, "candidate_failure": candidate_failure, "candidate_decision": candidate_decision},
    )

    scaling = _collect_scaling(run_root)
    _write_csv(run_root / "batch_scaling.csv", scaling)
    timing_rows = [
        {
            "lane": row["lane"], "device": row["device"], "batch": row["batch"],
            "boundary": row["scope"], "cold_s": row["cold_s"], "warm_min_s": row["warm_min_s"],
            "warm_median_s": row["warm_median_s"], "warm_max_s": row["warm_max_s"],
            "process_startup_import_s": "included only in fresh process, not isolated here",
            "compile_jit_s": "none for eager Torch lane", "cuda_synchronized": row["device_group"] == "cuda_v100",
            "performance_measurement_eligible": row["performance_measurement_eligible"],
            "cross_tool_ranking_eligible": row["cross_tool_ranking_eligible"],
        }
        for row in scaling
    ]
    timing_rows.extend(
        [
            {"lane": "stock_flowstar_native", "device": "cpu", "batch": 1, "boundary": "native reported certification core", "cold_s": "UNAVAILABLE", "warm_min_s": lanes["flowstar_stock"]["core_runtime_s"], "process_startup_import_s": "UNAVAILABLE", "compile_jit_s": "not applicable", "performance_measurement_eligible": True, "cross_tool_ranking_eligible": False},
            {"lane": "stock_diffreach_native", "device": "V100", "batch": 64, "boundary": "native verification", "cold_s": lanes["diffreach_stock"]["timing_s"]["verification_warmup"], "warm_min_s": lanes["diffreach_stock"]["timing_s"]["verification_after_jit"], "compile_jit_s": "included in warmup", "performance_measurement_eligible": True, "cross_tool_ranking_eligible": False},
        ]
    )
    fresh_path = run_root / "08_fresh_process_timing/fresh_process_timing.json"
    if not fresh_path.is_file():
        raise ValueError(f"referenced fresh-process timing absent: {fresh_path}")
    for row in _json(fresh_path)["rows"]:
        timing_rows.append(
            {
                    "lane": row["lane"],
                    "device": "cpu",
                    "batch": 64 if row["lane"].startswith("fixed") else 1,
                    "boundary": "rotated fresh process T=0.1",
                    "fresh_repetition": row["repetition"],
                    "rotation_position": row["rotation_position"],
                    "fresh_process_wall_s": row["fresh_process_wall_s"],
                    "certification_core_s": row["certification_core_s"],
                    "startup_import_config_serialization_composite_s": row[
                        "startup_import_config_serialization_composite_s"
                    ],
                    "compile_jit_s": row["compile_jit_s"],
                    "performance_measurement_eligible": True,
                    "cross_tool_ranking_eligible": False,
            }
        )
    _write_csv(run_root / "timing.csv", timing_rows)

    replay = fixed_eq["exact_rational_replay"]
    complete_lane = lanes["torch_authoritative_complete_o4"]
    soundness = [
        {
            "lane": "stock_flowstar_native",
            "basis": flow_lane["qualification"]["gate"],
            "formal_claim_scope": "none",
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=True,
                finite_outputs=True,
                numerical_soundness_class=_normalized_soundness_class(
                    flow_lane["soundness_classification"]
                ),
                numerical_soundness_scope="native build",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "lane": "stock_diffreach_native",
            "basis": diff_lane["dtype_device"],
            "formal_claim_scope": "none",
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=False,
                finite_outputs=True,
                numerical_soundness_class=_normalized_soundness_class(
                    diff_lane["soundness_classification"]
                ),
                numerical_soundness_scope="native build",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "lane": "torch_fixed_ordinary_cpu_cuda",
            "basis": "ordinary_binary64_directly_qualified="
            f"{replay['ordinary_binary64_directly_qualified']}",
            "formal_claim_scope": "none",
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=bool(fixed_eq["full_t10"]["completed"]),
                certificate_semantics_passed=True,
                finite_outputs=True,
                numerical_soundness_class=_normalized_soundness_class(
                    replay["ordinary_lane_classification"]
                ),
                numerical_soundness_scope="multi-step lane",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "lane": "torch_fixed_2ulp_companion_one_step",
            "basis": replay["arithmetic"],
            "formal_claim_scope": replay["scope"],
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=bool(replay["replay_envelope_qualified"]),
                finite_outputs=True,
                numerical_soundness_class=_normalized_soundness_class(
                    replay["replay_envelope_classification"]
                ),
                numerical_soundness_scope="one step",
                formal_claim_eligible=bool(replay["replay_envelope_qualified"]),
                performance_measurement_eligible=False,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "lane": "torch_complete_o4_baseline",
            "basis": complete_lane["range_policy"],
            "formal_claim_scope": (
                f"validated prefix {complete_lane['highest_validated_horizon']} only"
            ),
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=False,
                certificate_semantics_passed=True,
                finite_outputs=True,
                numerical_soundness_class=_normalized_soundness_class(
                    complete_lane["soundness_classification"]
                ),
                numerical_soundness_scope="multi-step lane",
                formal_claim_eligible=True,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "lane": "complete_polynomial_carry_primitive",
            "basis": "endpoint coefficient preservation from one-step grid",
            "formal_claim_scope": "primitive only; surrounding dense arithmetic separate",
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=bool(
                    one_step["all_candidate_carries_preserve_endpoint_coefficients"]
                ),
                finite_outputs=True,
                numerical_soundness_class=(
                "formally outward by construction"
                if one_step["all_candidate_carries_preserve_endpoint_coefficients"]
                else "unknown"
                ),
                numerical_soundness_scope="primitive",
                formal_claim_eligible=bool(
                    one_step["all_candidate_carries_preserve_endpoint_coefficients"]
                ),
                performance_measurement_eligible=False,
                cross_tool_ranking_eligible=False,
            ),
        },
    ]
    _write_csv(run_root / "soundness_matrix.csv", soundness)
    causal_stage = causal["causal_attribution"]["earliest_decision-changing_stage"]
    claims = [
        {
            "claim": "native entrypoints reproduced",
            "track": "N",
            "status": "valid" if all(
                lane["status"] == "completed" for lane in (flow_lane, diff_lane)
            ) else "invalid",
            "scope": "own contracts",
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=False,
                finite_outputs=True,
                numerical_soundness_class="unknown",
                numerical_soundness_scope="fixed workload",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "claim": "Torch fixed explicit-f64 operators reproduce DiffReach",
            "track": "M",
            "status": "valid" if fixed_qualified else "invalid",
            "scope": "frozen fixture and declared full lane",
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=fixed_qualified,
                finite_outputs=True,
                numerical_soundness_class="empirically sampled only",
                numerical_soundness_scope="fixed workload",
                formal_claim_eligible=False,
                performance_measurement_eligible=False,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "claim": "first native split is raw candidate remainder",
            "track": "M",
            "status": "valid" if "raw Picard remainder" in causal_stage else "invalid",
            "scope": causal_stage,
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=True,
                finite_outputs=True,
                numerical_soundness_class="empirically sampled only",
                numerical_soundness_scope="one step",
                formal_claim_eligible=False,
                performance_measurement_eligible=False,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "claim": "complete polynomial carry improves horizon",
            "track": "F",
            "status": "valid" if candidate_decision == "CANDIDATE_PROMOTED" else "invalid",
            "scope": f"candidate validated horizon {_validated_horizon(candidate_failure)}",
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=False,
                certificate_semantics_passed=False,
                finite_outputs=True,
                numerical_soundness_class="formally outward by construction",
                numerical_soundness_scope="multi-step lane",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "claim": "Torch globally faster or tighter than Flow*",
            "track": "N",
            "status": "valid" if matched.get("ranked", False) else "invalid",
            "scope": "contracts and eligibility differ",
            **_eligibility_fields(
                mathematical_contract_known=False,
                requested_horizon_completed=False,
                certificate_semantics_passed=False,
                finite_outputs=True,
                numerical_soundness_class="unknown",
                numerical_soundness_scope="fixed workload",
                formal_claim_eligible=False,
                performance_measurement_eligible=False,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "claim": "ordinary CUDA is universally outward",
            "track": "F",
            "status": "valid" if replay["ordinary_binary64_directly_qualified"] else "invalid",
            "scope": replay["scope"],
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=True,
                finite_outputs=True,
                numerical_soundness_class="empirically sampled only",
                numerical_soundness_scope="fixed workload",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "claim": "stock Flow* is primary formal comparator",
            "track": "N",
            "status": "valid" if flow_lane["primary_formal_comparison_eligible"] else "blocked",
            "scope": flow_lane["qualification"]["gate"],
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=True,
                certificate_semantics_passed=True,
                finite_outputs=True,
                numerical_soundness_class=_normalized_soundness_class(
                    flow_lane["soundness_classification"]
                ),
                numerical_soundness_scope="native build",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
        {
            "claim": "multi-step complete lane scales to B512",
            "track": "F",
            "status": "blocked" if any(
                "not a multi-step certificate" in str(row.get("scope_limitation", ""))
                for row in [
                    _json(run_root / "05_batch_scaling/complete_carry/cpu_b512/summary.json")
                ]
            ) else "valid",
            "scope": _json(
                run_root / "05_batch_scaling/complete_carry/cpu_b512/summary.json"
            )["scope_limitation"],
            **_eligibility_fields(
                mathematical_contract_known=True,
                requested_horizon_completed=False,
                certificate_semantics_passed=True,
                finite_outputs=True,
                numerical_soundness_class="empirically sampled only",
                numerical_soundness_scope="primitive",
                formal_claim_eligible=False,
                performance_measurement_eligible=True,
                cross_tool_ranking_eligible=False,
            ),
        },
    ]
    _write_csv(run_root / "claim_registry.csv", claims)

    _plot_required_figures(run_root)
    start_state = _json(run_root / "00_provenance/start_state.json")
    external = start_state["external_repositories"]
    manifest = {
        "schema": "torch_tm_mainline_realignment_manifest_v1",
        "run_id": RUN_ID,
        "branch": frozen_manifest.get("branch", _git("branch", "--show-current")),
        "source_sha": frozen_manifest.get("source_sha", _git("rev-parse", "HEAD")),
        "source_worktree_status_excluding_package": frozen_manifest.get(
            "source_worktree_status_excluding_package", _source_worktree_status()
        ),
        "package_tracking_status": _package_tracking_status(run_root),
        "external_sources": {
            "flowstar": external["flowstar"]["sha"],
            "diffreach": external["diffreach"]["sha"],
            "xiangru_local": external["xiangru"]["local_sha"],
            "xiangru_server_resident_remote": external["xiangru"]["server_resident_remote_sha"],
        },
        "required_machine_files": list(REQUIRED_MACHINE_FILES),
        "required_figures": list(REQUIRED_FIGURES),
        "formal_evidence_scope": "all recursive files listed by SHA256SUMS; debug directories retained but not promoted",
        "candidate_decision": candidate_decision,
    }
    _write_json(run_root / "manifest.json", manifest)
    validate_required_package(run_root)
    reject_public_absolute_paths(run_root)
    try:
        path_prefix = run_root.relative_to(ROOT).as_posix()
    except ValueError:
        if run_root.name != RUN_ID:
            raise ValueError(
                "external clean-copy run root must retain the canonical run id"
            )
        path_prefix = f"outputs/mainline_realignment_20260810/{RUN_ID}"
    count = write_sha256sums(run_root, path_prefix=path_prefix)
    valid, errors = verify_sha256sums(run_root, path_prefix=path_prefix)
    if not valid:
        raise RuntimeError(errors)
    print(json.dumps({"run_root": str(run_root), "hashed_files": count, "checksums_valid": valid}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=ROOT / "outputs/mainline_realignment_20260810" / RUN_ID,
    )
    args = parser.parse_args()
    build(args.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
