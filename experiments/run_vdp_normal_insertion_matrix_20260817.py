#!/usr/bin/env python3
"""Fresh legacy/candidate VDP matrix for the normal-insertion root-cause fix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments/run_vdp_dense_backend.py"
FLOWSTAR_LEDGER = (
    ROOT
    / "outputs/vdp_t1_t3_width_causal_source_ledger_20260814/20260814T120000Z"
    / "04_causal_runs/checkpoint_widths.csv"
)
MODES = {
    "legacy": "normalized_insertion",
    "candidate": "normalized_insertion_dependency_preserving",
}
FIXED_HORIZONS = (1.0, 3.0, 6.32)
CHANNELS = ("endpoint_x", "endpoint_y", "segment_x", "segment_y")


def _label(value: float) -> str:
    return f"T{format(value, 'g').replace('.', 'p')}"


def _command(output: Path, mode: str, schedule: str, horizon: float, wall_cap_s: float, device: str) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--output-dir",
        str(output),
        "--tm-backend",
        "dense",
        "--device",
        device,
        "--initialization-contract",
        "exact_decimal_contract",
        "--horizon",
        format(horizon, "g"),
        "--trace-flush-every",
        "0",
        "--wall-cap-s",
        format(wall_cap_s, "g"),
        "--reset-mode",
        MODES[mode],
        "--dense-range-method",
        "adaptive_subdivision",
        "--dense-range-trigger",
        "proactive_depth1_on_named_contexts",
        "--dense-range-max-depth",
        "1",
        "--dense-range-max-leaves",
        "4",
        "--dense-range-split-vars",
        "0,1",
        "--dense-range-contexts",
        "polynomial_truncation",
    ]
    if schedule == "fixed":
        command.extend(("--fixed-step", "0.01"))
    return command


def _run_request(
    root: Path,
    mode: str,
    schedule: str,
    horizon: float,
    wall_cap_s: float,
    device: str = "cpu",
) -> dict[str, Any]:
    output = root / schedule / device / mode / _label(horizon)
    output.mkdir(parents=True, exist_ok=False)
    command = _command(output, mode, schedule, horizon, wall_cap_s, device)
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (output / "matrix_stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output / "matrix_stderr.txt").write_text(completed.stderr, encoding="utf-8")
    summary_path = output / "summary.json"
    if completed.returncode not in (0, 1) or not summary_path.is_file():
        raise RuntimeError(
            f"missing scientific summary for {schedule}/{device}/{mode}/{horizon}: "
            f"returncode={completed.returncode}; stderr={completed.stderr[-2000:]}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    row = {
        "mode": mode,
        "reset_mode": MODES[mode],
        "schedule": schedule,
        "device": device,
        "requested_horizon": horizon,
        "status": summary["status"],
        "completed_horizon": summary["completed_horizon"],
        "completed_requested_horizon": summary["completed_requested_horizon"],
        "accepted_steps": summary["accepted_steps"],
        "rejected_attempts": summary["rejected_attempts"],
        "runtime_s": summary["runtime_s"],
        "matrix_elapsed_s": time.perf_counter() - started,
        "peak_rss_bytes": summary["peak_rss_bytes"],
        "endpoint_x": (summary.get("raw_endpoint") or {}).get("x_width"),
        "endpoint_y": (summary.get("raw_endpoint") or {}).get("y_width"),
        "segment_x": (summary.get("last_segment") or {}).get("x_width"),
        "segment_y": (summary.get("last_segment") or {}).get("y_width"),
        "message": summary["message"],
        "relative_output": str(output.relative_to(root)),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def _flowstar_widths() -> dict[float, dict[str, float]]:
    channel_map = {
        "endpoint_x": "endpoint_x",
        "endpoint_y": "endpoint_y",
        "segment_tube_x": "segment_x",
        "segment_tube_y": "segment_y",
    }
    selected: dict[float, dict[str, float]] = {value: {} for value in FIXED_HORIZONS}
    with FLOWSTAR_LEDGER.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time_value = float(row["time"])
            if time_value not in selected or row["channel"] not in channel_map:
                continue
            selected[time_value][channel_map[row["channel"]]] = float(row["flowstar_width"])
    if any(set(value) != set(CHANNELS) for value in selected.values()):
        raise ValueError("authoritative Flow* ledger is missing a required channel")
    return selected


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {
        (row["schedule"], row["device"], row["mode"], float(row["requested_horizon"])): row
        for row in rows
    }
    flowstar = _flowstar_widths()
    fixed: dict[str, Any] = {}
    for horizon in FIXED_HORIZONS:
        legacy = by_key[("fixed", "cpu", "legacy", horizon)]
        candidate = by_key[("fixed", "cpu", "candidate", horizon)]
        checkpoint: dict[str, Any] = {}
        for channel in CHANNELS:
            legacy_width = float(legacy[channel])
            candidate_width = float(candidate[channel])
            flowstar_width = flowstar[horizon][channel]
            legacy_excess = legacy_width - flowstar_width
            reduction = legacy_width - candidate_width
            checkpoint[channel] = {
                "flowstar_width": flowstar_width,
                "legacy_width": legacy_width,
                "candidate_width": candidate_width,
                "legacy_excess": legacy_excess,
                "candidate_reduction": reduction,
                "fraction_of_legacy_excess_removed": (
                    reduction / legacy_excess if legacy_excess > 0 else None
                ),
                "candidate_no_wider": candidate_width <= legacy_width,
            }
        fixed[_label(horizon)] = checkpoint
    native_legacy = by_key[("native", "cpu", "legacy", 10.0)]
    native_candidate = by_key[("native", "cpu", "candidate", 10.0)]
    t1_t3_main = all(
        fixed[_label(horizon)][channel]["fraction_of_legacy_excess_removed"] >= 0.10
        for horizon in (1.0, 3.0)
        for channel in CHANNELS
    )
    h1_mechanism_threshold = any(
        fixed[_label(horizon)][channel]["fraction_of_legacy_excess_removed"] >= 0.10
        for horizon in (3.0, 6.32)
        for channel in CHANNELS
    )
    t6_no_regression = all(
        fixed["T6p32"][channel]["candidate_no_wider"] for channel in CHANNELS
    )
    native_gate = float(native_candidate["completed_horizon"]) >= 6.397083942944808
    runtime_ratios = {
        f"{schedule}_{_label(horizon)}": (
            float(by_key[(schedule, "cpu", "candidate", horizon)]["runtime_s"])
            / float(by_key[(schedule, "cpu", "legacy", horizon)]["runtime_s"])
        )
        for schedule, horizons in (("fixed", FIXED_HORIZONS), ("native", (10.0,)))
        for horizon in horizons
    }
    runtime_gate = all(ratio <= 2.0 for ratio in runtime_ratios.values())
    cpu_cuda_consistency: dict[str, Any] = {}
    for mode in MODES:
        cpu_key = ("fixed", "cpu", mode, 0.1)
        cuda_key = ("fixed", "cuda", mode, 0.1)
        if cpu_key not in by_key or cuda_key not in by_key:
            continue
        cpu_row = by_key[cpu_key]
        cuda_row = by_key[cuda_key]
        width_deltas = {
            channel: abs(float(cpu_row[channel]) - float(cuda_row[channel]))
            for channel in CHANNELS
        }
        cpu_cuda_consistency[mode] = {
            "scope": "implementation consistency only; CUDA is not a directed-rounding soundness lane",
            "status_equal": cpu_row["status"] == cuda_row["status"],
            "completed_horizon_abs_delta": abs(
                float(cpu_row["completed_horizon"]) - float(cuda_row["completed_horizon"])
            ),
            "accepted_steps_equal": int(cpu_row["accepted_steps"]) == int(cuda_row["accepted_steps"]),
            "rejected_attempts_equal": int(cpu_row["rejected_attempts"])
            == int(cuda_row["rejected_attempts"]),
            "width_abs_deltas": width_deltas,
            "max_width_abs_delta": max(width_deltas.values()),
            "consistent_at_1e_12": (
                cpu_row["status"] == cuda_row["status"]
                and abs(float(cpu_row["completed_horizon"]) - float(cuda_row["completed_horizon"]))
                <= 1e-12
                and int(cpu_row["accepted_steps"]) == int(cuda_row["accepted_steps"])
                and int(cpu_row["rejected_attempts"]) == int(cuda_row["rejected_attempts"])
                and max(width_deltas.values()) <= 1e-12
            ),
        }
    return {
        "schema": "vdp_normal_insertion_scientific_matrix_v2",
        "fixed": fixed,
        "native": {"legacy": dict(native_legacy), "candidate": dict(native_candidate)},
        "runtime_ratios_candidate_over_legacy": runtime_ratios,
        "cpu_cuda_consistency_T0p1": cpu_cuda_consistency,
        "gates": {
            "H1_factorization_explains_at_least_one_T3_or_T6p32_channel_10pct": h1_mechanism_threshold,
            "T1_T3_all_four_channels_remove_10pct_legacy_excess": t1_t3_main,
            "T6p32_no_channel_regression": t6_no_regression,
            "native_at_least_6p397083942944808": native_gate,
            "runtime_at_most_2x_legacy": runtime_gate,
            "reaches_T10": bool(native_candidate["completed_requested_horizon"]),
        },
        "decision": (
            "H1_ACCEPTED__ALL_PRODUCTION_THRESHOLDS_MET"
            if all((t1_t3_main, t6_no_regression, native_gate, runtime_gate))
            else "H1_ACCEPTED__PRODUCTION_THRESHOLDS_PARTIAL"
            if h1_mechanism_threshold and t6_no_regression and runtime_gate
            else "H1_REJECTED_OR_PRODUCTION_REGRESSED"
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(root: Path, rows: list[dict[str, Any]], *, include_cuda: bool) -> dict[str, Any]:
    _write_csv(root / "requests.csv", rows)
    summary = _summarize(rows)
    summary["rows"] = rows
    summary["cuda_scope"] = (
        "implementation consistency and measured runtime only; no directed-rounding soundness or speedup claim"
        if include_cuda
        else "not run"
    )
    (root / "matrix.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": summary["decision"], "gates": summary["gates"]}, sort_keys=True))
    return summary


def run(root: Path, wall_cap_s: float, include_cuda: bool) -> dict[str, Any]:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing non-empty matrix root: {root}")
    root.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for horizon in FIXED_HORIZONS:
        for mode in MODES:
            rows.append(_run_request(root, mode, "fixed", horizon, wall_cap_s))
    for mode in MODES:
        rows.append(_run_request(root, mode, "native", 10.0, wall_cap_s))
    if include_cuda:
        for device in ("cpu", "cuda"):
            for mode in MODES:
                rows.append(_run_request(root, mode, "fixed", 0.1, wall_cap_s, device=device))
    return _write_summary(root, rows, include_cuda=include_cuda)


def augment_existing_cpu_cuda(root: Path, wall_cap_s: float) -> dict[str, Any]:
    """Add the missing CPU side of a completed CUDA consistency mini-matrix."""
    root = root.resolve()
    matrix_path = root / "matrix.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    rows = list(json.loads(matrix_path.read_text(encoding="utf-8"))["rows"])
    existing = {
        (row["schedule"], row["device"], row["mode"], float(row["requested_horizon"]))
        for row in rows
    }
    for mode in MODES:
        key = ("fixed", "cpu", mode, 0.1)
        if key not in existing:
            rows.append(_run_request(root, mode, "fixed", 0.1, wall_cap_s, device="cpu"))
    return _write_summary(root, rows, include_cuda=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wall-cap-s", type=float, default=1800.0)
    parser.add_argument("--include-cuda", action="store_true")
    parser.add_argument("--augment-existing-cpu-cuda", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.augment_existing_cpu_cuda:
        augment_existing_cpu_cuda(args.output_root, args.wall_cap_s)
    else:
        run(args.output_root, args.wall_cap_s, args.include_cuda)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
