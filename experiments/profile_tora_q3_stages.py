#!/usr/bin/env python3
"""Profile one fixed-shape TORA-Q3 step by mathematical stage and callsite."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Iterable

import torch

from torch_tm_flowpipe.batched_dense_tm import (
    compiled_point_enclosure_status,
    dense_transient_ledger_suppressed,
    monomial_interval_cache_status,
)
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_initial_model,
    compose_tora_q3_step,
    dense_validation_batch,
    dense_tora_q3_dr_step,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    tora_q3_boundary_from_model,
)


TARGET_OPERATORS = {"aten::item", "aten::_local_scalar_dense", "aten::to"}
FRAME_PATTERN = re.compile(r"(?P<path>.*?\.py)(?:\((?P<paren_line>\d+)\)|:(?P<colon_line>\d+))(?::\s*(?P<function>.*))?")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_sha256(values: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def event_stage(event: Any) -> str:
    current = getattr(event, "cpu_parent", None)
    while current is not None:
        name = str(getattr(current, "name", ""))
        if name.startswith("stage::"):
            return name.removeprefix("stage::")
        current = getattr(current, "cpu_parent", None)
    return "unscoped"


def event_stack(event: Any) -> list[str]:
    current = event
    while current is not None:
        stack = list(getattr(current, "stack", ()) or ())
        if stack:
            return [str(frame) for frame in stack]
        current = getattr(current, "cpu_parent", None)
    return []


def sanitize_callsite(frames: list[str], repository: Path) -> str:
    parsed: list[tuple[str, int, str]] = []
    for frame in frames:
        match = FRAME_PATTERN.search(frame)
        if match is None:
            continue
        path = match.group("path")
        line = int(match.group("paren_line") or match.group("colon_line"))
        function = (match.group("function") or "unknown").strip()
        parsed.append((path, line, function))
    root_text = str(repository.resolve()) + "/"
    for path, line, function in parsed:
        if path.startswith(root_text):
            return f"{path.removeprefix(root_text)}:{line}:{function}"
    for path, line, function in parsed:
        marker = "/torch_tm_flowpipe/"
        if marker in path:
            relative = "src/torch_tm_flowpipe/" + path.split(marker, 1)[1]
            return f"{relative}:{line}:{function}"
    if parsed:
        path, line, function = parsed[0]
        return f"<dependency>/{Path(path).name}:{line}:{function}"
    return "<stack-unavailable>:0:unknown"


def shape_signature(event: Any) -> str:
    shapes = getattr(event, "input_shapes", None)
    if not shapes:
        return "[]"
    return json.dumps(shapes, separators=(",", ":"), default=str)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_events(events: list[Any], repository: Path) -> dict[str, Any]:
    operator_counts: Counter[str] = Counter()
    operator_cpu_us: defaultdict[str, float] = defaultdict(float)
    operator_shapes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    stage_operator: defaultdict[str, Counter[str]] = defaultdict(Counter)
    targeted: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for event in events:
        name = str(getattr(event, "name", ""))
        if not name.startswith("aten::"):
            continue
        stage = event_stage(event)
        operator_counts[name] += 1
        operator_cpu_us[name] += float(getattr(event, "self_cpu_time_total", 0.0))
        operator_shapes[name][shape_signature(event)] += 1
        stage_operator[stage][name] += 1
        if name in TARGET_OPERATORS:
            callsite = sanitize_callsite(event_stack(event), repository)
            targeted[(stage, callsite, shape_signature(event))][name] += 1

    total_item = operator_counts["aten::item"]
    total_local = operator_counts["aten::_local_scalar_dense"]
    total_sync = max(total_item, total_local)
    total_to = operator_counts["aten::to"]

    stage_rows: list[dict[str, Any]] = []
    for stage, counts in sorted(stage_operator.items()):
        item = counts["aten::item"]
        local = counts["aten::_local_scalar_dense"]
        sync = max(item, local)
        stage_rows.append(
            {
                "stage": stage,
                "aten_event_count": sum(counts.values()),
                "aten_item_count": item,
                "local_scalar_dense_count": local,
                "host_scalar_sync_estimate": sync,
                "host_scalar_sync_percent": (
                    100.0 * sync / total_sync if total_sync else 0.0
                ),
                "aten_to_count": counts["aten::to"],
                "aten_to_percent": (
                    100.0 * counts["aten::to"] / total_to if total_to else 0.0
                ),
            }
        )
    stage_rows.sort(
        key=lambda row: (
            -int(row["host_scalar_sync_estimate"]),
            -int(row["aten_to_count"]),
            str(row["stage"]),
        )
    )

    sync_rows: list[dict[str, Any]] = []
    to_rows: list[dict[str, Any]] = []
    for (stage, callsite, shapes), counts in targeted.items():
        item = counts["aten::item"]
        local = counts["aten::_local_scalar_dense"]
        sync = max(item, local)
        if sync:
            sync_rows.append(
                {
                    "stage": stage,
                    "source_callsite": callsite,
                    "input_shapes": shapes,
                    "aten_item_count": item,
                    "local_scalar_dense_count": local,
                    "host_scalar_sync_estimate": sync,
                    "percent_of_total_host_sync": (
                        100.0 * sync / total_sync if total_sync else 0.0
                    ),
                }
            )
        if counts["aten::to"]:
            to_rows.append(
                {
                    "stage": stage,
                    "source_callsite": callsite,
                    "input_shapes": shapes,
                    "aten_to_count": counts["aten::to"],
                    "percent_of_total_aten_to": (
                        100.0 * counts["aten::to"] / total_to if total_to else 0.0
                    ),
                }
            )
    sync_rows.sort(
        key=lambda row: (-int(row["host_scalar_sync_estimate"]), str(row["source_callsite"]))
    )
    to_rows.sort(
        key=lambda row: (-int(row["aten_to_count"]), str(row["source_callsite"]))
    )

    operator_rows: list[dict[str, Any]] = []
    for operator, count in operator_counts.most_common():
        common_shape = operator_shapes[operator].most_common(1)
        operator_rows.append(
            {
                "operator": operator,
                "event_count": count,
                "self_cpu_time_us": operator_cpu_us[operator],
                "most_common_input_shapes": common_shape[0][0] if common_shape else "[]",
                "most_common_input_shapes_count": common_shape[0][1] if common_shape else 0,
            }
        )

    repository_callsites = sum(
        row["source_callsite"].startswith(("src/", "experiments/", "scripts/"))
        for row in [*sync_rows, *to_rows]
    )
    return {
        "totals": {
            "aten_event_count": sum(operator_counts.values()),
            "aten_item_count": total_item,
            "local_scalar_dense_count": total_local,
            "host_scalar_sync_estimate": total_sync,
            "aten_to_count": total_to,
            "targeted_repository_callsite_row_count": repository_callsites,
        },
        "stage_rows": stage_rows,
        "sync_rows": sync_rows,
        "to_rows": to_rows,
        "operator_rows": operator_rows,
    }


def render_report(summary: dict[str, Any], stage_rows: list[dict[str, Any]], sync_rows: list[dict[str, Any]], to_rows: list[dict[str, Any]]) -> str:
    totals = summary["totals"]
    lines = [
        "# TORA-Q3 GPU bottleneck report",
        "",
        "This report describes profiler event counts, not formal runtime. The raw Chrome trace remains private.",
        "",
        "## Baseline totals",
        "",
        f"- `aten::item`: {totals['aten_item_count']}",
        f"- `aten::_local_scalar_dense`: {totals['local_scalar_dense_count']}",
        f"- paired host-scalar synchronization estimate: {totals['host_scalar_sync_estimate']}",
        f"- `aten::to`: {totals['aten_to_count']}",
        "",
        "## Largest mathematical stages",
        "",
        "| stage | host-sync estimate | share | aten::to |",
        "|---|---:|---:|---:|",
    ]
    for row in stage_rows[:20]:
        lines.append(
            f"| {row['stage']} | {row['host_scalar_sync_estimate']} | "
            f"{float(row['host_scalar_sync_percent']):.3f}% | {row['aten_to_count']} |"
        )
    lines.extend(
        [
            "",
            "## Largest host-sync callsites",
            "",
            "| stage | source | host-sync estimate | share |",
            "|---|---|---:|---:|",
        ]
    )
    for row in sync_rows[:20]:
        lines.append(
            f"| {row['stage']} | `{row['source_callsite']}` | "
            f"{row['host_scalar_sync_estimate']} | "
            f"{float(row['percent_of_total_host_sync']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Largest device/dtype conversion callsites",
            "",
            "| stage | source | aten::to | share |",
            "|---|---|---:|---:|",
        ]
    )
    for row in to_rows[:20]:
        lines.append(
            f"| {row['stage']} | `{row['source_callsite']}` | "
            f"{row['aten_to_count']} | {float(row['percent_of_total_aten_to']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "The paired synchronization estimate is `max(aten::item, aten::_local_scalar_dense)` so one scalar extraction is not counted twice. Stage and callsite CSV files retain the complete sanitized aggregation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-trace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--point-enclosure-backend",
        choices=("eager", "compiled"),
        default="eager",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    if args.private_trace.exists():
        raise FileExistsError(args.private_trace)
    output.mkdir(parents=True, exist_ok=True)
    args.private_trace.parent.mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    control_lower = torch.full((48,), 9.8, dtype=torch.float64, device=device)
    control_upper = torch.full((48,), 10.2, dtype=torch.float64, device=device)
    base = build_tora_q3_initial_model(
        control_lower, control_upper, device=device
    )
    boundary = tora_q3_boundary_from_model(base)
    local, carry = normalize_tora_q3_boundary(
        boundary, identity_tora_q3_carry(48, device=device)
    )
    with dense_validation_batch():
        with dense_transient_ledger_suppressed():
            warmup = dense_tora_q3_dr_step(
                local,
                capture_trace=False,
                point_enclosure_backend=args.point_enclosure_backend,
            )
    synchronize(device)
    if not warmup.accepted:
        raise RuntimeError("profiler warm-up did not validate")

    from torch.profiler import ProfilerActivity, profile, record_function

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        with_stack=True,
        record_shapes=True,
    ) as trace:
        with dense_validation_batch():
            with record_function("stage::full_q3_step"):
                with dense_transient_ledger_suppressed():
                    local_step = dense_tora_q3_dr_step(
                        local,
                        capture_trace=False,
                        profile_stages=True,
                        point_enclosure_backend=args.point_enclosure_backend,
                    )
            physical_step = compose_tora_q3_step(
                local_step,
                carry,
                profile_stages=True,
            )
            projection = project_tora_q3_endpoint_to_affine(
                local_step.segment_tm,
                profile_stages=True,
            )
        synchronize(device)
    if not local_step.accepted or not physical_step.accepted:
        raise RuntimeError("profiled step did not validate")
    trace.export_chrome_trace(str(args.private_trace))
    events = list(trace.events())
    aggregates = aggregate_events(events, Path.cwd())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    profiled_paths = (
        Path("src/torch_tm_flowpipe/batched_dense_tm.py"),
        Path("src/torch_tm_flowpipe/tora_q3.py"),
        Path("experiments/profile_tora_q3_stages.py"),
    )
    instrumentation_diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *(str(path) for path in profiled_paths)],
        check=True,
        capture_output=True,
    ).stdout
    summary = {
        "schema": "tora_q3_stage_profiler_summary_v1",
        "status": "PASS",
        "commit": commit,
        "instrumented_worktree": True,
        "instrumentation_diff_sha256": hashlib.sha256(instrumentation_diff).hexdigest(),
        "profiled_source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in profiled_paths
        },
        "device": str(device),
        "dtype": "float64",
        "batch": 48,
        "term_count": local.poly.basis.num_terms,
        "with_stack": True,
        "record_shapes": True,
        "raw_trace_public": False,
        "raw_trace_sha256": hashlib.sha256(args.private_trace.read_bytes()).hexdigest(),
        "totals": aggregates["totals"],
        "output_status_sha256": tensor_sha256(
            (
                local_step.endpoint_lower,
                local_step.endpoint_upper,
                local_step.tube_lower,
                local_step.tube_upper,
                physical_step.endpoint_lower,
                physical_step.endpoint_upper,
                projection.center,
                projection.linear,
                projection.remainder_lower,
                projection.remainder_upper,
            )
        ),
        "point_enclosure_backend_requested": args.point_enclosure_backend,
        "point_enclosure_backend_status": compiled_point_enclosure_status(),
        "monomial_interval_cache_status": monomial_interval_cache_status(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_num_threads": torch.get_num_threads(),
        },
        "timing_claim": "profiler timing is excluded from formal runtime claims",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        output / "baseline_stage_counts.csv",
        [
            "stage",
            "aten_event_count",
            "aten_item_count",
            "local_scalar_dense_count",
            "host_scalar_sync_estimate",
            "host_scalar_sync_percent",
            "aten_to_count",
            "aten_to_percent",
        ],
        aggregates["stage_rows"],
    )
    write_csv(
        output / "top_host_sync_call_sites.csv",
        [
            "stage",
            "source_callsite",
            "input_shapes",
            "aten_item_count",
            "local_scalar_dense_count",
            "host_scalar_sync_estimate",
            "percent_of_total_host_sync",
        ],
        aggregates["sync_rows"],
    )
    write_csv(
        output / "top_to_call_sites.csv",
        [
            "stage",
            "source_callsite",
            "input_shapes",
            "aten_to_count",
            "percent_of_total_aten_to",
        ],
        aggregates["to_rows"],
    )
    write_csv(
        output / "operator_counts.csv",
        [
            "operator",
            "event_count",
            "self_cpu_time_us",
            "most_common_input_shapes",
            "most_common_input_shapes_count",
        ],
        aggregates["operator_rows"],
    )
    args.report.write_text(
        render_report(
            summary,
            aggregates["stage_rows"],
            aggregates["sync_rows"],
            aggregates["to_rows"],
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", **aggregates["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
