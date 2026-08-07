#!/usr/bin/env python3
"""Attribute TORA-Q3 Chrome-trace operators to sanitized Python source lines."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import json
from pathlib import Path
import re
from typing import Any, Iterable


TARGETS = {"aten::item", "aten::_local_scalar_dense", "aten::to"}
HEADER_PATTERN = re.compile(
    r'"ph":\s*"(?P<ph>[^"]+)".*?'
    r'"cat":\s*"(?P<cat>[^"]+)".*?'
    r'"name":\s*(?P<name>"(?:\\.|[^"\\])*").*?'
    r'"pid":\s*(?P<pid>\d+).*?"tid":\s*(?P<tid>\d+)'
)
TIMING_PATTERN = re.compile(
    r'"ts":\s*(?P<ts>[0-9.eE+-]+),\s*"dur":\s*(?P<dur>[0-9.eE+-]+)'
)
PYTHON_FRAME_PATTERN = re.compile(
    r"(?P<path>(?:torch_tm_flowpipe/|profile_tora_q3_stages\.py)[^(]+)"
    r"\((?P<line>\d+)\):\s*(?P<function>.*)"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_python_frame(name: str) -> str | None:
    match = PYTHON_FRAME_PATTERN.fullmatch(name)
    if match is None:
        return None
    path = match.group("path")
    if path.startswith("torch_tm_flowpipe/"):
        path = f"src/{path}"
    return f"{path}:{int(match.group('line'))}:{match.group('function').strip()}"


def _input_dims(line: str) -> str | None:
    marker = '"Input Dims": '
    if marker not in line:
        return None
    payload = line.split(marker, 1)[1]
    for suffix in (', "Ev Idx"', ", \"Call stack\""):
        if suffix in payload:
            payload = payload.split(suffix, 1)[0]
    try:
        return json.dumps(json.loads(payload), separators=(",", ":"))
    except json.JSONDecodeError:
        return None


def parse_trace(
    path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[float, float, int, str]], list[tuple[float, float, int, str]]]:
    targets: list[dict[str, Any]] = []
    source_intervals: list[tuple[float, float, int, str]] = []
    stage_intervals: list[tuple[float, float, int, str]] = []
    header: dict[str, Any] | None = None
    pending_target: dict[str, Any] | None = None

    def finish_target() -> None:
        nonlocal pending_target
        if pending_target is not None:
            pending_target.setdefault("input_shapes", "[]")
            targets.append(pending_target)
            pending_target = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_match = HEADER_PATTERN.search(line)
            if header_match is not None:
                finish_target()
                header = {
                    "ph": header_match.group("ph"),
                    "cat": header_match.group("cat"),
                    "name": json.loads(header_match.group("name")),
                    "pid": int(header_match.group("pid")),
                    "tid": int(header_match.group("tid")),
                }
                continue
            timing_match = TIMING_PATTERN.search(line)
            if timing_match is not None and header is not None:
                start = float(timing_match.group("ts"))
                duration = float(timing_match.group("dur"))
                end = start + duration
                name = str(header["name"])
                cat = str(header["cat"])
                thread = int(header["tid"])
                if cat == "cpu_op" and name in TARGETS:
                    pending_target = {
                        "operator": name,
                        "start": start,
                        "thread": thread,
                    }
                elif cat == "user_annotation" and name.startswith("stage::"):
                    stage_intervals.append(
                        (start, end, thread, name.removeprefix("stage::"))
                    )
                elif cat == "python_function":
                    source = sanitize_python_frame(name)
                    if source is not None:
                        source_intervals.append((start, end, thread, source))
                header = None
                continue
            if pending_target is not None:
                shapes = _input_dims(line)
                if shapes is not None:
                    pending_target["input_shapes"] = shapes
    finish_target()
    return targets, source_intervals, stage_intervals


def attribute_points(
    targets: list[dict[str, Any]],
    intervals: Iterable[tuple[float, float, int, str]],
    *,
    missing: str,
) -> list[str]:
    by_thread_targets: defaultdict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    by_thread_intervals: defaultdict[int, list[tuple[float, float, str]]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_thread_targets[int(target["thread"])].append((index, target))
    for start, end, thread, label in intervals:
        by_thread_intervals[thread].append((start, end, label))

    labels = [missing] * len(targets)
    for thread, points in by_thread_targets.items():
        ordered_points = sorted(points, key=lambda item: float(item[1]["start"]))
        ordered_intervals = sorted(by_thread_intervals.get(thread, []))
        active: list[tuple[float, float, float, int, str]] = []
        cursor = 0
        for target_index, target in ordered_points:
            timestamp = float(target["start"])
            while cursor < len(ordered_intervals) and ordered_intervals[cursor][0] <= timestamp:
                start, end, label = ordered_intervals[cursor]
                duration = end - start
                heapq.heappush(active, (duration, -start, end, cursor, label))
                cursor += 1
            while active and active[0][2] < timestamp:
                heapq.heappop(active)
            if active:
                labels[target_index] = active[0][4]
    return labels


def aggregate_attribution(
    targets: list[dict[str, Any]],
    source_labels: list[str],
    stage_labels: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    counts: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for target, source, stage in zip(targets, source_labels, stage_labels, strict=True):
        counts[(stage, source, str(target["input_shapes"]))][str(target["operator"])] += 1

    operator_totals = Counter(str(target["operator"]) for target in targets)
    sync_total = max(
        operator_totals["aten::item"], operator_totals["aten::_local_scalar_dense"]
    )
    to_total = operator_totals["aten::to"]
    sync_rows: list[dict[str, Any]] = []
    to_rows: list[dict[str, Any]] = []
    for (stage, source, shapes), row_counts in counts.items():
        item = row_counts["aten::item"]
        local = row_counts["aten::_local_scalar_dense"]
        sync = max(item, local)
        if sync:
            sync_rows.append(
                {
                    "stage": stage,
                    "source_callsite": source,
                    "input_shapes": shapes,
                    "aten_item_count": item,
                    "local_scalar_dense_count": local,
                    "host_scalar_sync_estimate": sync,
                    "percent_of_total_host_sync": 100.0 * sync / sync_total,
                }
            )
        to_count = row_counts["aten::to"]
        if to_count:
            to_rows.append(
                {
                    "stage": stage,
                    "source_callsite": source,
                    "input_shapes": shapes,
                    "aten_to_count": to_count,
                    "percent_of_total_aten_to": 100.0 * to_count / to_total,
                }
            )
    sync_rows.sort(
        key=lambda row: (-int(row["host_scalar_sync_estimate"]), str(row["source_callsite"]))
    )
    to_rows.sort(
        key=lambda row: (-int(row["aten_to_count"]), str(row["source_callsite"]))
    )
    attributable = Counter(
        target["operator"]
        for target, label in zip(targets, source_labels, strict=True)
        if not label.startswith("<")
    )
    summary = {
        "operator_counts": dict(sorted(operator_totals.items())),
        "source_attributed_operator_counts": dict(sorted(attributable.items())),
        "source_attribution_percent": {
            operator: 100.0 * attributable[operator] / count
            for operator, count in sorted(operator_totals.items())
        },
        "host_scalar_sync_estimate": sync_total,
        "aten_to_count": to_total,
        "source_host_sync_row_count": len(sync_rows),
        "source_to_row_count": len(to_rows),
    }
    return sync_rows, to_rows, summary


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-trace", type=Path, required=True)
    parser.add_argument("--profiler-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.profiler_summary.read_text(encoding="utf-8"))["totals"]
    targets, sources, stages = parse_trace(args.private_trace)
    source_labels = attribute_points(targets, sources, missing="<repository-frame-unavailable>")
    stage_labels = attribute_points(targets, stages, missing="unscoped")
    sync_rows, to_rows, summary = aggregate_attribution(
        targets, source_labels, stage_labels
    )
    for key in ("aten_item_count", "local_scalar_dense_count", "aten_to_count"):
        operator = {
            "aten_item_count": "aten::item",
            "local_scalar_dense_count": "aten::_local_scalar_dense",
            "aten_to_count": "aten::to",
        }[key]
        if int(expected[key]) != int(summary["operator_counts"].get(operator, 0)):
            raise ValueError(f"raw trace {operator} count does not match profiler summary")
    summary.update(
        {
            "schema": "tora_q3_chrome_source_attribution_v1",
            "status": "PASS",
            "method": "innermost containing same-thread repository python_function interval",
            "raw_trace_public": False,
            "raw_trace_sha256": file_sha256(args.private_trace),
            "profiler_summary_sha256": file_sha256(args.profiler_summary),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "source_host_sync_call_sites.csv",
        [
            "stage",
            "source_callsite",
            "input_shapes",
            "aten_item_count",
            "local_scalar_dense_count",
            "host_scalar_sync_estimate",
            "percent_of_total_host_sync",
        ],
        sync_rows,
    )
    write_csv(
        args.output_dir / "source_to_call_sites.csv",
        [
            "stage",
            "source_callsite",
            "input_shapes",
            "aten_to_count",
            "percent_of_total_aten_to",
        ],
        to_rows,
    )
    (args.output_dir / "source_attribution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
