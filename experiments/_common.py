from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch


def write_csv(path: str | None, rows: list[dict]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "h",
        "order",
        "status",
        "final_width",
        "flowpipe_width",
        "runtime_s",
        "validation_attempts",
        "containment_failures",
        "device",
        "dtype",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def max_final_width(result) -> float:
    return float(result.final_tm.max_width().detach().cpu())


def max_flowpipe_width(result) -> float:
    widths = [seg.tm.max_width() for seg in result.segments]
    return float(torch.max(torch.stack(widths)).detach().cpu()) if widths else 0.0


def now() -> float:
    return time.perf_counter()


def dtype_device(result) -> tuple[str, str]:
    dom = result.final_tm.domain
    if dom:
        return str(dom[0].lo.device), str(dom[0].lo.dtype).replace("torch.", "")
    return "cpu", "float64"


def interval_contains_all(interval, values: Iterable[float], *, tol: float = 1e-10) -> int:
    failures = 0
    for v in values:
        if not interval.contains(v, tol=tol):
            failures += 1
    return failures
