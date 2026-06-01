"""Best-effort parser for Flow* text output.

Flow* can emit several text formats depending on version and settings.  This
parser starts with conservative box parsing: it scans stdout/stderr or generated
text files for variable ranges of the form ``x in [lo, hi]`` and returns (i) a
proxy for the final reachable box and (ii) the hull of all seen boxes.

For the current ``chenxin415/flowstar`` toolbox runner, generated C++ programs
emit GNUPLOT interval files.  When ``parse_files(..., numeric_plot_vars=...)`` is
used, this parser also scans two-column numeric plot blocks and interprets the
second column as the plotted state variable range.  This is still a box/range
comparison, not Taylor-model coefficient parsing.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_RANGE_PATTERNS = [
    re.compile(rf"\b(?P<var>[A-Za-z_]\w*)\s+(?:in|=|:)\s*\[\s*(?P<lo>{_NUMBER})\s*,\s*(?P<hi>{_NUMBER})\s*\]"),
    re.compile(rf"\b(?P<var>[A-Za-z_]\w*)\s*\[\s*(?P<lo>{_NUMBER})\s*,\s*(?P<hi>{_NUMBER})\s*\]"),
]
_NUMERIC_PAIR = re.compile(rf"^\s*(?P<x>{_NUMBER})\s+(?P<y>{_NUMBER})(?:\s|$)")


@dataclass
class ParsedFlowstarOutput:
    status: str
    endpoint_box: dict[str, tuple[float, float]]
    last_segment_box: dict[str, tuple[float, float]]
    tube_box: dict[str, tuple[float, float]]
    num_boxes: int
    message: str = ""

    @property
    def final_box(self) -> dict[str, tuple[float, float]]:
        """Compatibility alias for explicit endpoint text or GNUPLOT last segment."""
        return self.endpoint_box or self.last_segment_box

    @property
    def flowpipe_box(self) -> dict[str, tuple[float, float]]:
        """Compatibility alias for the full tube hull."""
        return self.tube_box


def _merge_box(dst: dict[str, tuple[float, float]], var: str, lo: float, hi: float) -> None:
    if lo > hi:
        lo, hi = hi, lo
    if var not in dst:
        dst[var] = (lo, hi)
    else:
        old_lo, old_hi = dst[var]
        dst[var] = (min(old_lo, lo), max(old_hi, hi))


def _merge_boxes(dst: dict[str, tuple[float, float]], src: Mapping[str, tuple[float, float]]) -> None:
    for var, (lo, hi) in src.items():
        _merge_box(dst, var, lo, hi)


def parse_text(text: str, variables: Iterable[str] | None = None) -> ParsedFlowstarOutput:
    variables_set = set(variables or [])
    boxes: list[dict[str, tuple[float, float]]] = []
    current: dict[str, tuple[float, float]] = {}
    for line in text.splitlines():
        found_on_line = False
        for pat in _RANGE_PATTERNS:
            for m in pat.finditer(line):
                var = m.group("var")
                if variables_set and var not in variables_set:
                    continue
                lo = float(m.group("lo"))
                hi = float(m.group("hi"))
                current[var] = (min(lo, hi), max(lo, hi))
                found_on_line = True
        if found_on_line and variables_set and variables_set.issubset(current.keys()):
            boxes.append(dict(current))
            current = {}
    if current:
        boxes.append(dict(current))
    if not boxes:
        return ParsedFlowstarOutput("unparsed", {}, {}, {}, 0, "no variable ranges found")
    endpoint_box = boxes[-1]
    hull: dict[str, tuple[float, float]] = {}
    for box in boxes:
        for var, (lo, hi) in box.items():
            _merge_box(hull, var, lo, hi)
    return ParsedFlowstarOutput("parsed", endpoint_box, {}, hull, len(boxes))


def _filename_mentions_var(path: Path, var: str) -> bool:
    stem = path.stem
    tokens = re.split(r"[^A-Za-z0-9]+", stem)
    return var in tokens or stem.endswith(f"_{var}") or f"_{var}_" in stem


def _parse_numeric_pair_blocks(text: str, var: str) -> list[dict[str, tuple[float, float]]]:
    """Parse GNUPLOT-like two-column blocks as boxes for one plotted variable."""
    boxes: list[dict[str, tuple[float, float]]] = []
    ys: list[float] = []

    def flush() -> None:
        nonlocal ys
        if ys:
            boxes.append({var: (min(ys), max(ys))})
            ys = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("set ") or line.startswith("plot "):
            flush()
            continue
        m = _NUMERIC_PAIR.match(line)
        if m:
            ys.append(float(m.group("y")))
        else:
            flush()
    flush()
    return boxes


def parse_files(
    paths: Iterable[str | Path],
    variables: Iterable[str] | None = None,
    numeric_plot_vars: Iterable[str] | None = None,
) -> ParsedFlowstarOutput:
    variables_list = list(variables or [])
    numeric_vars = list(numeric_plot_vars or [])
    explicit_text_parts: list[str] = []
    numeric_boxes: list[dict[str, tuple[float, float]]] = []

    for p in paths:
        path = Path(p)
        if not (path.exists() and path.is_file()):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        explicit_text_parts.append(text)
        if path.suffix.lower() in {".stdout", ".stderr", ".txt", ".cpp", ".model"}:
            # Avoid interpreting unrelated diagnostic numbers as plot points.
            continue
        for var in numeric_vars:
            if _filename_mentions_var(path, var):
                numeric_boxes.extend(_parse_numeric_pair_blocks(text, var))

    explicit = parse_text("\n".join(explicit_text_parts), variables=variables_list)
    endpoint_box: dict[str, tuple[float, float]] = {}
    last_segment_box: dict[str, tuple[float, float]] = {}
    tube_box: dict[str, tuple[float, float]] = {}
    num_boxes = 0

    if explicit.status == "parsed":
        endpoint_box.update(explicit.endpoint_box)
        tube_box.update(explicit.tube_box)
        num_boxes += explicit.num_boxes

    # Numeric plot files usually contain one variable per file. These GNUPLOT
    # blocks are flowpipe segment boxes, not endpoint boxes.
    last_for_var: dict[str, tuple[float, float]] = {}
    for box in numeric_boxes:
        num_boxes += 1
        for var, (lo, hi) in box.items():
            _merge_box(tube_box, var, lo, hi)
            last_for_var[var] = (lo, hi)
    _merge_boxes(last_segment_box, last_for_var)

    if not endpoint_box and not last_segment_box and not tube_box:
        return ParsedFlowstarOutput("unparsed", {}, {}, {}, 0, "no variable ranges found")
    return ParsedFlowstarOutput("parsed", endpoint_box, last_segment_box, tube_box, num_boxes)


def widths(box: Mapping[str, tuple[float, float]], metric_vars: Iterable[str]) -> list[float]:
    vals: list[float] = []
    for var in metric_vars:
        if var in box:
            lo, hi = box[var]
            vals.append(max(0.0, float(hi) - float(lo)))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Flow* stdout/plot text for box ranges.")
    parser.add_argument("files", nargs="+")
    parser.add_argument("--vars", nargs="*", default=None)
    parser.add_argument("--numeric-plot-vars", nargs="*", default=None)
    args = parser.parse_args()
    print(parse_files(args.files, variables=args.vars, numeric_plot_vars=args.numeric_plot_vars))


if __name__ == "__main__":
    main()
