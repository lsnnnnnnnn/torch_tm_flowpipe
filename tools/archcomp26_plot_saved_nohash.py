"""Plot two fixed ARCH-COMP26 saved-record views without content digests.

This is a visualization of saved interval boxes, not a solver or certificate.
It intentionally does not import the source exporter with digest receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import struct


ACC_STATES = ("x_lead", "v_lead", "a_lead", "x_ego", "v_ego", "a_ego")
ACC_RECORD = struct.Struct("<QQd24d")
COLORS = ("#1f77b4", "#d95f02", "#7570b3", "#1b9e77")


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank row")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(row)
    return rows


def _bounds(value: object, count: int, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label}: expected {count} intervals")
    result = []
    for index, pair in enumerate(value):
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                       or not math.isfinite(x) for x in pair)
                or pair[0] > pair[1]):
            raise ValueError(f"{label}[{index}]: invalid interval")
        result.append([float(pair[0]), float(pair[1])])
    return result


def _outward(value: Fraction, toward: float) -> float:
    rounded = float(value)
    if not math.isfinite(rounded):
        raise ValueError("derived margin is nonfinite")
    exact_rounded = Fraction.from_float(rounded)
    if (toward < 0 and exact_rounded > value) or (toward > 0 and exact_rounded < value):
        rounded = math.nextafter(rounded, toward)
    return rounded


def _acc_margin(tube: list[list[float]]) -> list[float]:
    # Rational 7/5 represents the stated decimal 1.4 exactly. Saved binary64
    # box endpoints are converted exactly, then each result rounds outward.
    lo = (Fraction.from_float(tube[0][0]) - Fraction.from_float(tube[3][1])
          - Fraction(7, 5) * Fraction.from_float(tube[4][1]) - 10)
    hi = (Fraction.from_float(tube[0][1]) - Fraction.from_float(tube[3][0])
          - Fraction(7, 5) * Fraction.from_float(tube[4][0]) - 10)
    return [_outward(lo, -math.inf), _outward(hi, math.inf)]


def _acc_native(path: Path) -> list[list[float]]:
    size = path.stat().st_size
    if size != 50 * ACC_RECORD.size:
        raise ValueError(f"{path}: expected exactly 50 native six-state records")
    intervals = []
    with path.open("rb") as handle:
        for step in range(1, 51):
            lane, recorded_step, h, *flat = ACC_RECORD.unpack(handle.read(ACC_RECORD.size))
            if lane != 0 or recorded_step != step or h != 0.1:
                raise ValueError(f"{path}: lane/step/h mismatch at record {step}")
            rows = [flat[index:index + 4] for index in range(0, 24, 4)]
            tube = _bounds([row[:2] for row in rows], 6, f"{path}:step {step} tube")
            _bounds([row[2:] for row in rows], 6, f"{path}:step {step} endpoint")
            intervals.append(_acc_margin(tube))
    return intervals


def _acc_author(path: Path) -> list[list[float]]:
    rows = _jsonl(path)
    if len(rows) != 50:
        raise ValueError(f"{path}: expected 50 saved range rows")
    intervals = []
    for step, row in enumerate(rows, 1):
        if row.get("substep") != step or row.get("accepted") is not True or row.get("local_h") != 0.1:
            raise ValueError(f"{path}: step {step} is absent, rejected, or has wrong h")
        tube = _bounds(row.get("tube"), 6, f"{path}:step {step} tube")
        _bounds(row.get("endpoint"), 6, f"{path}:step {step} endpoint")
        intervals.append(_acc_margin(tube))
    return intervals


def _acc(args: argparse.Namespace) -> dict:
    source_paths = (args.native, args.huan, args.xiangru, args.p3)
    labels = ("Native corrected VAR", "Huan", "Xiangru", "ours/P3")
    intervals = (_acc_native(args.native), _acc_author(args.huan),
                 _acc_author(args.xiangru), _acc_author(args.p3))
    initial = _acc_margin([[90, 110], [32, 32.2], [0, 0], [10, 11], [30, 30.2], [0, 0]])
    series = []
    for label, path, bounds in zip(labels, source_paths, intervals):
        result_path = path.with_name("RESULT.json")
        evidence = {"binding": "adjacent declaration; not content-bound to range records"}
        if result_path.is_file():
            result = _object(result_path)
            evidence.update({"path": str(result_path.resolve()), "status": result.get("status"),
                             "completed_substeps": result.get("completed_substeps"),
                             "driver_return": result.get("driver_return"),
                             "exit_code": result.get("exit_code")})
        config_path = path.with_name("START.json")
        series.append({"label": label, "source": _source(path),
                       "run_configuration": (_source(config_path) if config_path.is_file() else None),
                       "run_evidence": evidence,
                       "acceptance": ("not encoded in native binary" if label.startswith("Native")
                                      else "accepted=true in every saved JSONL row"),
                       "tube_intervals": bounds, "min_lower": min(row[0] for row in bounds)})
    return {"schema": "archcomp26-saved-interval-plot-v1", "plot": "acc-four-method-tube-margin",
            "contract": "2026 ACC participant-order variant, four methods, T=5, full initial box",
            "step_size": 0.1, "expected_steps": 50, "view": "tube", "initial_interval": initial,
            "property": "all-time x_lead-x_ego-1.4*v_ego-10 >= 0",
            "derivation": "outward rational image of each saved six-state axis-aligned tube box; correlations unavailable",
            "series": series,
            "qualification": "saved-box projection; adjacent run outcomes unbound; not an independent end-to-end NN certificate"}


def _quad(args: argparse.Namespace) -> dict:
    rows = _jsonl(args.observations)
    if len(rows) != 1000:
        raise ValueError(f"{args.observations}: expected 1000 pooled rows")
    tube = []
    endpoints = []
    for step, row in enumerate(rows, 1):
        if (row.get("substep") != step or row.get("accepted_count") != 1024
                or row.get("rejected_lanes") != []
                or row.get("status_counts") != {"0": 1024}):
            raise ValueError(f"{args.observations}: incomplete acceptance at step {step}")
        union = row.get("tube_endpoint_union_12x4")
        if not isinstance(union, list) or len(union) != 12:
            raise ValueError(f"{args.observations}: step {step} lacks 12-state pooled union")
        for index, state in enumerate(union):
            if not isinstance(state, list) or len(state) != 4:
                raise ValueError(f"{args.observations}: step {step} state {index} invalid")
            _bounds([state[:2], state[2:]], 2, f"{args.observations}:step {step} state {index}")
        tube.append([float(union[2][0]), float(union[2][1])])
        endpoints.append([float(union[2][2]), float(union[2][3])])
    result_path = args.observations.with_name("RESULT.json")
    result = _object(result_path) if result_path.is_file() else {}
    return {"schema": "archcomp26-saved-interval-plot-v1", "plot": "quad-paper-p3-pooled-tube-x3",
            "contract": "2026 selected paper-equation QUAD, T=5, 1024 boxes, 1000 substeps",
            "step_size": 0.005, "expected_steps": 1000, "view": "tube",
            "initial_interval": [-0.4, 0.4], "target_endpoint_interval": [0.94, 1.06],
            "series": [{"label": "P3 pooled union of 1024 accepted boxes",
                        "source": _source(args.observations),
                        "run_configuration": (_source(args.observations.with_name("START.json"))
                                              if args.observations.with_name("START.json").is_file() else None),
                        "run_evidence": {"binding": "adjacent declaration; not content-bound to pooled rows",
                                         "path": str(result_path.resolve()) if result else None,
                                         "status": result.get("status"),
                                         "completed_substeps": result.get("completed_substeps"),
                                         "end_to_end_strict_certificate": result.get("end_to_end_strict_certificate")},
                        "acceptance": "1024 accepted, status 0, no rejected lanes in every JSONL row",
                        "tube_intervals": tube, "last_saved_endpoint": endpoints[-1]}],
            "qualification": "per-step axis-aligned union; no per-lane geometry or octagon; target only at t=5; not an end-to-end certificate"}


def _matlab_vector(values: list[float]) -> str:
    return "[" + " ".join(format(value, ".17g") for value in values) + "]"


def _write_matlab(data: dict, path: Path) -> None:
    h, steps = data["step_size"], data["expected_steps"]
    lines = ["% Saved ARCH-COMP26 interval visualization. No numerical solver is run.",
             "% Source paths and qualification are recorded in the adjacent .render.json.",
             "figure('Color','w'); hold on; grid on;", "t = " + _matlab_vector([step * h for step in range(steps + 1)]) + ";"]
    for index, item in enumerate(data["series"], 1):
        lo = [row[0] for row in item["tube_intervals"]]
        hi = [row[1] for row in item["tube_intervals"]]
        lines += [f"lo{index} = {_matlab_vector(lo)};", f"hi{index} = {_matlab_vector(hi)};",
                  f"c{index} = {_matlab_vector([int(COLORS[index-1][k:k+2],16)/255 for k in (1,3,5)])};",
                  f"for k = 1:{steps}",
                  f"  patch([t(k) t(k+1) t(k+1) t(k)], [lo{index}(k) lo{index}(k) hi{index}(k) hi{index}(k)], c{index}, 'FaceAlpha', 0.08, 'EdgeColor', 'none');",
                  "end",
                  f"stairs(t, [lo{index} lo{index}(end)], 'Color', c{index}, 'LineWidth', 1.1, 'DisplayName', '{item['label']} lower');",
                  f"stairs(t, [hi{index} hi{index}(end)], 'Color', c{index}, 'LineWidth', 1.1, 'HandleVisibility', 'off');"]
    if data["plot"].startswith("acc-"):
        lines += [f"plot([0 {steps*h:.17g}], [0 0], 'k--', 'LineWidth', 1.2, 'DisplayName', 'Safe margin = 0');",
                  "ylabel('x_{lead} - x_{ego} - 1.4 v_{ego} - 10');"]
    else:
        lo, hi = data["target_endpoint_interval"]
        lines += [f"plot([{steps*h:.17g} {steps*h:.17g}], [{lo:.17g} {hi:.17g}], 'm-', 'LineWidth', 4, 'DisplayName', 'T=5 endpoint target');",
                  "ylabel('x_3');"]
    lo, hi = data["initial_interval"]
    lines += [f"plot([0 0], [{lo:.17g} {hi:.17g}], 'ko-', 'LineWidth', 2, 'DisplayName', 'Initial interval');",
              "xlabel('t (s)');", "legend('Location', 'best');", "xlim([0 t(end)]);",
              f"title('{data['contract']}');", "hold off;"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render(data: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, steps = data["step_size"], data["expected_steps"]
    edges = [index * h for index in range(steps + 1)]
    fig, ax = plt.subplots(figsize=(11.8, 6.6))
    for index, item in enumerate(data["series"]):
        lo = [row[0] for row in item["tube_intervals"]]
        hi = [row[1] for row in item["tube_intervals"]]
        color = COLORS[index]
        label = item["label"]
        ax.fill_between(edges, lo + lo[-1:], hi + hi[-1:], step="post",
                        color=color, alpha=0.075)
        style = "--" if label == "Xiangru" else "-"
        ax.step(edges, lo + lo[-1:], where="post", color=color, linewidth=1.1,
                linestyle=style, label=label)
        ax.step(edges, hi + hi[-1:], where="post", color=color, linewidth=1.1,
                linestyle=style)
    init_lo, init_hi = data["initial_interval"]
    ax.plot([0, 0], [init_lo, init_hi], color="black", marker="o", linewidth=2,
            label="Initial interval")
    if data["plot"].startswith("acc-"):
        ax.axhline(0, color="black", linestyle=":", linewidth=1.5, label="Safe margin = 0")
        ax.set_ylabel(r"$x_{lead}-x_{ego}-1.4v_{ego}-10$")
        ax.set_ylim(bottom=min(0, min(item["min_lower"] for item in data["series"])) - 1)
        overlap = (" Huan/Xiangru plotted bounds coincide." if
                   data["series"][1]["tube_intervals"] == data["series"][2]["tube_intervals"] else "")
        footer = ("Saved tube boxes; exact 1.4 coefficient, outward-rounded box image; coordinate correlations unavailable.\n"
                  "Native acceptance absent from ranges.bin; Huan/Xiangru/P3 accepted=true in 50 saved rows. Adjacent results unbound." + overlap + "\n"
                  "Sources: acc_native_var_tail_full50_001/ranges.bin; acc_huan_full50_001, acc_xiangru_full50_001, acc_p3_full50_001/ranges.jsonl.")
    else:
        lo, hi = data["target_endpoint_interval"]
        ax.plot([steps*h, steps*h], [lo, hi], color="#b2188b", linewidth=5,
                solid_capstyle="round", label="T=5 endpoint target")
        ax.set_ylabel(r"$x_3$")
        footer = ("P3 JSONL records one 1024-box union per step, not per-lane boxes or an octagon.\n"
                  "All 1024 lanes accepted in saved rows; [0.94,1.06] is the endpoint target at t=5 only. No end-to-end certificate.\n"
                  "Source: quad_paper_p3_nohash_v1/full50_001/data/observations.jsonl; saved endpoint differs slightly from RESULT final_hull.")
    ax.set_xlim(0, steps*h)
    ax.set_xlabel("t (s)")
    ax.set_title(data["contract"] + " — saved tube")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.subplots_adjust(bottom=0.23, left=0.09, right=0.97, top=0.91)
    fig.text(0.09, 0.07, footer, fontsize=7.6, va="bottom")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="plot", required=True)
    acc = sub.add_parser("acc-margin", help="four saved ACC tube sources")
    for name in ("native", "huan", "xiangru", "p3"):
        acc.add_argument(f"--{name}", required=True, type=Path)
    acc.add_argument("--output", required=True, type=Path)
    quad = sub.add_parser("quad-p3-x3", help="P3 pooled QUAD JSONL x3 tube")
    quad.add_argument("--observations", required=True, type=Path)
    quad.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    data = _acc(args) if args.plot == "acc-margin" else _quad(args)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry = output.with_suffix(".geometry.json")
    geometry.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _render(data, output)
    artifacts = {suffix: _source(output.with_suffix(suffix))
                 for suffix in (".geometry.json", ".png", ".pdf")}
    summary = ({"minimum_tube_margin_lower_by_method":
                {item["label"]: item["min_lower"] for item in data["series"]}}
               if data["plot"].startswith("acc-") else
               {"last_saved_endpoint_x3": data["series"][0]["last_saved_endpoint"],
                "last_saved_tube_x3": data["series"][0]["tube_intervals"][-1]})
    receipt = {"schema": "archcomp26-saved-interval-render-v1",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "plot": data["plot"], "content_digest_policy": "none computed",
               "sources": [item["source"] for item in data["series"]],
               "summary": summary,
               "plot_configuration": {"step_size": data["step_size"], "expected_steps": data["expected_steps"],
                                      "view": data["view"], "qualification": data["qualification"]},
               "artifacts": artifacts,
               "matlab_files_generated": 0}
    output.with_suffix(".render.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
