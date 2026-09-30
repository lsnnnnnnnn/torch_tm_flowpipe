"""Render the deterministic, non-final ARCH-COMP26 execution status report."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from .archcomp26_preflight import resolve_cell, validate_matrix


ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    "manifest": Path("benchmarks/archcomp26/manifest.json"),
    "matrix": Path("benchmarks/archcomp26/execution_matrix.json"),
    "contract_audits": Path(
        "benchmarks/archcomp26/evidence/contract_audits_20261001.json"
    ),
    "huan_parity": Path(
        "research/gpu_verified_20260930/report/evidence/huan_parity_campaign.json"
    ),
    "p3_audit": Path(
        "research/gpu_verified_20260930/report/evidence/quad_trig_audit.json"
    ),
    "plot_receipt": Path("docs/evidence/flowpipe_plot_validation_20261001.json"),
    "native_recheck": Path("docs/evidence/remote_native_quad_recheck_20261001.json"),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_matrix(
    manifest: Mapping[str, Any], matrix: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    errors = validate_matrix(manifest, matrix)
    if errors:
        raise ValueError("invalid execution matrix: " + "; ".join(errors))
    methods = list(manifest["methods"])
    rows: list[dict[str, Any]] = []
    run_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()
    for instance in manifest["instances"]:
        cells = {
            method: resolve_cell(matrix, instance["id"], method)
            for method in methods
        }
        run_counts.update(cell["run"]["status"] for cell in cells.values())
        support_counts.update(cell["support"]["status"] for cell in cells.values())
        rows.append({"instance": instance, "cells": cells})
    return rows, run_counts, support_counts


def collect_status(root: Path = ROOT) -> dict[str, Any]:
    paths = {name: root / relative for name, relative in INPUTS.items()}
    payload = {name: _load(path) for name, path in paths.items()}
    manifest = payload["manifest"]
    rows, run_counts, support_counts = summarize_matrix(
        manifest, payload["matrix"]
    )
    audits = payload["contract_audits"]["audits"]
    missing_audits = sorted({
        row["instance"]["id"]
        for row in rows
        if row["instance"].get("contract_audit") not in audits
    })
    if missing_audits:
        raise ValueError(f"missing contract audits: {missing_audits}")

    huan = payload["huan_parity"]
    huan_rows = huan.get("rows")
    if huan.get("status") != "completed" or not isinstance(huan_rows, list) \
            or len(huan_rows) != 5:
        raise ValueError("Huan parity evidence is not a completed five-run campaign")
    if any(
        row.get("accepted_lane_steps") != 1_024_000
        or row.get("numerical_metrics_equal") is not True
        for row in huan_rows
    ):
        raise ValueError("Huan parity rows do not share the qualified full-run result")
    timing_fields = (
        "author_elapsed_s", "driver_call_wall_s", "process_wall_s"
    )
    huan_medians = {
        field: median(float(row[field]) for row in huan_rows)
        for field in timing_fields
    }

    return {
        "manifest": manifest,
        "rows": rows,
        "run_counts": dict(sorted(run_counts.items())),
        "support_counts": dict(sorted(support_counts.items())),
        "audits": audits,
        "huan": {
            "rows": huan_rows,
            "medians": huan_medians,
            "scope": huan["scope"],
        },
        "p3": payload["p3_audit"],
        "plot": payload["plot_receipt"],
        "native": payload["native_recheck"],
        "input_identities": {
            name: {"path": str(INPUTS[name]), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def render_markdown(status: Mapping[str, Any]) -> str:
    manifest = status["manifest"]
    methods = list(manifest["methods"])
    labels = {
        "pytorch_gpu": "PyTorch/GPU",
        "huan": "Huan",
        "xiangru": "Xiangru",
        "flowstar_native": "Flow* native",
    }
    paused = manifest["execution_policy"]["experiments_paused"]
    run_counts = ", ".join(
        f"{key}={value}" for key, value in status["run_counts"].items()
    )
    support_counts = ", ".join(
        f"{key}={value}" for key, value in status["support_counts"].items()
    )
    lines = [
        "# ARCH-COMP26 four-way execution status",
        "",
        "> Automatically generated status report; this is not the final experiment report.",
        "> Historical evidence is never promoted into an unexecuted 2026 matrix cell.",
        "",
        "## Boundary",
        "",
        f"- Experiments paused: **{'yes' if paused else 'no'}**.",
        f"- Instances: {len(status['rows'])}; methods: {len(methods)}; cells: "
        f"{len(status['rows']) * len(methods)}.",
        f"- Run states: `{run_counts}`.",
        f"- Support states: `{support_counts}`.",
        f"- Resume gate: {manifest['execution_policy']['resume_gate']}",
        "",
        "## Input identities",
        "",
        "| Input | Repository-relative path | SHA-256 |",
        "|---|---|---|",
    ]
    for name, identity in status["input_identities"].items():
        lines.append(
            f"| `{name}` | `{identity['path']}` | `{identity['sha256']}` |"
        )

    lines.extend([
        "",
        "## Contract audit coverage",
        "",
        "Every row remains an unresolved execution contract. Audit status records "
        "what is known and why launch is still blocked.",
        "",
        "| Instance | Contract | Audit | Audit status | Known issues |",
        "|---|---|---|---|---:|",
    ])
    for row in status["rows"]:
        instance = row["instance"]
        audit_name = instance["contract_audit"]
        lines.append(
            f"| `{instance['id']}` | `{instance['contract']['status']}` | "
            f"`{audit_name}` | {status['audits'][audit_name]['status']} | "
            f"{len(instance['known_issues'])} |"
        )

    lines.extend([
        "",
        "## 16×4 execution matrix",
        "",
        "| Instance | " + " | ".join(labels[method] for method in methods) + " |",
        "|---|" + "---|" * len(methods),
    ])
    for row in status["rows"]:
        values = [
            f"`{row['cells'][method]['run']['status']}`" for method in methods
        ]
        lines.append(
            f"| `{row['instance']['id']}` | " + " | ".join(values) + " |"
        )

    huan = status["huan"]
    p3 = status["p3"]
    native = status["native"]["terminal_state"]
    lines.extend([
        "",
        "## Archived reference evidence (not matrix results)",
        "",
        "| Evidence | Coverage | Time | Guarantee boundary |",
        "|---|---|---:|---|",
        "| Huan QUAD parity/box | 5 complete runs; 1,024,000 accepted "
        f"lane-steps each | process median {huan['medians']['process_wall_s']:.6f} s "
        f"(driver {huan['medians']['driver_call_wall_s']:.6f} s; internal "
        f"{huan['medians']['author_elapsed_s']:.6f} s) | parity reproduction; "
        "not a strict NNCS certificate |",
        "| Current PyTorch P3 + trig reuse | "
        f"{p3['steps']['recorded_steps']} steps; "
        f"{p3['steps']['accepted_lane_steps']} accepted lane-steps | "
        f"single watchdog run {p3['watch']['process_wall_s']:.6f} s | "
        f"fullbatch qualification `{str(p3['fullbatch_qualification']).lower()}`; "
        f"end-to-end strict certificate "
        f"`{str(p3['end_to_end_strict_certificate']).lower()}` |",
        "| Native QUAD original job | "
        f"{native['complete_integration_steps']} complete steps; "
        f"{native['accepted_lane_steps']} accepted lane-steps | timeout after "
        f"{native['watch_elapsed_s']:.6f} s | no complete T=5 time or endpoint "
        "width |",
        "",
        "The Huan/P3 contracts differ in order, validation, arithmetic guarantees, "
        "instrumentation, and timing boundaries; their times are not a same-contract "
        "speedup ratio.",
        "",
        "## Plotting status",
        "",
        f"- Validation date: {status['plot']['validated_at']}.",
        f"- Saved artifact sets checked: {len(status['plot']['artifacts'])}.",
        f"- Independent geometry check: "
        f"`{status['plot']['independent_geometry_check']['result']}`.",
        f"- Experiments started by plotting validation: "
        f"`{str(status['plot']['experiments_started']).lower()}`.",
        f"- MATLAB: `{status['plot']['environment']['matlab']}`; Octave: "
        f"`{status['plot']['environment']['octave']}`. Generated `.m` files have "
        "static checks only.",
        "- Current exporter evidence covers axis-aligned box projections; it does "
        "not establish native octagon/support-direction parity.",
        "",
        "## Next gate",
        "",
        "Do not launch from this report. After explicit experiment-resume "
        "authorization, first recheck the original native QUAD job identity, PIDs, "
        "output directories, and any replacement run read-only. Never overwrite or "
        "duplicate `full1000_v1`. Then resolve a cell's contract, support, command, "
        "source/binary identity, and runtime budget before its preflight can pass.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--write", type=Path)
    output.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    report = render_markdown(collect_status())
    if args.write:
        args.write.write_text(report, encoding="utf-8")
    elif args.check:
        if not args.check.is_file() or args.check.read_text(encoding="utf-8") != report:
            print(f"stale status report: {args.check}")
            return 1
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
