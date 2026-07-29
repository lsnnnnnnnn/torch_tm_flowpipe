#!/usr/bin/env python3
"""Collect provenance, normalized result tables, and correctness gates."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from common import (
    analytic_endpoint,
    git_sha,
    load_spec,
    sha256_manifest,
    validate_record,
    write_csv,
    write_json,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONDA = Path("/srv/local/shengenli/miniforge3/condabin/conda")


def _run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        process = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        return {
            "command": command,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _json_probe(environment: str, source: str) -> dict[str, Any]:
    result = _run(
        [str(CONDA), "run", "-n", environment, "python", "-c", source]
    )
    if result["returncode"] == 0:
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            pass
    return result


def _repo_audit(path: Path) -> dict[str, Any]:
    commands = {
        "status_short": ["git", "status", "--short"],
        "branches": ["git", "branch", "-avv"],
        "head": ["git", "rev-parse", "HEAD"],
        "log": [
            "git",
            "log",
            "--oneline",
            "--decorate",
            "--graph",
            "--all",
            "-n",
            "100",
        ],
        "remotes": ["git", "remote", "-v"],
        "worktrees": ["git", "worktree", "list"],
        "untracked": ["git", "ls-files", "--others", "--exclude-standard"],
    }
    return {
        "path": str(path),
        **{
            name: _run(command, cwd=path)
            for name, command in commands.items()
        },
    }


def _frozen_files(spec: Mapping[str, Any]) -> list[Path]:
    paths: list[Path] = []
    for relative in spec["frozen_inputs"]:
        root = REPO_ROOT / relative
        if not root.exists():
            continue
        paths.extend(path for path in root.rglob("*") if path.is_file())
    return paths


def collect_environment(
    spec: Mapping[str, Any], output: Path, *, initialize: bool
) -> dict[str, Any]:
    repositories = {
        name: Path(path)
        for name, path in spec["repositories"].items()
    }
    torch_probe = _json_probe(
        str(spec["environments"]["torch"]),
        (
            "import json,platform,torch;"
            "print(json.dumps({'python':platform.python_version(),"
            "'torch':torch.__version__,'cuda_version':torch.version.cuda,"
            "'cuda_available':torch.cuda.is_available(),"
            "'cuda_devices':torch.cuda.device_count(),"
            "'default_dtype':str(torch.get_default_dtype())}))"
        ),
    )
    jax_probe = _json_probe(
        str(spec["environments"]["diffreach"]),
        (
            "import json,platform,jax,jaxlib;"
            "jax.config.update('jax_enable_x64',True);"
            "print(json.dumps({'python':platform.python_version(),"
            "'jax':jax.__version__,'jaxlib':jaxlib.__version__,"
            "'x64':bool(jax.config.jax_enable_x64),"
            "'devices':[str(x) for x in jax.devices()]}))"
        ),
    )
    cpu_model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.startswith("model name")
            ),
            "",
        )
    environment = {
        "date_utc": _run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"])[
            "stdout"
        ].strip(),
        "host_python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "torch_environment": spec["environments"]["torch"],
        "diffreach_environment": spec["environments"]["diffreach"],
        "flowstar_environment": spec["environments"]["flowstar"],
        "torch_probe": torch_probe,
        "jax_probe": jax_probe,
        "gcc": _run(["gcc", "--version"]),
        "gxx": _run(["g++", "--version"]),
        "mpfr": _run(["pkg-config", "--modversion", "mpfr"]),
        "gmp": _run(["pkg-config", "--modversion", "gmp"]),
        "tmux": _run(["tmux", "-V"]),
        "repositories": {
            name: {
                "path": str(path),
                "sha": git_sha(path),
            }
            for name, path in repositories.items()
        },
    }
    write_json(output / "environment.json", environment)
    audit = {
        name: _repo_audit(path)
        for name, path in repositories.items()
    }
    write_json(output / "repository_audit.json", audit)
    manifest = sha256_manifest(_frozen_files(spec), REPO_ROOT)
    if initialize:
        write_json(
            output / "frozen_input_manifest_initial.json", manifest
        )
    else:
        initial_path = output / "frozen_input_manifest_initial.json"
        initial = (
            json.loads(initial_path.read_text(encoding="utf-8"))
            if initial_path.exists()
            else []
        )
        integrity = {
            "initial_file_count": len(initial),
            "final_file_count": len(manifest),
            "unchanged": initial == manifest,
            "added_or_changed": [
                row for row in manifest if row not in initial
            ],
            "removed_or_changed": [
                row for row in initial if row not in manifest
            ],
        }
        write_json(output / "frozen_input_manifest_final.json", manifest)
        write_json(output / "frozen_input_integrity.json", integrity)
        if initial and not integrity["unchanged"]:
            raise RuntimeError("a frozen input artifact changed")
    return environment


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _truth(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "passed", "1"}:
        return True
    if text in {"false", "failed", "0"}:
        return False
    return None


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _last_rows(
    rows: Iterable[Mapping[str, Any]], protocol: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("protocol") != protocol:
            continue
        key = tuple(
            str(row.get(field, ""))
            for field in (
                "tool",
                "variant",
                "system",
                "h",
                "state_index",
                "interval_kind",
            )
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for values in grouped.values():
        maximum = max(_number(row.get("step_index")) for row in values)
        result.extend(
            dict(row)
            for row in values
            if _number(row.get("step_index")) == maximum
        )
    return result


def _one_step_summary(
    spec: Mapping[str, Any], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("protocol") != "one_step_common_input":
            continue
        item = dict(row)
        exact = analytic_endpoint(
            str(row.get("system")),
            spec["systems"][str(row.get("system"))]["initial_box"],
            _number(row.get("time")),
        )
        state = int(_number(row.get("state_index")))
        if exact is not None:
            exact_width = exact[state][1] - exact[state][0]
            item["analytic_exact_width"] = exact_width
            item["exact_inflation_ratio"] = (
                _number(row.get("width")) / exact_width
                if exact_width
                else ""
            )
        else:
            item["analytic_exact_width"] = ""
            item["exact_inflation_ratio"] = ""
        result.append(item)
    return result


def collect_tables(
    spec: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    controlled = _read_csv(output / "controlled_raw.csv")
    native = _read_csv(output / "native_raw.csv")
    repetitions: list[dict[str, Any]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        repetitions.extend(
            _read_csv(output / f"pareto_repetitions_{tool}.csv")
        )
    correctness_rows = _read_csv(output / "flowstar_correctness.csv")
    flowstar_ablation = _read_csv(
        output / "flowstar_component_ablation.csv"
    )
    raw: list[dict[str, Any]] = []
    for source, values in (
        ("controlled", controlled),
        ("native", native),
        ("runtime_repetition", repetitions),
        ("flowstar_correctness", correctness_rows),
        ("flowstar_ablation", flowstar_ablation),
    ):
        raw.extend({"result_source": source, **row} for row in values)
    write_csv(output / "raw_results.csv", raw)

    one_step = _one_step_summary(spec, controlled)
    affine = _last_rows(controlled, "common_affine_carry")
    box = _last_rows(controlled, "common_box_carry")
    native_low = _last_rows(native, "native_low_order")
    write_csv(output / "one_step_summary.csv", one_step)
    write_csv(output / "affine_carry_summary.csv", affine)
    write_csv(output / "box_carry_summary.csv", box)
    write_csv(output / "native_low_order_summary.csv", native_low)

    failures: list[dict[str, Any]] = []
    for row in raw:
        validation = _truth(
            row.get(
                "native_validation_passed",
                row.get("native_validation_status", ""),
            )
        )
        category = row.get(
            "failure_category", row.get("failure_message", "")
        )
        if validation is False or category:
            failures.append(
                {
                    "result_source": row.get("result_source", ""),
                    "tool": row.get("tool", ""),
                    "variant": row.get("variant", ""),
                    "protocol": row.get("protocol", ""),
                    "system": row.get("system", ""),
                    "h": row.get("h", ""),
                    "step_index": row.get("step_index", ""),
                    "failure_category": category
                    or "native_validation_failure",
                    "message": row.get(
                        "message", row.get("failure_message", "")
                    ),
                }
            )
    write_csv(output / "failure_summary.csv", failures)

    representation_records: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
    for path in sorted((output / "common_segments").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (
            record["tool"],
            record["variant"],
            record["system"],
            float(record["h"]),
            record.get("native_metadata", {}).get("requested_order"),
        )
        representation_records[key] = (path, record)
    representation_checks = []
    for path, record in representation_records.values():
        representation_checks.append(
            {
                "path": str(path),
                "tool": record["tool"],
                "system": record["system"],
                **validate_record(record),
            }
        )
    analytic_rows = [
        row
        for row in controlled + native
        if str(row.get("analytic_reference_contained", "")).strip()
    ]
    native_rows = [
        row
        for row in controlled + native
        if str(row.get("native_validation_passed", "")).strip()
    ]
    flowstar_summary_path = output / "flowstar_correctness_summary.json"
    flowstar_summary = (
        json.loads(flowstar_summary_path.read_text(encoding="utf-8"))
        if flowstar_summary_path.exists()
        else {}
    )
    frozen_path = output / "frozen_input_integrity.json"
    frozen = (
        json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_path.exists()
        else {}
    )
    correctness = {
        "controlled_and_native_rows_checked": len(controlled) + len(native),
        "native_validation_checks": len(native_rows),
        "native_validation_failures": sum(
            _truth(row.get("native_validation_passed")) is False
            for row in native_rows
        ),
        "analytic_checks": len(analytic_rows),
        "analytic_violations": sum(
            _truth(row.get("analytic_reference_contained")) is False
            for row in analytic_rows
        ),
        "common_segment_records": len(representation_checks),
        "common_segment_point_checks": sum(
            int(row["point_evaluation_checks"])
            for row in representation_checks
        ),
        "common_segment_representation_failures": sum(
            not row["passed"] for row in representation_checks
        ),
        "common_segment_endpoint_tube_violations": sum(
            int(row["endpoint_vs_tube_violations"])
            for row in representation_checks
        ),
        "flowstar": flowstar_summary,
        "frozen_inputs": frozen,
        "failure_rows": len(failures),
    }
    correctness["primary_gates_passed"] = bool(
        correctness["native_validation_failures"] == 0
        and correctness["analytic_violations"] == 0
        and correctness["common_segment_representation_failures"] == 0
        and correctness["common_segment_endpoint_tube_violations"] == 0
        and flowstar_summary.get("passed", False)
        and frozen.get("unchanged", False)
    )
    write_json(output / "correctness_checks.json", correctness)
    return {
        "raw_rows": len(raw),
        "one_step_rows": len(one_step),
        "affine_rows": len(affine),
        "box_rows": len(box),
        "native_low_order_rows": len(native_low),
        "failure_rows": len(failures),
        "primary_gates_passed": correctness["primary_gates_passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["initialize", "finalize", "tables", "all"],
        required=True,
    )
    args = parser.parse_args()
    spec = load_spec(args.spec)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    if args.mode in {"initialize", "all"}:
        result["environment_initial"] = collect_environment(
            spec, output, initialize=True
        )
    if args.mode in {"finalize", "all"}:
        result["environment_final"] = collect_environment(
            spec, output, initialize=False
        )
    if args.mode in {"tables", "all"}:
        result["tables"] = collect_tables(spec, output)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
