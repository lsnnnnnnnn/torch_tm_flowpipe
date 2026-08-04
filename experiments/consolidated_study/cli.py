#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe.protocol.config import expected_configuration_rows
from torch_tm_flowpipe.protocol.backend_identity import (
    FlowstarBackendIdentity,
    inspect_primary_flowstar_backend,
)
from torch_tm_flowpipe.protocol.provenance import prepare_output_directory
from torch_tm_flowpipe.protocol.gates import validate_cross_tool_gate_manifest
from torch_tm_flowpipe.protocol.schema import (
    RUNTIME_BOUNDARY_VERSION,
    SCHEMA_VERSION,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(
    command: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    process = _run(["git", *arguments], cwd=cwd)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "git command failed")
    return process.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    required_fields: Sequence[str] = (),
) -> None:
    values = [dict(row) for row in rows]
    fields = sorted(
        {field for row in values for field in row} | set(required_fields)
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(values)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value


def _repository_state(path: Path) -> dict[str, Any]:
    status = _git("status", "--porcelain=v2", cwd=path)
    diff = _run(["git", "diff", "--binary"], cwd=path)
    return {
        "path": str(path),
        "remote": _git("remote", "get-url", "origin", cwd=path),
        "sha": _git("rev-parse", "HEAD", cwd=path),
        "dirty": bool(status),
        "status_porcelain_v2": status,
        "dirty_patch_sha256": hashlib.sha256(
            diff.stdout.encode("utf-8")
        ).hexdigest(),
    }


def _resolve_dependencies() -> dict[str, Path]:
    work_parent = REPO_ROOT.parent
    defaults = {
        "diffreach": work_parent / "DiffReach",
        "flowstar": work_parent / "flowstar",
    }
    environment_names = {
        "diffreach": "DIFFREACH_ROOT",
        "flowstar": "FLOWSTAR_ROOT",
    }
    resolved = {
        name: Path(os.environ.get(environment_names[name], default))
        .expanduser()
        .resolve()
        for name, default in defaults.items()
    }
    missing = [
        f"{name}={path}"
        for name, path in resolved.items()
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "missing pinned comparison dependency: " + ", ".join(missing)
        )
    return resolved


def _resolve_diffreach_python() -> Path:
    candidates = [
        os.environ.get("DIFFREACH_PYTHON", ""),
        str(REPO_ROOT.parent / "work" / "diffreach312" / "bin" / "python"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            # Preserve the virtual-environment launcher path. Resolving its
            # symlink would bypass the environment and lose pinned packages.
            return Path(candidate).expanduser().absolute()
    raise FileNotFoundError(
        "set DIFFREACH_PYTHON to the isolated Python containing pinned JAX"
    )


def _run_with_log(
    command: Sequence[str],
    log_path: Path,
    *,
    environment: Mapping[str, str],
) -> None:
    process = _run(command, environment=environment)
    log_path.write_text(
        "command="
        + " ".join(command)
        + f"\nexit_code={process.returncode}\n\n[stdout]\n"
        + process.stdout
        + "\n[stderr]\n"
        + process.stderr,
        encoding="utf-8",
    )
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}); see {log_path}"
        )


def _run_test_matrix(
    commands: Sequence[Sequence[str]],
    log_path: Path,
    *,
    environment: Mapping[str, str],
) -> None:
    sections: list[str] = []
    failed = False
    for index, command in enumerate(commands, start=1):
        process = _run(command, environment=environment)
        sections.append(
            f"[test_command_{index}]\n"
            + "command="
            + " ".join(command)
            + f"\nexit_code={process.returncode}\n\n[stdout]\n"
            + process.stdout
            + "\n[stderr]\n"
            + process.stderr
        )
        failed = failed or process.returncode != 0
    log_path.write_text("\n\n".join(sections), encoding="utf-8")
    if failed:
        raise RuntimeError(f"test matrix failed; see {log_path}")


def _merge_observations(output: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        for row in _read_csv(output / f"pareto_repetitions_{tool}.csv"):
            rows.append({"observation_source": tool, **row})
    return rows


def _standardize_tables(output: Path) -> dict[str, int]:
    raw = _merge_observations(output)
    summary = _read_csv(output / "runtime_summary.csv")
    primary = _read_csv(output / "native_pareto_summary.csv")
    excluded = _read_csv(output / "native_pareto_excluded.csv")
    exploratory = _read_csv(output / "EXPLORATORY.csv")
    failures = [
        row
        for row in summary
        if row.get("failure_category") != "completed"
        or row.get("completed_requested_horizon", "").lower() != "true"
    ]
    _write_csv(output / "RAW_OBSERVATIONS.csv", raw)
    _write_csv(output / "SUMMARY.csv", summary)
    _write_csv(output / "FAILURES.csv", failures)
    _write_csv(output / "ELIGIBILITY.csv", [*primary, *excluded])
    _write_csv(output / "PRIMARY_PARETO.csv", primary)
    _write_csv(output / "EXPLORATORY.csv", exploratory)
    return {
        "raw_observations": len(raw),
        "summary_rows": len(summary),
        "failure_rows": len(failures),
        "eligible_rows": len(primary),
        "excluded_rows": len(excluded),
        "exploratory_rows": len(exploratory),
    }


def _make_figure(output: Path, code_sha: str) -> dict[str, Any]:
    matplotlib_cache = output / ".matplotlib-cache"
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary_path = output / "PRIMARY_PARETO.csv"
    summary_path = output / "SUMMARY.csv"
    primary = _read_csv(primary_path)
    rows = primary or _read_csv(summary_path)
    source = primary_path if primary else summary_path
    figure_dir = output / "figures"
    figure_dir.mkdir()
    figure_path = figure_dir / "width_runtime.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for tool in sorted({row["tool"] for row in rows}):
        selected = [row for row in rows if row["tool"] == tool]
        x = [float(row["steady_total_configuration_time_s"]) for row in selected]
        y = [float(row["width_at_evaluation_time"]) for row in selected]
        axis.scatter(x, y, label=tool)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("steady total configuration time (s)")
    axis.set_ylabel("raw endpoint maximum width")
    axis.set_title(
        "Formal primary Pareto"
        if primary
        else "Smoke diagnostics (not authoritative)"
    )
    if rows:
        axis.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    shutil.rmtree(matplotlib_cache, ignore_errors=True)
    manifest = {
        "figure": str(figure_path.relative_to(output)),
        "source_files": source.name,
        "source_sha256": _sha256(source),
        "filters": (
            "primary_numerical_eligible == True"
            if primary
            else "smoke summary; no headline inference"
        ),
        "grouping": "tool",
        "x": "steady_total_configuration_time_s",
        "y": "width_at_evaluation_time",
        "series": "tool",
        "row_count": len(rows),
        "generator_commit": code_sha,
    }
    _write_csv(output / "FIGURE_MANIFEST.csv", [manifest])
    return manifest


def _write_report(
    output: Path,
    *,
    run_id: str,
    profile_name: str,
    code_sha: str,
    counts: Mapping[str, int],
) -> None:
    status = (
        "candidate_pending_independent_audit"
        if profile_name == "formal"
        else "smoke_non_authoritative"
    )
    (output / "REPORT.md").write_text(
        "\n".join(
            [
                "# Consolidated three-tool run",
                "",
                f"- run_id: `{run_id}`",
                f"- profile: `{profile_name}`",
                f"- status: `{status}`",
                f"- code_sha: `{code_sha}`",
                f"- runtime_boundary_version: `{RUNTIME_BOUNDARY_VERSION}`",
                f"- schema_version: `{SCHEMA_VERSION}`",
                f"- raw_observations: {counts['raw_observations']}",
                f"- summary_rows: {counts['summary_rows']}",
                f"- eligible_primary_rows: {counts['eligible_rows']}",
                f"- excluded_rows: {counts['excluded_rows']}",
                f"- failure_rows: {counts['failure_rows']}",
                "",
                "Only `PRIMARY_PARETO.csv` is eligible for headline claims. "
                "Smoke results and `EXPLORATORY.csv` are non-authoritative.",
                "",
                f"`SUMMARY.csv` SHA-256: `{_sha256(output / 'SUMMARY.csv')}`",
                f"`PRIMARY_PARETO.csv` SHA-256: "
                f"`{_sha256(output / 'PRIMARY_PARETO.csv')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _environment_record(
    dependencies: Mapping[str, Path],
    diffreach_python: Path,
    flowstar_identity: FlowstarBackendIdentity,
) -> dict[str, Any]:
    package_probe = _run(
        [sys.executable, "-m", "pip", "freeze"],
    )
    jax_probe = _run(
        [
            str(diffreach_python),
            "-c",
            (
                "import json,jax,jaxlib,platform;"
                "print(json.dumps({'python':platform.python_version(),"
                "'jax':jax.__version__,'jaxlib':jaxlib.__version__,"
                "'devices':[str(x) for x in jax.devices()]}))"
            ),
        ]
    )
    return {
        "captured_utc": _utc_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": package_probe.stdout.splitlines(),
        "jax_probe": (
            json.loads(jax_probe.stdout)
            if jax_probe.returncode == 0
            else {
                "returncode": jax_probe.returncode,
                "stderr": jax_probe.stderr,
            }
        ),
        "cuda_available": False,
        "cuda_skip_reason": (
            "Apple Silicon host has no NVIDIA CUDA device"
        ),
        "repositories": {
            "torch_tm_flowpipe": _repository_state(REPO_ROOT),
            **{
                name: _repository_state(path)
                for name, path in dependencies.items()
            },
        },
        "flowstar_backend_identity": flowstar_identity.to_record(),
    }


def _load_cross_tool_gates(profile: Mapping[str, Any]) -> dict[str, Any]:
    relative = profile.get("cross_tool_gate_manifest")
    if not relative:
        return {"status": "not_required_for_non_authoritative_profile", "gates": {}}
    path = REPO_ROOT / "benchmarks" / str(relative)
    value = _load_yaml(path)
    gates = value.get("gates")
    if not isinstance(gates, dict) or not gates:
        raise ValueError(f"{path} must define non-empty gates")
    value["path"] = str(path.relative_to(REPO_ROOT))
    value["sha256"] = _sha256(path)
    return value


def _require_cross_tool_gates(gates: Mapping[str, Any]) -> None:
    decision = validate_cross_tool_gate_manifest(gates, repo_root=REPO_ROOT)
    if decision.errors:
        raise RuntimeError(
            "formal comparison gate manifest is invalid: "
            + "; ".join(decision.errors)
        )
    if decision.pending:
        raise RuntimeError(
            "formal comparison is blocked by unverified gates: "
            + ", ".join(decision.pending)
        )


def _write_sha256s(output: Path) -> None:
    excluded = {
        "SHA256SUMS",
        "INDEPENDENT_AUDIT.json",
        "final_acceptance.json",
    }
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append(
                f"{_sha256(path)}  {path.relative_to(output)}"
            )
    (output / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def _profile_path(name: str) -> Path:
    return REPO_ROOT / "benchmarks" / f"{name}.yaml"


def _remove_temporary_builds(output: Path) -> list[str]:
    """Remove reproducible native executables while retaining source and logs."""
    removed: list[str] = []
    flowstar_logs = output / "logs" / "flowstar"
    if not flowstar_logs.is_dir():
        return removed
    for source in sorted(flowstar_logs.rglob("*.cpp")):
        executable = source.with_suffix("")
        if executable.is_file():
            removed.append(str(executable.relative_to(output)))
            executable.unlink()
    return removed


def execute(profile_name: str, output: Path) -> Path:
    profile = _load_yaml(_profile_path(profile_name))
    benchmark_path = REPO_ROOT / "benchmarks" / profile["canonical_spec"]
    benchmark = _load_yaml(benchmark_path)
    code_sha = _git("rev-parse", "HEAD")
    initial_status = _git("status", "--porcelain=v2")
    if profile_name == "formal" and initial_status:
        raise RuntimeError(
            "formal run requires a clean code-freeze worktree"
        )

    dependencies = _resolve_dependencies()
    flowstar_identity = inspect_primary_flowstar_backend(
        dependencies["flowstar"], environment=os.environ
    )
    diffreach_python = _resolve_diffreach_python()
    cross_tool_gates = _load_cross_tool_gates(profile)
    if profile_name == "formal":
        _require_cross_tool_gates(cross_tool_gates)

    output_preexisted = output.exists()
    prepare_output_directory(output)
    run_id = output.name
    logs = output / "logs"
    logs.mkdir()
    expected = expected_configuration_rows(benchmark, profile)
    run_manifest = {
        "run_id": run_id,
        "profile": profile_name,
        "started_utc": _utc_now(),
        "code_sha": code_sha,
        "code_freeze_sha": code_sha,
        "output_preexisted": output_preexisted,
        "output_directory_empty_at_start": True,
        "schema_version": SCHEMA_VERSION,
        "runtime_boundary_version": RUNTIME_BOUNDARY_VERSION,
        "status": "running",
        "authoritative_requested": bool(profile["authoritative"]),
    }
    _write_json(output / "RUN_MANIFEST.json", run_manifest)
    _write_json(
        output / "CONFIG_MANIFEST.json",
        {
            "profile": profile,
            "profile_sha256": _sha256(_profile_path(profile_name)),
            "benchmark_sha256": _sha256(benchmark_path),
            "required_repetitions": int(
                profile["runtime_repetitions"]
            ),
            "expected_configuration_count": len(expected),
            "expected_configurations": expected,
        },
    )

    environment = os.environ.copy()
    environment.update(
        {
            "TORCH_REPO_ROOT": str(REPO_ROOT),
            "TORCH_REPAIRED_ROOT": str(REPO_ROOT),
            "DIFFREACH_ROOT": str(dependencies["diffreach"]),
            "FLOWSTAR_ROOT": str(dependencies["flowstar"]),
            "FLOWSTAR_BACKEND_CLASS": flowstar_identity.backend_class,
            "FLOWSTAR_BACKEND_SHA": flowstar_identity.repository_sha,
            "FLOWSTAR_BACKEND_DIRTY": str(flowstar_identity.dirty).lower(),
            "FLOWSTAR_BACKEND_PRIMARY_ELIGIBLE": str(
                flowstar_identity.primary_eligible
            ).lower(),
            "FLOWSTAR_EXECUTION_ROUTE": flowstar_identity.execution_route,
            "TORCH_BACKEND_SHA": code_sha,
            "DIFFREACH_BACKEND_SHA": _git(
                "rev-parse", "HEAD", cwd=dependencies["diffreach"]
            ),
            "CROSS_TOOL_GATES_VERIFIED": str(
                profile_name == "formal"
            ).lower(),
            "FLOWSTAR_SYSTEM_INCLUDE": environment.get(
                "FLOWSTAR_SYSTEM_INCLUDE", "/opt/homebrew/include"
            ),
            "FLOWSTAR_SYSTEM_LIB": environment.get(
                "FLOWSTAR_SYSTEM_LIB", "/opt/homebrew/lib"
            ),
        }
    )
    _write_json(
        output / "ENVIRONMENT.json",
        _environment_record(
            dependencies, diffreach_python, flowstar_identity
        ),
    )
    _write_json(
        output / "PROVENANCE.json",
        {
            "captured_utc": _utc_now(),
            "code_sha": code_sha,
            "runner": str(Path(__file__).relative_to(REPO_ROOT)),
            "runner_command": sys.argv,
            "benchmark": str(benchmark_path.relative_to(REPO_ROOT)),
            "benchmark_sha256": _sha256(benchmark_path),
            "profile_sha256": _sha256(_profile_path(profile_name)),
            "runtime_boundary_version": RUNTIME_BOUNDARY_VERSION,
            "schema_version": SCHEMA_VERSION,
            "dependencies": {
                name: _repository_state(path)
                for name, path in dependencies.items()
            },
            "flowstar_backend_identity": flowstar_identity.to_record(),
            "cross_tool_gates": cross_tool_gates,
        },
    )

    test_commands: list[list[str]] = [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_protocol_contracts.py",
        ]
    ]
    if profile_name == "formal":
        test_commands = [
            [sys.executable, "-m", "pytest", "-q"],
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "experiments/three_tool_deep_study/tests",
            ],
            [
                str(diffreach_python),
                "-m",
                "pytest",
                "-q",
                (
                    "experiments/first_order_followup/tests/"
                    "test_diffreach_projection.py"
                ),
            ],
        ]
    _run_test_matrix(
        test_commands,
        output / "COMPLETE_TEST.log",
        environment=environment,
    )

    pareto_script = (
        REPO_ROOT / "experiments" / "three_tool_deep_study" / "run_pareto.py"
    )
    smoke_argument = ["--smoke"] if profile_name == "smoke" else []
    interpreters = {
        "torch": Path(sys.executable),
        "diffreach": diffreach_python,
        "flowstar": Path(sys.executable),
    }
    commands: list[list[str]] = []
    for tool in ("torch", "diffreach", "flowstar"):
        command = [
            str(interpreters[tool]),
            str(pareto_script),
            "--spec",
            str(benchmark_path),
            "--output-dir",
            str(output),
            "--tool",
            tool,
            *smoke_argument,
        ]
        commands.append(command)
        _run_with_log(
            command,
            logs / f"{tool}.log",
            environment=environment,
        )
    collect_command = [
        str(sys.executable),
        str(pareto_script),
        "--spec",
        str(benchmark_path),
        "--output-dir",
        str(output),
        "--tool",
        "collect",
        *smoke_argument,
    ]
    commands.append(collect_command)
    _run_with_log(
        collect_command,
        logs / "collect.log",
        environment=environment,
    )

    removed_temporary_builds = _remove_temporary_builds(output)
    counts = _standardize_tables(output)
    _make_figure(output, code_sha)
    _write_report(
        output,
        run_id=run_id,
        profile_name=profile_name,
        code_sha=code_sha,
        counts=counts,
    )
    audit_command = [
        str(sys.executable),
        str(REPO_ROOT / "analysis" / "independent_audit.py"),
        str(output),
    ]
    commands.append(audit_command)
    run_manifest.update(
        {
            "completed_utc": _utc_now(),
            "status": "candidate_complete",
            "runner_commands": commands,
            "test_commands": test_commands,
            "counts": counts,
        }
    )
    _write_json(output / "RUN_MANIFEST.json", run_manifest)
    provenance = json.loads(
        (output / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    provenance.update(
        {
            "runner_commands": commands,
            "test_commands": test_commands,
            "temporary_builds_removed": removed_temporary_builds,
        }
    )
    _write_json(output / "PROVENANCE.json", provenance)
    _write_sha256s(output)
    audit_process = _run(audit_command, environment=environment)
    if audit_process.returncode:
        raise RuntimeError(
            "independent acceptance failed:\n"
            + audit_process.stdout
            + audit_process.stderr
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the canonical three-tool comparison pipeline."
    )
    parser.add_argument("profile", choices=["smoke", "formal"])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output_dir
        if args.output_dir is not None
        else REPO_ROOT / "artifacts" / "runs" / run_id
    ).resolve()
    result = execute(args.profile, output)
    print(result)


if __name__ == "__main__":
    main()
