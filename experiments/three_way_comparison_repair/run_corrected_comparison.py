#!/usr/bin/env python3
"""Orchestrate environment capture, historical reproduction, and repaired runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from common import (
    HERE,
    REPO_ROOT,
    git_sha,
    load_spec,
    manifest_digest,
    sha256_manifest,
    timestamp,
    write_json,
)


def _command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    process = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _environment() -> dict[str, Any]:
    commands = {
        "conda_env_list": ["conda", "env", "list"],
        "py11": [
            "conda",
            "run",
            "-n",
            "py11",
            "python",
            "-c",
            "import json,sys,torch; print(json.dumps({'python':sys.version,'torch':torch.__version__,'default_dtype':str(torch.get_default_dtype()),'cuda_available':torch.cuda.is_available()}))",
        ],
        "diffreach312": [
            "conda",
            "run",
            "-n",
            "diffreach312",
            "python",
            "-c",
            "import json,sys,jax,jaxlib; print(json.dumps({'python':sys.version,'jax':jax.__version__,'jaxlib':jaxlib.__version__,'jax_x64':bool(jax.config.read('jax_enable_x64')),'devices':[str(x) for x in jax.devices()]}))",
        ],
        "gcc": ["gcc", "--version"],
        "gxx": ["g++", "--version"],
        "mpfr": ["pkg-config", "--modversion", "mpfr"],
        "gmp": ["pkg-config", "--modversion", "gmp"],
        "nvidia_smi": ["nvidia-smi"],
        "lscpu": ["lscpu"],
        "memory": ["free", "-b"],
        "tmux_path": ["bash", "-c", "command -v tmux"],
        "tmux_version": ["tmux", "-V"],
    }
    return {
        "captured_at": timestamp(),
        "timezone": "Etc/UTC",
        "commands": {name: _command(command) for name, command in commands.items()},
    }


def _repository_state(spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    paths = {
        "torch_repair": Path(spec["repositories"]["torch"]),
        "diffreach": Path(spec["repositories"]["diffreach"]),
        "flowstar_original": Path(spec["repositories"]["flowstar_original"]),
        "flowstar_audit": Path(spec["repositories"]["flowstar_audit"]),
    }
    commands = [
        ["git", "status", "--short", "--untracked-files=all"],
        ["git", "branch", "-avv"],
        ["git", "rev-parse", "HEAD"],
        ["git", "log", "--oneline", "--decorate", "--graph", "--all", "-n", "100"],
        ["git", "remote", "-v"],
        ["git", "worktree", "list", "--porcelain"],
    ]
    chunks = [
        "PRE-EDIT BASE SNAPSHOT",
        "Torch base branch: codex/three-way-common-contract-comparison",
        "Torch base SHA: 7251adfe8d2f3a5f3fd7a4a89f4b5a2075a19b10",
        "DiffReach SHA: dd628eb443b517d6415de93e7035b4baef73963e",
        "Flow* SHA: b85a3211748cb77b736fe4ad42ee02d8d2b81148",
        "The Torch and DiffReach tracked worktrees were clean. The original Flow* "
        "checkout had pre-existing untracked build and plot artifacts; it was not edited.",
        "",
        "EXECUTION-TIME REPOSITORY STATE",
    ]
    summary: dict[str, Any] = {
        "torch_base_branch": "codex/three-way-common-contract-comparison",
        "torch_base_sha": "7251adfe8d2f3a5f3fd7a4a89f4b5a2075a19b10",
    }
    for name, path in paths.items():
        chunks.extend(["", f"## {name}: {path}"])
        summary[name] = {
            "path": str(path),
            "sha": git_sha(path),
        }
        for command in commands:
            result = _command(command, path)
            chunks.extend(
                [
                    "",
                    "$ " + " ".join(command),
                    result["stdout"],
                    result["stderr"],
                ]
            )
    return "\n".join(chunks), summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _historical_reproduction(output: Path) -> dict[str, Any]:
    frozen = (
        REPO_ROOT
        / "experiments"
        / "three_way_common_contract"
        / "results"
        / "20260724T132534Z"
    )
    reproduction = output / "historical_reproduction"
    reproduction.mkdir(parents=True, exist_ok=True)
    for path in frozen.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json", ".yaml"}:
            shutil.copy2(path, reproduction / path.name)
    old_here = REPO_ROOT / "experiments" / "three_way_common_contract"
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(reproduction / ".matplotlib")
    plot = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "py11",
            "python",
            str(old_here / "plot_results.py"),
            "--output-dir",
            str(reproduction),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    report = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "py11",
            "python",
            str(old_here / "generate_report.py"),
            "--output-dir",
            str(reproduction),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    plot_matches = 0
    plot_count = 0
    for generated in sorted((reproduction / "plots").glob("*.png")):
        historical = frozen / "plots" / generated.name
        if historical.exists():
            plot_count += 1
            plot_matches += _sha256(generated) == _sha256(historical)
    generated_report = reproduction / "three_way_common_contract_report.md"
    historical_report = frozen / "three_way_common_contract_report.md"
    report_match = (
        generated_report.exists()
        and historical_report.exists()
        and _sha256(generated_report) == _sha256(historical_report)
    )
    status = (
        "exact"
        if plot.returncode == 0
        and report.returncode == 0
        and report_match
        and plot_count == plot_matches
        else "numerically_reproduced_artifact_bytes_differ"
    )
    return {
        "status": status,
        "plot_returncode": plot.returncode,
        "report_returncode": report.returncode,
        "plot_count": plot_count,
        "plot_sha_matches": plot_matches,
        "report_sha_match": report_match,
        "plot_stderr": plot.stderr,
        "report_stderr": report.stderr,
        "reproduction_directory": str(reproduction),
    }


def _run_logged(
    command: list[str], *, log: Path, cwd: Path = REPO_ROOT
) -> None:
    print("$ " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise SystemExit(
            f"command failed with {process.returncode}; inspect {log}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default=str(HERE / "benchmark_spec.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-historical-reproduction", action="store_true")
    args = parser.parse_args()
    spec = load_spec(args.spec)
    repository_state, repository_summary = _repository_state(spec)
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else HERE / "results" / timestamp()
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    (output / "plots").mkdir()
    shutil.copy2(args.spec, output / "benchmark_spec.yaml")
    (output / "repository_state.txt").write_text(
        repository_state, encoding="utf-8"
    )
    write_json(output / "repository_summary.json", repository_summary)
    write_json(output / "environment.json", _environment())
    frozen_root = Path(spec["repositories"]["torch"]) / spec["frozen_result"]
    frozen_manifest = sha256_manifest(frozen_root)
    write_json(
        output / "frozen_manifest_before.json",
        {
            "manifest": frozen_manifest,
            "manifest_digest": manifest_digest(frozen_manifest),
        },
    )
    historical = (
        {
            "status": "skipped",
            "plot_count": 0,
            "plot_sha_matches": 0,
            "report_sha_match": False,
        }
        if args.skip_historical_reproduction
        else _historical_reproduction(output)
    )
    write_json(output / "historical_reproduction.json", historical)
    flowstar_root = Path(spec["repositories"]["flowstar_audit"])
    _run_logged(
        ["make", "-j8"],
        log=output / "logs" / "flowstar_build.log",
        cwd=flowstar_root / "flowstar-toolbox",
    )
    suffix = ["--smoke"] if args.smoke else []
    _run_logged(
        [
            "conda",
            "run",
            "-n",
            "py11",
            "python",
            str(HERE / "run_torch_audit.py"),
            "--spec",
            str(args.spec),
            "--output-dir",
            str(output),
            *suffix,
        ],
        log=output / "logs" / "torch.log",
    )
    _run_logged(
        [
            "conda",
            "run",
            "-n",
            "diffreach312",
            "python",
            str(HERE / "run_diffreach_audit.py"),
            "--spec",
            str(args.spec),
            "--output-dir",
            str(output),
            *suffix,
        ],
        log=output / "logs" / "diffreach.log",
    )
    _run_logged(
        [
            "conda",
            "run",
            "-n",
            "py11",
            "python",
            str(HERE / "run_flowstar_audit.py"),
            "--spec",
            str(args.spec),
            "--output-dir",
            str(output),
            *suffix,
        ],
        log=output / "logs" / "flowstar.log",
    )
    _run_logged(
        [
            "conda",
            "run",
            "-n",
            "py11",
            "python",
            str(HERE / "collect_results.py"),
            "--spec",
            str(args.spec),
            "--output-dir",
            str(output),
            "--strict",
        ],
        log=output / "logs" / "collect.log",
    )
    plot_environment = os.environ.copy()
    plot_environment["MPLCONFIGDIR"] = str(output / ".matplotlib")
    print("$ plot_results.py", flush=True)
    with (output / "logs" / "plot.log").open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "py11",
                "python",
                str(HERE / "plot_results.py"),
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            env=plot_environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise SystemExit(f"plot generation failed; inspect {output / 'logs' / 'plot.log'}")
    _run_logged(
        [
            "conda",
            "run",
            "-n",
            "py11",
            "python",
            str(HERE / "generate_report.py"),
            "--output-dir",
            str(output),
        ],
        log=output / "logs" / "report.log",
    )
    print(output)


if __name__ == "__main__":
    main()
