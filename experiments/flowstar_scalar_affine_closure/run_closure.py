#!/usr/bin/env python3
"""Run the bounded clean-stock Flow* scalar-affine correctness closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis import (
    box,
    containment_defect,
    endpoint_corner_defect,
    first_loss_rows,
    high_precision_outward_oracle,
    monotonicity_certificate,
    parse_oracle,
    parse_trace,
    term_map,
    validate_field_separation,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SUPPLIED_STARTING_ANCHOR = "438ee68fd71fa6182eb66cac17229e20dd3cb7d3f"
STARTING_SHA = "438ee68fd71fa6182eb66cac17229e20dd3cb7d3"
FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
EXPECTED_LIB_SHA256 = "b5ff500af66354b0518cf12e7d951f4525f435e8e2d695cf84b91821992c9d9a"
EXPECTED_OFFICIAL_VDP_SHA256 = "266ba4edf9b905a185efcae4f72c28f2a9ca34362c5960e2995ea0e2bb35d51f"
DOCKER_IMAGE = "sha256:6549fefc0ae934982bf902f6a1f6ee9a2baf0def2ee763b278f914e4bbd096bf"
DOCKER_HOST = "unix:///run/user/1061/docker.sock"
PRIMARY = {
    "h": "0.01",
    "order": 4,
    "x0_lower": "0",
    "x0_upper": "0.1",
    "candidate_remainder": "0.0001",
    "cutoff": "1e-15",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


class CommandRecorder:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.path = run_root / "commands.jsonl"

    def run(
        self,
        label: str,
        argv: list[str],
        *,
        cwd: Path,
        output_dir: Path,
        environment: dict[str, str] | None = None,
        timeout: float = 300.0,
    ) -> subprocess.CompletedProcess[str]:
        output_dir.mkdir(parents=True, exist_ok=False)
        started = utc_now()
        process = subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        ended = utc_now()
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        stdout_path.write_text(process.stdout, encoding="utf-8")
        stderr_path.write_text(process.stderr, encoding="utf-8")
        record = {
            "label": label,
            "argv": argv,
            "cwd": str(cwd.resolve()),
            "start_time_utc": started,
            "end_time_utc": ended,
            "exit_code": process.returncode,
            "timeout_seconds": timeout,
            "timeout_expired": False,
            "stdout": str(stdout_path.relative_to(self.run_root)),
            "stderr": str(stderr_path.relative_to(self.run_root)),
            "stdout_sha256": sha256(stdout_path),
            "stderr_sha256": sha256(stderr_path),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        if process.returncode:
            raise RuntimeError(
                f"{label} exited {process.returncode}: {process.stderr[-2000:]}"
            )
        return process


def docker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["DOCKER_HOST"] = DOCKER_HOST
    for name in (
        "FLOWSTAR_AUDIT_TRACE",
        "FLOWSTAR_AUDIT_DISABLE_REFINEMENT",
        "FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT",
        "FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION",
        "LD_LIBRARY_PATH",
    ):
        environment.pop(name, None)
    return environment


def docker_prefix(flowstar_root: Path, run_root: Path, workdir: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{flowstar_root.resolve()}:/flowstar:ro",
        "-v",
        f"{REPO_ROOT.resolve()}:/repo:ro",
        "-v",
        f"{run_root.resolve()}:/evidence",
        "-w",
        workdir,
        DOCKER_IMAGE,
    ]


def compile_binaries(
    recorder: CommandRecorder, flowstar_root: Path, run_root: Path
) -> dict[str, Path]:
    build = run_root / "build"
    build.mkdir()
    sources = {
        "generated_stock_trace": HERE / "generated_stock_trace.cpp",
        "official_scalar_affine": HERE / "official_scalar_affine.cpp",
        "mpfr_oracle": HERE / "mpfr_oracle.cpp",
    }
    binaries: dict[str, Path] = {}
    for name, source in sources.items():
        binary = build / name
        if name == "mpfr_oracle":
            compile_args = [
                "g++",
                "-O2",
                "-std=c++11",
                f"/repo/{source.relative_to(REPO_ROOT)}",
                "-o",
                f"/evidence/build/{name}",
                "-lmpfr",
                "-lgmp",
            ]
        else:
            compile_args = [
                "g++",
                "-O3",
                "-w",
                "-fpermissive",
                "-std=c++11",
                "-I",
                "/flowstar/flowstar-toolbox",
                f"/repo/{source.relative_to(REPO_ROOT)}",
                "-L",
                "/flowstar/flowstar-toolbox",
                "-o",
                f"/evidence/build/{name}",
                "-lflowstar",
                "-lmpfr",
                "-lgmp",
                "-lgsl",
                "-lgslcblas",
                "-lm",
                "-lglpk",
            ]
        recorder.run(
            f"compile_{name}",
            docker_prefix(flowstar_root, run_root, "/evidence/build") + compile_args,
            cwd=REPO_ROOT,
            output_dir=run_root / "command_logs" / f"compile_{name}",
            environment=docker_environment(),
        )
        binaries[name] = binary
    return binaries


def container_run_argv(
    flowstar_root: Path,
    run_root: Path,
    binary: str,
    workdir: Path,
    args: list[str],
) -> list[str]:
    return docker_prefix(
        flowstar_root,
        run_root,
        f"/evidence/{workdir.relative_to(run_root)}",
    ) + [f"/evidence/build/{binary}", *args]


def trace_args(config: dict[str, Any]) -> list[str]:
    return [
        str(config["h"]),
        str(config["order"]),
        str(config["x0_lower"]),
        str(config["x0_upper"]),
        str(config["candidate_remainder"]),
        str(config["cutoff"]),
    ]


def run_trace(
    recorder: CommandRecorder,
    flowstar_root: Path,
    run_root: Path,
    *,
    label: str,
    binary: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    workdir = run_root / label
    workdir.mkdir()
    process = recorder.run(
        label,
        container_run_argv(
            flowstar_root, run_root, binary, workdir, trace_args(config)
        ),
        cwd=REPO_ROOT,
        output_dir=run_root / "command_logs" / label,
        environment=docker_environment(),
    )
    parsed = parse_trace(process.stdout)
    parsed_path = workdir / "parsed_trace.json"
    write_json(parsed_path, parsed)
    return parsed, parsed_path


def run_oracle(
    recorder: CommandRecorder,
    flowstar_root: Path,
    run_root: Path,
    *,
    label: str,
    x0_lower: str,
    x0_upper: str,
    h: str,
) -> tuple[dict[str, Any], Path]:
    workdir = run_root / label
    workdir.mkdir()
    argv = docker_prefix(
        flowstar_root, run_root, f"/evidence/{workdir.relative_to(run_root)}"
    ) + ["/evidence/build/mpfr_oracle", x0_lower, x0_upper, h]
    process = recorder.run(
        label,
        argv,
        cwd=REPO_ROOT,
        output_dir=run_root / "command_logs" / label,
        environment=docker_environment(),
    )
    parsed = parse_oracle(process.stdout)
    parsed_path = workdir / "parsed_oracle.json"
    write_json(parsed_path, parsed)
    return parsed, parsed_path


def oracle_endpoint(oracle: dict[str, Any]) -> list[float]:
    return [
        float(oracle["bounds"]["endpoint_lower"]["binary64"]),
        float(oracle["bounds"]["endpoint_upper"]["binary64"]),
    ]


def oracle_tube(oracle: dict[str, Any]) -> list[float]:
    return [
        float(oracle["bounds"]["tube_lower"]["binary64"]),
        float(oracle["bounds"]["tube_upper"]["binary64"]),
    ]


def rk4_endpoint(x0: float, h: float, substeps: int = 1000) -> float:
    value = x0
    dt = h / substeps
    for _ in range(substeps):
        k1 = 1.0 + 2.0 * value
        k2 = 1.0 + 2.0 * (value + 0.5 * dt * k1)
        k3 = 1.0 + 2.0 * (value + 0.5 * dt * k2)
        k4 = 1.0 + 2.0 * (value + dt * k3)
        value += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return value


def capture_toolchain_and_linkage(
    recorder: CommandRecorder,
    flowstar_root: Path,
    run_root: Path,
    binaries: dict[str, Path],
) -> dict[str, Any]:
    records: dict[str, Any] = {"toolchain": {}, "dynamic_linkage": {}}
    for label, command in (
        ("compiler", ["g++", "--version"]),
        ("linker", ["ld", "--version"]),
        ("dynamic_loader", ["ldd", "--version"]),
        (
            "flowstar_object_comment",
            [
                "readelf",
                "--string-dump=.comment",
                "/flowstar/flowstar-toolbox/Continuous.o",
            ],
        ),
    ):
        process = recorder.run(
            f"identity_{label}",
            docker_prefix(flowstar_root, run_root, "/evidence") + command,
            cwd=REPO_ROOT,
            output_dir=run_root / "command_logs" / f"identity_{label}",
            environment=docker_environment(),
        )
        records["toolchain"][label] = process.stdout.strip()

    for name in binaries:
        process = recorder.run(
            f"linkage_{name}",
            docker_prefix(flowstar_root, run_root, "/evidence")
            + ["ldd", f"/evidence/build/{name}"],
            cwd=REPO_ROOT,
            output_dir=run_root / "command_logs" / f"linkage_{name}",
            environment=docker_environment(),
        )
        records["dynamic_linkage"][name] = process.stdout.strip().splitlines()
    return records


def backend_identity(
    flowstar_root: Path,
    binaries: dict[str, Path],
    build_records: dict[str, Any],
) -> dict[str, Any]:
    library = flowstar_root / "flowstar-toolbox" / "libflowstar.a"
    official = flowstar_root / "benchmarks" / "continuous" / "vanderpol" / "vanderpol"
    source_status = git("status", "--porcelain=v1", cwd=flowstar_root)
    tracked_diff = git("diff", "--no-ext-diff", cwd=flowstar_root)
    source_sha = git("rev-parse", "HEAD", cwd=flowstar_root)
    library_hash = sha256(library)
    official_hash = sha256(official)
    if source_sha != FLOWSTAR_SHA:
        raise RuntimeError(f"unexpected Flow* SHA: {source_sha}")
    if tracked_diff:
        raise RuntimeError("clean-stock Flow* has a tracked diff")
    if library_hash != EXPECTED_LIB_SHA256:
        raise RuntimeError(f"unexpected clean library hash: {library_hash}")
    if official_hash != EXPECTED_OFFICIAL_VDP_SHA256:
        raise RuntimeError(f"unexpected official VDP hash: {official_hash}")
    excluded_checkout = Path("/srv/local/shengenli/flowstar")
    excluded_library = excluded_checkout / "flowstar-toolbox" / "libflowstar.a"
    return {
        "backend_identity": "clean-stock",
        "execution_routes": ["official-stock-native-api", "generated-stock diagnostic"],
        "checkout_path": str(flowstar_root.resolve()),
        "remote": git("remote", "get-url", "origin", cwd=flowstar_root),
        "source_sha": source_sha,
        "detached_head": git("branch", "--show-current", cwd=flowstar_root) == "",
        "git_status_porcelain": source_status.splitlines(),
        "tracked_diff": tracked_diff,
        "tracked_source_clean": True,
        "untracked_build_outputs_only": all(
            line.startswith("?? ")
            and (
                line.endswith((".o", ".a", ".plt"))
                or line.endswith("/vanderpol")
                or "modelParser" in line
                or "lex.yy" in line
            )
            for line in source_status.splitlines()
        ),
        "compiler": build_records["toolchain"]["compiler"].splitlines()[0],
        "linker": build_records["toolchain"]["linker"].splitlines()[0],
        "dynamic_loader": build_records["toolchain"]["dynamic_loader"].splitlines()[0],
        "flowstar_object_comment": build_records["toolchain"]["flowstar_object_comment"],
        "docker_image": DOCKER_IMAGE,
        "build_command": "make -C flowstar-toolbox -j1 && make -C benchmarks/continuous/vanderpol -j1",
        "library": {"path": str(library), "sha256": library_hash},
        "official_vdp_binary": {"path": str(official), "sha256": official_hash},
        "diagnostic_sources": {
            path.name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256(path),
            }
            for path in (
                HERE / "generated_stock_trace.cpp",
                HERE / "official_scalar_affine.cpp",
                HERE / "mpfr_oracle.cpp",
                HERE / "analysis.py",
                HERE / "run_closure.py",
            )
        },
        "diagnostic_binaries": {
            name: {
                "ephemeral_path": str(path),
                "sha256": sha256(path),
                "retained": False,
                "retention_reason": "compact committed evidence keeps hashes and linkage, not binaries",
                "dynamic_linkage": build_records["dynamic_linkage"][name],
            }
            for name, path in binaries.items()
        },
        "link_contract": {
            "library_search_path": str(library.parent),
            "argument": "-lflowstar",
            "archive_is_static": True,
            "ld_library_path_unset": True,
        },
        "patched_checkout_explicitly_excluded": {
            "path": str(excluded_checkout),
            "library_sha256": sha256(excluded_library) if excluded_library.is_file() else None,
            "used": False,
        },
        "audit_behavior_environment_variables_enabled": [],
    }


def start_state(run_root: Path, flowstar_root: Path) -> dict[str, Any]:
    return {
        "run_id": run_root.name,
        "branch_creation_observation_utc": "2026-08-04T12:38:40Z",
        "branch_creation_worktree_clean": True,
        "branch": git("branch", "--show-current"),
        "supplied_starting_anchor": SUPPLIED_STARTING_ANCHOR,
        "supplied_anchor_resolution": (
            "invalid 41-hex token with an extra trailing 'f'; resolved from the exact "
            "remote branch tip and selected worktree parent"
        ),
        "starting_sha": STARTING_SHA,
        "head_at_run_start": git("rev-parse", "HEAD"),
        "origin_main_sha": git("rev-parse", "origin/main"),
        "origin_native_reproduction_sha": git(
            "rev-parse", "origin/codex/native-reproduction-no-adapters-20260804"
        ),
        "ancestry_verified": subprocess.run(
            ["git", "merge-base", "--is-ancestor", STARTING_SHA, "HEAD"],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        == 0,
        "worktree_path": str(REPO_ROOT),
        "worktrees": git("worktree", "list", "--porcelain").splitlines(),
        "preserved_dirty_user_worktree": "/srv/local/shengenli/torch_tm_flowpipe",
        "stock_flowstar_path": str(flowstar_root.resolve()),
        "stock_flowstar_sha": git("rev-parse", "HEAD", cwd=flowstar_root),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def write_environment(path: Path) -> None:
    lines = [
        f"captured_utc={utc_now()}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"python_executable={sys.executable}",
        f"platform={platform.platform()}",
        f"docker_image={DOCKER_IMAGE}",
        "container_compiler=Ubuntu GCC 11.4.0",
    ]
    for module_name in ("torch", "jax", "numpy", "mpmath"):
        try:
            module = __import__(module_name)
            lines.append(f"{module_name}={getattr(module, '__version__', 'present')}")
        except Exception as exc:  # environment evidence, not control flow
            lines.append(f"{module_name}=unavailable:{type(exc).__name__}:{exc}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def matrix_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for h in ("0.01", "0.005", "0.0025"):
        configs.append({**PRIMARY, "id": f"step_h{h}", "h": h, "diagnostic": "step"})
    for order in (4, 5, 6):
        configs.append({**PRIMARY, "id": f"order_o{order}", "order": order, "diagnostic": "order"})
    for label, lower, upper in (
        ("lower_corner", "0", "0"),
        ("upper_corner", "0.1", "0.1"),
        ("interval", "0", "0.1"),
    ):
        configs.append(
            {
                **PRIMARY,
                "id": f"corner_{label}",
                "x0_lower": lower,
                "x0_upper": upper,
                "diagnostic": "corner",
            }
        )
    deduplicated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for config in configs:
        key = tuple(config[field] for field in ("h", "order", "x0_lower", "x0_upper"))
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = config
        else:
            existing["diagnostic"] = f"{existing['diagnostic']}+{config['diagnostic']}"
            existing["id"] = f"{existing['id']}__{config['id']}"
    return list(deduplicated.values())


def primary_comparison(run_root: Path, first_label: str, second_label: str) -> dict[str, Any]:
    first_stdout = (run_root / "command_logs" / first_label / "stdout.log").read_bytes()
    second_stdout = (run_root / "command_logs" / second_label / "stdout.log").read_bytes()
    first_parsed = json.loads(
        (run_root / first_label / "parsed_trace.json").read_text(encoding="utf-8")
    )
    second_parsed = json.loads(
        (run_root / second_label / "parsed_trace.json").read_text(encoding="utf-8")
    )
    return {
        "exactness_rule": "all emitted non-timing fields and raw stdout bytes must be identical",
        "timing_fields_emitted": False,
        "repeat_1_stdout_sha256": hashlib.sha256(first_stdout).hexdigest(),
        "repeat_2_stdout_sha256": hashlib.sha256(second_stdout).hexdigest(),
        "stdout_byte_identical": first_stdout == second_stdout,
        "parsed_trace_identical": first_parsed == second_parsed,
        "passed": first_stdout == second_stdout and first_parsed == second_parsed,
    }


def finalize_manifest(run_root: Path) -> None:
    excluded = {run_root / "artifact_manifest.json", run_root / "sha256sums.txt"}
    files = sorted(path for path in run_root.rglob("*") if path.is_file() and path not in excluded)
    records = [
        {
            "path": str(path.relative_to(run_root)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    write_json(
        run_root / "artifact_manifest.json",
        {
            "schema_version": 1,
            "run_id": run_root.name,
            "files": records,
            "provenance": "portable_committed",
        },
    )
    (run_root / "sha256sums.txt").write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in records),
        encoding="utf-8",
    )


def discard_ephemeral_binaries(run_root: Path, binaries: dict[str, Path]) -> None:
    records = []
    for name, path in binaries.items():
        records.append(
            {
                "name": name,
                "ephemeral_path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "retained": False,
            }
        )
        path.unlink()
    build_dir = run_root / "build"
    build_dir.rmdir()
    write_json(
        run_root / "ephemeral_binary_manifest.json",
        {
            "policy": "hash_and_linkage_only; generated binaries were deleted before artifact finalization",
            "binaries": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flowstar-root",
        type=Path,
        default=Path("/srv/local/shengenli/flowstar_stock_gcc11"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "flowstar_scalar_affine_correctness_closure"
        / "20260804T131445Z",
    )
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    flowstar_root = args.flowstar_root.resolve()

    if args.finalize_only:
        if not run_root.is_dir():
            raise SystemExit(f"missing run root: {run_root}")
        finalize_manifest(run_root)
        return

    if run_root.exists():
        raise SystemExit(f"refusing to overwrite existing run root: {run_root}")
    run_root.mkdir(parents=True)
    recorder = CommandRecorder(run_root)
    write_json(run_root / "start_state.json", start_state(run_root, flowstar_root))
    write_environment(run_root / "environment.txt")
    write_json(
        run_root / "primary_config.json",
        {
            **PRIMARY,
            "model": "x' = 1 + 2*x",
            "state_variable": "x",
            "accepted_step_policy": "fixed",
            "effective_rhs_order": 4,
            "preconditioning": "native diagonal scaling",
            "symbolic_remainder": "disabled",
            "interval_precision_bits": 53,
            "containment_tolerance": None,
            "repaired_hull_used": False,
        },
    )

    binaries = compile_binaries(recorder, flowstar_root, run_root)
    build_records = capture_toolchain_and_linkage(
        recorder, flowstar_root, run_root, binaries
    )
    identity = backend_identity(flowstar_root, binaries, build_records)
    write_json(run_root / "backend_identity.json", identity)

    primary_trace_1, _ = run_trace(
        recorder,
        flowstar_root,
        run_root,
        label="primary_repeat_1",
        binary="generated_stock_trace",
        config=PRIMARY,
    )
    primary_trace_2, _ = run_trace(
        recorder,
        flowstar_root,
        run_root,
        label="primary_repeat_2",
        binary="generated_stock_trace",
        config=PRIMARY,
    )
    repeat_comparison = primary_comparison(
        run_root, "primary_repeat_1", "primary_repeat_2"
    )
    if not repeat_comparison["passed"]:
        raise RuntimeError("primary repetitions are not deterministic")
    write_json(run_root / "primary_repeat_comparison.json", repeat_comparison)

    official_trace, _ = run_trace(
        recorder,
        flowstar_root,
        run_root,
        label="official_stock_native_api",
        binary="official_scalar_affine",
        config=PRIMARY,
    )
    primary_oracle, _ = run_oracle(
        recorder,
        flowstar_root,
        run_root,
        label="oracle_primary",
        x0_lower="0",
        x0_upper="0.1",
        h="0.01",
    )
    primary_endpoint_oracle = oracle_endpoint(primary_oracle)
    primary_tube_oracle = oracle_tube(primary_oracle)
    fallback = high_precision_outward_oracle("0", "0.1", "0.01")
    oracle_evidence = {
        "classification": "formal_mpfr_directed_oracle",
        "closed_form": "x(t;x0)=(x0+1/2)*exp(2*t)-1/2",
        "mpfr": primary_oracle,
        "endpoint_binary64_outward": primary_endpoint_oracle,
        "tube_binary64_outward": primary_tube_oracle,
        "monotonicity": monotonicity_certificate(0.0, 0.1, 0.01),
        "secondary_high_precision_outward_sanity": fallback,
        "secondary_rk4_1000_substeps": [
            rk4_endpoint(0.0, 0.01),
            rk4_endpoint(0.1, 0.01),
        ],
        "old_rk4_values_are_not_the_analytic_oracle": True,
    }
    write_json(run_root / "analytic_oracle.json", oracle_evidence)

    fields = validate_field_separation(primary_trace_1)
    raw_endpoint = fields["endpoint_raw"]
    generated_defect = containment_defect(raw_endpoint, primary_endpoint_oracle)
    tube_defect = containment_defect(fields["full_tube"], primary_tube_oracle)

    accepted_h = float(official_trace["domains"]["official_accepted"]["0"]["upper"])
    official_oracle, _ = run_oracle(
        recorder,
        flowstar_root,
        run_root,
        label="oracle_official_accepted_right",
        x0_lower="0",
        x0_upper="0.1",
        h=f"{accepted_h:.17g}",
    )
    official_endpoint = box(
        official_trace, "official_accepted_right_endpoint", "full_initial_interval"
    )
    official_tube = box(official_trace, "official_full_tube", "full_initial_interval")
    official_endpoint_defect = containment_defect(
        official_endpoint, oracle_endpoint(official_oracle)
    )
    official_tube_defect = containment_defect(official_tube, oracle_tube(official_oracle))

    mirror_terms_equal = term_map(
        primary_trace_1, "accepted_mirror_tmv_pre"
    ) == term_map(primary_trace_1, "accepted_native_tmv_pre")
    mirror_remainder_equal = (
        primary_trace_1["stages"]["accepted_mirror_tmv_pre"]["remainders"]
        == primary_trace_1["stages"]["accepted_native_tmv_pre"]["remainders"]
    )
    generated_domain = primary_trace_1["domains"]["accepted"]
    parity = {
        "configuration_parity": {
            "model_text_and_constants": True,
            "initial_set_representation": True,
            "order_and_cutoff": True,
            "candidate_remainder": True,
            "preconditioning": True,
            "symbolic_remainder": True,
        },
        "schedule_parity": {
            "passed": accepted_h == float(generated_domain["0"]["upper"]),
            "official_accepted_h": accepted_h,
            "generated_accepted_h": float(generated_domain["0"]["upper"]),
            "reason": "ODE::reach starts its loop at THRESHOLD_HIGH and shortens the sole segment by 1e-12",
        },
        "numerical_range_parity": {
            "passed": False,
            "reason": "accepted right endpoints differ; comparison at one nominal h would conflate schedule and range semantics",
            "official_accepted_right_endpoint": official_endpoint,
            "generated_requested_h_endpoint_raw": raw_endpoint,
        },
        "field_parity": {
            "raw_polynomial_coefficients": "unavailable_on_official_public_API_route",
            "stored_remainder": "unavailable_on_official_public_API_route",
            "candidate_and_picard_fields": "unavailable_on_official_public_API_route",
            "endpoint_raw": "different accepted times",
            "last_segment": "both available, different accepted time domains",
            "full_tube": "both available, different accepted time domains",
        },
        "diagnostic_replay_matches_accepted_object": {
            "terms_exact": mirror_terms_equal,
            "remainder_exact": mirror_remainder_equal,
            "passed": mirror_terms_equal and mirror_remainder_equal,
        },
        "official_stock_result": {
            "completed": bool(official_trace["official"].get("completed")),
            "segments": official_trace["official"].get("segments"),
            "endpoint_defect_at_accepted_right": official_endpoint_defect,
            "tube_defect": official_tube_defect,
        },
        "generated_stock_result": {
            "advanced": primary_trace_1["status"].get("advanced") == 1,
            "endpoint_defect_at_requested_h": generated_defect,
            "tube_defect": tube_defect,
        },
    }
    if not parity["diagnostic_replay_matches_accepted_object"]["passed"]:
        raise RuntimeError("read-only diagnostic replay does not match accepted object")
    write_json(run_root / "official_generated_parity.json", parity)

    rows = first_loss_rows(primary_trace_1, primary_endpoint_oracle)
    first_loss = next((row for row in rows if row["first_loss"]), None)
    if first_loss is None:
        raise RuntimeError("no first containment loss was found")
    picard_terms = term_map(primary_trace_1, "polynomial_picard_order_4")
    expected_tau4_xi = 0.05 * (2.0**4) / math.factorial(4)
    first_loss_evidence = {
        "path": "generated-stock diagnostic linked to clean stock archive",
        "rows": rows,
        "first_loss": first_loss,
        "prevalidation_polynomial_diagnostic": {
            "stage": "polynomial_picard_order_4",
            "enclosure_contract": False,
            "observed_terms": {",".join(map(str, key)): value for key, value in picard_terms.items()},
            "expected_tau4_initial_generator_coefficient": expected_tau4_xi,
            "observed_tau4_initial_generator_term": picard_terms.get((4, 1)),
            "term_missing": (4, 1) not in picard_terms,
            "source": "TaylorModel.h:3698-3705 called by Continuous.cpp:954-956",
        },
        "mechanism": (
            "The order-4 polynomial Picard result lacks the tau^4*xi initial-"
            "dependency term. Refinement 1 still covers the closed-form corners, "
            "but refinement 2 accepts a much smaller stored remainder using only "
            "the stock subset/width test; that remainder no longer covers the "
            "missing dependency term and analytic tail."
        ),
        "selected_outcome": "F_clean_stock_flowstar_core_behavior",
        "official_path_confirmation": {
            "under_enclosed_at_its_accepted_right_endpoint": not official_endpoint_defect["contained"],
            "defect": official_endpoint_defect,
            "internal_refinement_fields_available": False,
        },
        "correctness_gate": "OPEN",
        "primary_comparison_eligible": False,
    }
    write_json(run_root / "first_containment_loss.json", first_loss_evidence)

    matrix_rows: list[dict[str, Any]] = []
    trace_cache: dict[tuple[Any, ...], tuple[dict[str, Any], str]] = {
        ("0.01", 4, "0", "0.1"): (primary_trace_1, "primary_repeat_1")
    }
    oracle_cache: dict[tuple[str, str, str], dict[str, Any]] = {
        ("0", "0.1", "0.01"): primary_oracle
    }
    for config in matrix_configs():
        key = (
            str(config["h"]),
            int(config["order"]),
            str(config["x0_lower"]),
            str(config["x0_upper"]),
        )
        if key in trace_cache:
            trace, trace_label = trace_cache[key]
        else:
            trace_label = f"matrix_{config['id'].replace('.', 'p')}"
            trace, _ = run_trace(
                recorder,
                flowstar_root,
                run_root,
                label=trace_label,
                binary="generated_stock_trace",
                config=config,
            )
            trace_cache[key] = (trace, trace_label)
        oracle_key = (
            str(config["x0_lower"]),
            str(config["x0_upper"]),
            str(config["h"]),
        )
        oracle = oracle_cache.get(oracle_key)
        if oracle is None:
            oracle_label = "oracle_" + "_".join(oracle_key).replace(".", "p")
            oracle, _ = run_oracle(
                recorder,
                flowstar_root,
                run_root,
                label=oracle_label,
                x0_lower=oracle_key[0],
                x0_upper=oracle_key[1],
                h=oracle_key[2],
            )
            oracle_cache[oracle_key] = oracle
        expected = oracle_endpoint(oracle)
        exported = box(trace, "endpoint_raw", "full_initial_interval")
        defect = containment_defect(exported, expected)
        remainder = trace["stages"]["accepted_native_tmv_pre"]["remainders"]["0"]
        matrix_rows.append(
            {
                "id": config["id"],
                "diagnostic": config["diagnostic"],
                "path": "generated-stock",
                "trace": trace_label,
                "h": config["h"],
                "requested_order": config["order"],
                "effective_rhs_order": trace["config"]["effective_rhs_order"],
                "x0_lower": config["x0_lower"],
                "x0_upper": config["x0_upper"],
                "candidate_remainder": config["candidate_remainder"],
                "cutoff": config["cutoff"],
                "exported_lower": exported[0],
                "exported_upper": exported[1],
                "oracle_lower": expected[0],
                "oracle_upper": expected[1],
                "signed_lower_error": exported[0] - expected[0],
                "signed_upper_error": exported[1] - expected[1],
                **defect,
                "accepted_remainder_lower": remainder["lower"],
                "accepted_remainder_upper": remainder["upper"],
                "accepted_polynomial_term_count": len(
                    trace["stages"]["accepted_native_tmv_pre"]["terms"]
                ),
            }
        )
    matrix_path = run_root / "step_order_corner_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matrix_rows[0]))
        writer.writeheader()
        writer.writerows(matrix_rows)

    step_rows = sorted(
        (row for row in matrix_rows if "step" in row["diagnostic"]),
        key=lambda row: float(row["h"]),
        reverse=True,
    )
    slopes = []
    for previous, current in zip(step_rows, step_rows[1:]):
        if previous["max_defect"] > 0 and current["max_defect"] > 0:
            slopes.append(
                math.log(previous["max_defect"] / current["max_defect"])
                / math.log(float(previous["h"]) / float(current["h"]))
            )
    write_json(
        run_root / "diagnostic_matrix_analysis.json",
        {
            "rows": len(matrix_rows),
            "scope": "required bounded step/order/corner diagnostics only",
            "observed_step_error_slopes": slopes,
            "primary_configuration_repaired_by_matrix": False,
            "interpretation": (
                "Step/order/corner variations are attribution diagnostics; no row "
                "changes the failed primary contract."
            ),
        },
    )
    (run_root / "test_results.txt").write_text(
        "Pending final repository and compiled smoke validation.\n", encoding="utf-8"
    )
    discard_ephemeral_binaries(run_root, binaries)
    finalize_manifest(run_root)
    print(run_root)


if __name__ == "__main__":
    main()
