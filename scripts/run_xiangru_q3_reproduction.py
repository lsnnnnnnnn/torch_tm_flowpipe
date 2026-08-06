#!/usr/bin/env python3
"""Run the unmodified Xiangru complete-Q3 B48 baseline with raw evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence


XIANGRU_ROOT = Path("/srv/local/shengenli/CROWN-Reach_Development_native_27d2905")
DIFFREACH_ROOT = Path("/srv/local/shengenli/DiffReach")
XIANGRU_PYTHON = Path("/srv/local/shengenli/native_envs/crownreach28/bin/python")
DIFFREACH_PYTHON = Path("/srv/local/shengenli/native_envs/diffreach083/bin/python")
CONFIG_ROOT = Path(
    "/srv/local/shengenli/torch_tm_flowpipe_xiangru_q3_audit/outputs/"
    "native_reproduction_no_adapters/20260804T081205Z/xiangru/"
    "x3_native_generated_config_workspace"
)
EXPECTED_COMMIT = "27d29050a5f214b56f211ca9cb411e734ed80230"
EXPECTED_CONFIG_SHA256 = "13b28acaa6addad27b12f2e00d1e6a81920bfebe849382ddb6b9a736bb4ff090"
EXPECTED_CONTROLLER_SHA256 = "bb80479ce51b6f2558ac4a47cae2831ff3f49275ffaf7b1b874adf3c3b14703e"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run(argv: Sequence[str], *, cwd: Path) -> str:
    return subprocess.run(
        list(argv), cwd=cwd, check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def _git_clean_at_expected_commit() -> None:
    head = _run(["git", "rev-parse", "HEAD"], cwd=XIANGRU_ROOT)
    status = _run(["git", "status", "--porcelain=v2"], cwd=XIANGRU_ROOT)
    if head != EXPECTED_COMMIT:
        raise ValueError(f"Xiangru reproduction HEAD changed: {head}")
    if status:
        raise ValueError("Xiangru reproduction worktree is dirty")


def _validate_inputs() -> tuple[Path, Path]:
    config = CONFIG_ROOT / "diffreach_config.json"
    controller = CONFIG_ROOT / "controller_transformed.onnx"
    observed = {"config": _sha256(config), "controller": _sha256(controller)}
    expected = {"config": EXPECTED_CONFIG_SHA256, "controller": EXPECTED_CONTROLLER_SHA256}
    if observed != expected:
        raise ValueError(f"frozen baseline inputs changed: expected {expected}, observed {observed}")
    return config, controller


def _environment() -> dict[str, Any]:
    probe = (
        "import json,numpy,torch,sys; "
        "print(json.dumps({'python':sys.version,'pytorch':torch.__version__,"
        "'pytorch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'cuda_device_count':torch.cuda.device_count(),'numpy':numpy.__version__,"
        "'default_dtype':str(torch.get_default_dtype())}))"
    )
    python_environment = json.loads(
        _run([str(XIANGRU_PYTHON), "-c", probe], cwd=XIANGRU_ROOT)
    )
    return {
        **python_environment,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cuda_visible_devices": "0",
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "xiangru_root": str(XIANGRU_ROOT),
        "xiangru_commit": EXPECTED_COMMIT,
        "diffreach_root": str(DIFFREACH_ROOT),
        "diffreach_commit": _run(["git", "rev-parse", "HEAD"], cwd=DIFFREACH_ROOT),
    }


def _write_source_manifest(path: Path) -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-s", "-z"], cwd=XIANGRU_ROOT, check=True,
        stdout=subprocess.PIPE,
    ).stdout.split(b"\0")
    rows = []
    for encoded in tracked:
        if not encoded:
            continue
        metadata, encoded_path = encoded.split(b"\t", 1)
        mode, object_sha, stage = metadata.decode("ascii").split()
        relative = encoded_path.decode("utf-8", errors="surrogateescape")
        source = XIANGRU_ROOT / relative
        if mode == "160000":
            rows.append(f"gitlink:{object_sha}  {relative}\n")
            continue
        if mode == "120000" and source.is_symlink():
            link_target = os.readlink(source).encode("utf-8", errors="surrogateescape")
            rows.append(f"{_sha256_bytes(link_target)}  {relative} -> {os.readlink(source)}\n")
            continue
        if stage != "0" or not source.is_file():
            raise FileNotFoundError(f"tracked source is not a regular file: {source}")
        rows.append(f"{_sha256(source)}  {relative}\n")
    path.write_text("".join(rows), encoding="utf-8")


def _bwrap_command(raw_outputs: Path) -> list[str]:
    inner_root = "/home/xiangru4/CROWN-Reach_Development"
    output_json = "/tmp/xiangru_q3_audit/rep1_q3/s3r_q3_b48_rep1.json"
    output_markdown = "/tmp/xiangru_q3_audit/rep1_q3/s3r_q3_b48_rep1.md"
    return [
        "bwrap", "--ro-bind", "/", "/", "--tmpfs", "/home", "--tmpfs", "/tmp",
        "--dir", "/home/xiangru4", "--ro-bind", str(XIANGRU_ROOT), inner_root,
        "--ro-bind", str(DIFFREACH_ROOT), "/home/xiangru4/DiffReach",
        "--bind", str(CONFIG_ROOT),
        f"{inner_root}/experiments/reachability/results/20260724T234338.154274Z__tora_homogeneous__diffreach__full",
        "--dir", "/tmp/xiangru_q3_audit", "--bind", str(raw_outputs),
        "/tmp/xiangru_q3_audit/rep1_q3", "--proc", "/proc", "--dev-bind", "/dev", "/dev",
        "--chdir", inner_root, str(XIANGRU_PYTHON),
        f"{inner_root}/experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py",
        "--device", "cuda", "--policies", "b48_static", "--methods", "complete_q3",
        "--cap", "200", "--q3-engine", "dynamic", "--controller-backend", "autolirpa",
        "--controller-platform", "cuda", "--controller-mode", "eager",
        "--controller-composition", "outward", "--diffreach-root", "/home/xiangru4/DiffReach",
        "--diffreach-python", str(DIFFREACH_PYTHON), "--output-json", output_json,
        "--output-markdown", output_markdown,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty reproduction directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw_outputs = output / "raw_outputs"
    raw_outputs.mkdir()

    _git_clean_at_expected_commit()
    config, controller = _validate_inputs()
    environment = _environment()
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(config, output / "config_resolved.json")
    snapshot = output / "original_config_snapshot"
    snapshot.mkdir()
    shutil.copyfile(config, snapshot / "diffreach_config.json")
    shutil.copyfile(controller, snapshot / "controller_transformed.onnx")
    _write_source_manifest(output / "source_manifest.sha256")

    command = _bwrap_command(raw_outputs)
    timed_command = [
        "/usr/bin/time", "-v", "-o", str(output / "resource_usage.txt"), *command
    ]
    (output / "command.txt").write_text(shlex.join(timed_command) + "\n", encoding="utf-8")
    (output / "original_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    started_utc = _utc_now()
    started = time.monotonic()
    timeout_expired = False
    try:
        completed = subprocess.run(
            timed_command, cwd=XIANGRU_ROOT, env=env, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=args.timeout_seconds,
        )
        exit_code: int | None = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timeout_expired = True
        exit_code = None
        stdout = error.stdout or b""
        stderr = error.stderr or b""
    wall_seconds = time.monotonic() - started
    ended_utc = _utc_now()
    (output / "stdout.log").write_bytes(stdout)
    (output / "stderr.log").write_bytes(stderr)
    (output / "exit_code.txt").write_text(
        "timeout\n" if exit_code is None else f"{exit_code}\n", encoding="utf-8"
    )
    (output / "wall_time.txt").write_text(f"{wall_seconds:.17g}\n", encoding="utf-8")
    metadata = {
        "start_time_utc": started_utc,
        "end_time_utc": ended_utc,
        "wall_seconds": wall_seconds,
        "timeout_seconds": args.timeout_seconds,
        "timeout_expired": timeout_expired,
        "exit_code": exit_code,
        "stdout_sha256": _sha256(output / "stdout.log"),
        "stderr_sha256": _sha256(output / "stderr.log"),
        "source_manifest_sha256": _sha256(output / "source_manifest.sha256"),
        "config_sha256": _sha256(output / "config_resolved.json"),
        "controller_sha256": _sha256(snapshot / "controller_transformed.onnx"),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 124 if timeout_expired else int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
