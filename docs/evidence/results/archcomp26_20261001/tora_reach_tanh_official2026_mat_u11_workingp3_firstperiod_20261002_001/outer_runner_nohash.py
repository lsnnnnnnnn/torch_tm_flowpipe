#!/usr/bin/env python3
"""Run one ARCH-COMP cell in a new directory, without content digests.

Example: python tools/run_archcomp26_nohash.py --run-dir runs/example \
  --instance quad-reach --method huan --contract-label quad-short-v1 \
  --cwd /path/to/repo --timeout-s 600 --env CUDA_VISIBLE_DEVICES=3 \
  -- python -B driver.py config.yaml
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time


RECORDED_ENV = {
    "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TORCH_EXTENSIONS_DIR",
    "PYTHONDONTWRITEBYTECODE",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def stop_group(process):
    # Keep the leader unreaped until after SIGKILL so its process-group ID cannot be reused.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    time.sleep(1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run(args, parser):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("--timeout-s must be positive and finite")
    cwd = args.cwd.resolve()
    if not cwd.is_dir():
        parser.error(f"working directory does not exist: {cwd}")

    environment = os.environ.copy()
    selected = set(RECORDED_ENV)
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            parser.error("invalid --env assignment; expected NAME=VALUE")
        if re.search(r"TOKEN|SECRET|PASSWORD|PASSWD|(?:^|_)KEY(?:_|$)|APIKEY|ACCESSKEY|PRIVATEKEY|CREDENTIAL", key, re.I):
            parser.error(f"refusing to record a sensitive --env variable: {key}")
        environment[key] = value
        selected.add(key)

    run_dir = args.run_dir.resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError:
        parser.error(f"run directory already exists; inspect it before any new attempt: {run_dir}")

    write_json(run_dir / "START.json", {
        "schema": "archcomp26-nohash-run-v1",
        "instance": args.instance,
        "method": args.method,
        "contract_label": args.contract_label,
        "argv": command,
        "cwd": str(cwd),
        "selected_environment": {key: environment[key] for key in sorted(selected) if key in environment},
        "environment_policy": "inherit with recorded selected variables and explicit overrides",
        "timeout_s": args.timeout_s,
        "host": {
            "name": socket.gethostname(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "started_utc": utc_now(),
    })

    process = None
    timed_out = False
    failure = None
    exit_code = None
    started = time.monotonic()
    try:
        with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(
                command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=args.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_group(process)
                exit_code = process.returncode
                failure = "timeout"
    except OSError as error:
        failure = f"spawn_error: {error}"
    except KeyboardInterrupt:
        failure = "interrupted"
    finally:
        if process is not None and process.poll() is None:
            stop_group(process)
            exit_code = process.returncode

    if failure is None and exit_code != 0:
        failure = "nonzero_exit"
    write_json(run_dir / "RESULT.json", {
        "status": "timeout" if timed_out else ("completed" if failure is None else "failed"),
        "failure": failure,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_s": time.monotonic() - started,
        "ended_utc": utc_now(),
        "pid": process.pid if process is not None else None,
    })
    return 0 if failure is None else (124 if timed_out else 1)


def self_check():
    with tempfile.TemporaryDirectory(prefix="archcomp26-nohash-check-") as directory:
        root = Path(directory)
        def invoke(run_dir, timeout, code):
            return subprocess.run([
                sys.executable, __file__, "--run-dir", str(run_dir),
                "--instance", "fake", "--method", "fake",
                "--contract-label", "self-check", "--cwd", directory,
                "--timeout-s", str(timeout), "--", sys.executable, "-c", code,
            ], capture_output=True, text=True)

        complete = root / "complete"
        assert invoke(complete, 5, "print('stdout'); import sys; print('stderr', file=sys.stderr)").returncode == 0
        assert (complete / "stdout.log").read_text().strip() == "stdout"
        assert (complete / "stderr.log").read_text().strip() == "stderr"
        assert json.loads((complete / "RESULT.json").read_text())["status"] == "completed"
        assert invoke(complete, 5, "print('duplicate')").returncode == 2
        assert (complete / "stdout.log").read_text().strip() == "stdout"

        secret_run = root / "secret"
        secret = subprocess.run([
            sys.executable, __file__, "--run-dir", str(secret_run),
            "--instance", "fake", "--method", "fake", "--contract-label", "self-check",
            "--cwd", directory, "--timeout-s", "5", "--env", "API_KEY=fake-value",
            "--", sys.executable, "-c", "pass",
        ], capture_output=True, text=True)
        assert secret.returncode == 2 and not secret_run.exists()
        assert "fake-value" not in secret.stderr

        marker, spawned = root / "orphan_marker", root / "spawned"
        grandchild = f"import time,pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('orphan')"
        child = (
            f"import subprocess,sys,time,pathlib; "
            f"subprocess.Popen([sys.executable,'-c',{grandchild!r}]); "
            f"pathlib.Path({str(spawned)!r}).write_text('yes'); time.sleep(10)"
        )
        timeout = root / "timeout"
        assert invoke(timeout, 1.5, child).returncode == 124
        assert spawned.is_file(), "fake child did not spawn its grandchild before the timeout"
        assert json.loads((timeout / "RESULT.json").read_text())["status"] == "timeout"
        time.sleep(3.2)
        assert not marker.exists(), "the grandchild survived the timeout"
    print("self-check passed: success, duplicate refusal, secret guard, whole-group timeout")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-check"]:
        return self_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--contract-label", required=True)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--timeout-s", required=True, type=float)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    raise SystemExit(main())
