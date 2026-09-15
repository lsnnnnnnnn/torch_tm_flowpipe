"""One supported entrypoint for research review, recomputation and limited reruns."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .data import ROOT, experiment, load_registry, profile_check, summarize


def execute(command, *, output=None, quiet=False):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", "")])
    process = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                             capture_output=quiet)
    if process.returncode:
        detail = ((process.stdout or "") + (process.stderr or ""))[-2000:] if quiet else ""
        raise RuntimeError(f"runner exited {process.returncode}; output: {output}\n{detail}")
    return process.returncode


def cpu_smoke():
    from experiments.endpoint_roundoff_repair.frozen import setup, step
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = []
    for plant in ("van_der_pol", "brusselator"):
        config, current, state = setup(plant)
        for index in (1, 2):
            segment = step(plant, current, state, index, h=.01 if plant == "van_der_pol" else .02)
            if segment.status != "validated" or segment.reset_tm is None:
                raise RuntimeError(f"{plant} step {index} was not accepted: {segment.status} {segment.message}")
            current, state = segment.reset_tm, segment.flowstar_normal_state
        cases.append(dict(plant=plant, accepted_steps=state.step_index,
                          current_box_hex=[[float(v.lo).hex(),float(v.hi).hex()]
                                           for v in current.range_box()],
                          queue_size=len(state.symbolic_queue.J)))
    print(json.dumps({"backend":"cpu","cases":cases,"status":"accepted"},indent=2))
    return cases


def gpu_smoke(backend):
    with tempfile.TemporaryDirectory(prefix="review-smoke-") as scratch:
        for plant in ("van_der_pol", "brusselator"):
            target = Path(scratch) / plant
            if backend == "gpu-range":
                command = [sys.executable, "-m", "experiments.live_gpu_packets.long_horizon",
                           "--plant", plant, "--route", "Gp", "--steps", "2",
                           "--output", str(target)]
                result_file = "LONG_HORIZON_RESULT.json"
            else:
                command = [sys.executable, "-m", "experiments.live_range_solver.runner",
                           "--plant", plant, "--route", "Gr", "--batch", "1",
                           "--original", "--steps", "2", "--output", str(target)]
                result_file = "summary.json"
            execute(command, output=target, quiet=True)
            result = json.loads((target/result_file).read_text())
            if backend == "gpu-range" and not result["achieved"]:
                raise RuntimeError(f"{plant} GPU range smoke did not finish")
            if backend == "resident" and result.get("successful_tasks") != 1:
                raise RuntimeError(f"{plant} resident smoke did not finish")
    print(json.dumps({"backend":backend,"plants":["van_der_pol","brusselator"],
                      "accepted_steps_per_plant":2,"status":"accepted"},indent=2))


def run_experiment(experiment_id, backend, output):
    registry = load_registry()
    item = experiment(registry, experiment_id)
    if item is None:
        raise ValueError(f"unknown experiment: {experiment_id}")
    profile_check()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("new scientific runs require a clean source checkout")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    if experiment_id in ("vdp-fixed-full", "brusselator-fixed-full", "vdp-adaptive"):
        if backend == "resident":
            raise ValueError("resident composition has no accepted B1 full-horizon route")
        if experiment_id == "vdp-adaptive" and backend != "cpu":
            raise ValueError("adaptive VDP is supported only on the CPU research reference")
        plant = "brusselator" if experiment_id == "brusselator-fixed-full" else "van_der_pol"
        if backend == "cpu":
            head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
            command = [sys.executable, "-m", "experiments.endpoint_roundoff_repair.run",
                       "--plant", plant, "--scientific-sha", head,
                       "--steps", "1000", "--output", str(output)]
            if experiment_id == "vdp-adaptive":
                command.append("--adaptive")
        elif backend == "gpu-range":
            command = [sys.executable, "-m", "experiments.live_gpu_packets.long_horizon",
                       "--plant", plant, "--route", "Gp", "--steps", "1000",
                       "--output", str(output)]
        else:
            raise ValueError(f"unsupported backend: {backend}")
    elif experiment_id in ("vdp-resident-prefix", "brusselator-resident-prefix",
                           "vdp-resident-b2", "brusselator-resident-b2"):
        if backend != "resident":
            raise ValueError("resident prefix requires --backend resident")
        plant = "van_der_pol" if experiment_id.startswith("vdp") else "brusselator"
        command = [sys.executable, "-m", "experiments.live_range_solver.runner",
                   "--plant", plant, "--route", "Gr"]
        if experiment_id.endswith("-b2"):
            command.extend(["--batch", "2", "--steps", "2"])
        else:
            command.extend(["--batch", "1", "--original", "--steps", "20"])
        command.extend(["--output", str(output)])
    else:
        raise ValueError(f"{experiment_id} is indexed historical evidence, not a current rerun")
    print(json.dumps({"experiment":experiment_id,"backend":backend,"command":command,
                      "output":str(output)},indent=2), flush=True)
    execute(command, output=output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("list", help="list registered research questions and evidence levels")
    summary = subs.add_parser("summarize", help="recompute derived tables from immutable saved data")
    summary.add_argument("--all", action="store_true", required=True)
    summary.add_argument("--out", type=Path, required=True)
    plots = subs.add_parser("plot", help="redraw review figures from managed normalized data")
    plots.add_argument("--all", action="store_true", required=True)
    plots.add_argument("--out", type=Path, required=True)
    smoke = subs.add_parser("smoke", help="two accepted steps on each system")
    smoke.add_argument("--backend", choices=("cpu","gpu-range","resident"), required=True)
    run = subs.add_parser("run", help="new full reference/GPU-range run or resident prefix")
    run.add_argument("--experiment", required=True)
    run.add_argument("--backend", choices=("cpu","gpu-range","resident"), required=True)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "list":
        registry = load_registry()
        for item in registry["experiments"]:
            print(f"{item['id']}\t{item['reproduction_level']}\t{item['plain_title']}")
        return 0
    if args.action == "summarize":
        summarize(args.out)
        return 0
    if args.action == "plot":
        from .plot import plot_all
        plot_all(args.out)
        return 0
    if args.action == "smoke":
        profile_check()
        if args.backend == "cpu":
            cpu_smoke()
        else:
            gpu_smoke(args.backend)
        return 0
    if args.action == "run":
        run_experiment(args.experiment, args.backend, args.out)
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"review-suite: {error}", file=sys.stderr)
        raise SystemExit(2)
