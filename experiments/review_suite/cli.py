"""One supported entrypoint for research review, recomputation and limited reruns."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .data import ROOT, experiment, load_registry, profile_check, summarize

WHOLE_ENGINE_REVISION = "280abb400610f56210a7a5be61d5f98be3e27251"
ACCELERATED_ENGINE_REVISION = "f0ddc6e49b048e217b8cefbf1f889c91ba8aa99e"


def checked_engine_root(engine_root, *, revision=WHOLE_ENGINE_REVISION):
    """Bind the supported route to the complete-horizon, strict repair version."""
    if engine_root is None:
        raise ValueError("whole-engine requires --engine-root")
    root = Path(engine_root).resolve()
    if not (root / "src/flowstar_gpu/__init__.py").is_file():
        raise FileNotFoundError(f"external engine source is missing: {root}")
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"cannot verify external engine checkout: {root}") from error
    if head != revision or dirty:
        raise ValueError(f"supported whole-engine requires clean {revision}; "
                         f"found {head}, dirty={bool(dirty)} at {root}")
    return root


def whole_engine_run(plant, device, engine_root, output, *, steps=1000, quiet=False, accelerated=False):
    if device not in ("cpu", "cuda"):
        raise ValueError("whole-engine requires explicit --device cpu or --device cuda")
    revision = ACCELERATED_ENGINE_REVISION if accelerated else WHOLE_ENGINE_REVISION
    engine_root = checked_engine_root(engine_root, revision=revision)
    command = [sys.executable, "-m", "experiments.whole_engine_feasibility.candidate",
               "--engine-root", str(engine_root), "--plant", plant,
               "--batch", "1", "--steps", str(steps), "--device", device,
               "--record", "--output", str(output)]
    algorithms = {"composition": "horner", "glue": "graph", "support_policy": "structural",
                  "validation_policy": "defer_polynomial",
                  "rhs_form": "regrouped" if plant == "van_der_pol" else "original"}
    if accelerated:
        for option, value in algorithms.items():
            command.extend(["--" + option.replace("_", "-"), value])
    execute(command, output=output, quiet=quiet)
    result = json.loads((Path(output) / "summary.json").read_text())
    if (result.get("completed") is not True or result.get("failure") is not None
            or result.get("accepted_steps") != [steps]):
        raise RuntimeError(f"{plant} whole-engine did not complete; preserved output: {output}")
    if (result["engine_source"]["head"] != revision
            or result["engine_source"]["status"]
            or Path(result["engine_source"]["root"]).resolve() != engine_root):
        raise RuntimeError(f"whole-engine source identity changed; inspect output: {output}")
    if accelerated and any(result.get("algorithms", {}).get(key) != value
                           for key, value in algorithms.items()):
        raise RuntimeError(f"accelerated algorithm contract changed; inspect output: {output}")
    return result


def whole_engine_smoke(device, engine_root, *, accelerated=False):
    scratch = Path(tempfile.mkdtemp(prefix="review-whole-engine-smoke-"))
    try:
        cases = []
        for plant in ("van_der_pol", "brusselator"):
            result = whole_engine_run(plant, device, engine_root, Path(scratch) / plant,
                                      steps=2, quiet=True, accelerated=accelerated)
            cases.append({key: result[key] for key in
                          ("plant", "accepted_steps", "engine_source", "device", "extensions")})
    except BaseException:
        print(f"whole-engine smoke evidence retained: {scratch}", file=sys.stderr)
        raise
    else:
        shutil.rmtree(scratch)
    print(json.dumps({"backend": "whole-engine-accelerated" if accelerated else "whole-engine",
                      "cases": cases, "status": "accepted"}, indent=2))


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


def run_experiment(experiment_id, backend, output, *, engine_root=None, device=None):
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
    if backend in ("whole-engine", "whole-engine-accelerated"):
        if experiment_id not in ("vdp-fixed-full", "brusselator-fixed-full"):
            raise ValueError("whole-engine supports only the two fixed original-box horizons")
        plant = "van_der_pol" if experiment_id == "vdp-fixed-full" else "brusselator"
        whole_engine_run(plant, device, engine_root, output,
                         accelerated=backend == "whole-engine-accelerated")
        return output
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
    smoke.add_argument("--backend", choices=("cpu","gpu-range","resident","whole-engine","whole-engine-accelerated"), required=True)
    run = subs.add_parser("run", help="new full reference/GPU-range run or resident prefix")
    run.add_argument("--experiment", required=True)
    run.add_argument("--backend", choices=("cpu","gpu-range","resident","whole-engine","whole-engine-accelerated"), required=True)
    run.add_argument("--out", type=Path, required=True)
    for action in (smoke, run):
        action.add_argument("--engine-root", type=Path, help="clean external engine checkout; whole-engine only")
        action.add_argument("--device", choices=("cpu", "cuda"), help="explicit device; whole-engine only")
    observe = subs.add_parser("observe", help="physical bounds from complete whole-engine factored models")
    observe.add_argument("--input", type=Path, required=True, help="factored.jsonl.gz with adjacent summary.json")
    observe.add_argument("--out", type=Path, required=True)
    observe.add_argument("--arithmetic", choices=("fraction", "outward"), default="fraction")
    args = parser.parse_args(argv)
    if args.action in ("smoke", "run"):
        if args.backend in ("whole-engine", "whole-engine-accelerated"):
            if args.engine_root is None or args.device is None:
                parser.error("whole-engine requires --engine-root and --device")
        elif args.engine_root is not None or args.device is not None:
            parser.error("--engine-root and --device apply only to --backend whole-engine")
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
        if args.backend in ("whole-engine", "whole-engine-accelerated"):
            whole_engine_smoke(args.device, args.engine_root,
                               accelerated=args.backend == "whole-engine-accelerated")
        elif args.backend == "cpu":
            cpu_smoke()
        else:
            gpu_smoke(args.backend)
        return 0
    if args.action == "run":
        run_experiment(args.experiment, args.backend, args.out,
                       engine_root=args.engine_root, device=args.device)
        return 0
    if args.action == "observe":
        module = ("experiments.flowstar_acceleration.observe_outward" if args.arithmetic == "outward"
                  else "experiments.whole_engine_feasibility.observe_saved")
        options = (["--export-backend", "outward_binary64", "--range-backend", "outward_binary64"]
                   if args.arithmetic == "outward" else [])
        execute([sys.executable, "-m", module, "--input", str(args.input.resolve()),
                 "--output", str(args.out.resolve()), *options], output=args.out)
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"review-suite: {error}", file=sys.stderr)
        raise SystemExit(2)
