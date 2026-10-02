#!/usr/bin/env python3
"""Read-only input and resource gate for one isolated 128-row control run."""

import importlib.util
import json
from pathlib import Path
import subprocess

import psutil
import yaml


ROOT = Path(__file__).resolve().parent
BASE = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
REFERENCE = BASE / "runs/quad_fullbatch_p3_20260928/full1024_p3_trig_gpu14_1000_v1"
CONTROL_256 = BASE / "runs/archcomp26_20261001/p3_quad_old_weighted256_full1000_nohash_20261003_001"
RUN_ID = "p3_quad_old_weighted128_full1000_samewrapper_nohash_20261003_001"


def read_json(path):
    return json.loads(path.read_text())


def main():
    assert ROOT.name == RUN_ID
    assert not (ROOT / "run_001").exists()
    assert not (ROOT / "launcher_pid.txt").exists()

    baseline_adapter = (CONTROL_256 / "weighted_chunk256.py").read_text()
    baseline_runner = (CONTROL_256 / "runner.py").read_text()
    baseline_launcher = (CONTROL_256 / "launch_once.sh").read_text()
    expected_adapter = (baseline_adapter.replace("256", "128")
                        .replace("[1,2,10,255,128]", "[1,2,10,127,128]")
                        .replace("[0,257]", "[0,129]"))
    assert (ROOT / "weighted_chunk128.py").read_text() == expected_adapter
    assert (ROOT / "runner.py").read_text() == (baseline_runner
            .replace("weighted_chunk256.py", "weighted_chunk128.py")
            .replace("weighted256", "weighted128"))
    assert (ROOT / "launch_once.sh").read_text() == (baseline_launcher
            .replace(CONTROL_256.name, RUN_ID).replace("weighted256", "weighted128"))
    assert (ROOT / "nncs_watchdog_gpu14.py").read_bytes() == (CONTROL_256 / "nncs_watchdog_gpu14.py").read_bytes()
    assert (ROOT / "compare_full_saved.py").read_text() == (CONTROL_256 / "compare_full_saved.py").read_text().replace("weighted256", "weighted128")
    assert (ROOT / "quad_author_resolved.yaml").read_bytes() == (CONTROL_256 / "quad_author_resolved.yaml").read_bytes()
    assert (ROOT / "quad_author_resolved.yaml").read_bytes() == (BASE / "runs/archcomp_failure_20260923/quad_author_resolved.yaml").read_bytes()

    cfg = yaml.safe_load((ROOT / "quad_author_resolved.yaml").read_text())
    assert cfg["steps"] == 50 and cfg["sr_queue"] == 1000 and cfg["ode_order"] == 2
    spec = importlib.util.spec_from_file_location("control_128_adapter", ROOT / "weighted_chunk128.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    assert adapter.check_local()["status"] == "passed_CPU_padding_only"

    old = read_json(REFERENCE / "RESULT.json")
    prior_outer = read_json(CONTROL_256 / "run_001/RESULT.json")
    prior_inner = read_json(CONTROL_256 / "run_001/data/RESULT.json")
    prior_comparison = read_json(CONTROL_256 / "FULL_SAVED_COMPARISON.json")
    assert old["overall_horizon_completed"] and old["completed_step"] == 1000
    assert old["accepted_lane_steps"] == 1024000 and old["controller_calls"] == 50
    assert prior_outer["status"] == prior_inner["status"] == "completed"
    assert prior_outer["exit_code"] == prior_inner["driver_return"] == 0
    assert prior_inner["completed_substeps"] == 1000
    assert prior_inner["accepted_lane_substeps"] == 1024000
    assert prior_inner["nn_calls"] == 50
    assert prior_comparison["status"] == "all_1000_saved_steps_direct_equal"
    assert prior_comparison["compared_steps"] == 1000
    assert all((REFERENCE / f"observer_{step}.pt").is_file() for step in range(1, 1001))
    assert all((CONTROL_256 / f"run_001/data/observer_{step:04d}_{kind}.npy").is_file()
               for step in range(1, 1001) for kind in ("bounds", "accepted", "status"))

    import runner
    assert cfg["model_dir"] == str(runner.MODEL) and runner.MODEL.is_file()
    assert all(path.is_file() for path in runner.EXTENSIONS.values())
    used = subprocess.check_output(["nvidia-smi", "-i", "3", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
    assert used == "0", f"GPU3 occupied: {used} MiB"
    active = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        line = " ".join(process.info["cmdline"] or [])
        if (("p3_quad_old" in line and ("runner.py" in line or "nncs_watchdog_gpu14.py" in line))
                or "run_fullbatch_p3_trig_continue.py" in line):
            active.append(process.info["pid"])
    assert not active, active

    receipt = {
        "schema": "quad-old-author-p3-weighted128-samewrapper-full1000-preflight-v1",
        "status": "passed",
        "run_id_unused": True,
        "source_only_expected_256_to_128_changes": True,
        "same_old_author_yaml_direct_equal": True,
        "archived_reference_steps": 1000,
        "archived_reference_observer_files": 1000,
        "prior_256_steps": 1000,
        "prior_256_observer_arrays": 3000,
        "prior_256_direct_old_reference_comparison": prior_comparison["status"],
        "precompiled_extensions_present": len(runner.EXTENSIONS),
        "adapter_cpu_check": adapter.check_local(),
        "gpu3_used_mib": 0,
        "related_processes": active,
        "qualification": "input and resource gate only; no 128-row GPU result yet",
    }
    (ROOT / "PREFLIGHT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "run_id_unused": True,
                      "gpu3_used_mib": 0, "archived_reference_steps": 1000}))


if __name__ == "__main__":
    main()
