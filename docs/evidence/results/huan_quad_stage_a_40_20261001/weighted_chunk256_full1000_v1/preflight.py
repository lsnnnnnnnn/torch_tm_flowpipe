#!/usr/bin/env python3
"""No-digest gate before one old-author QUAD P3 weighted256 full run."""

import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import psutil
import torch
import yaml


ROOT = Path(__file__).resolve().parent
BASE = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = BASE / "runs/quad_fullbatch_p3_20260928/full1024_p3_trig_gpu14_1000_v1"
SHORT = BASE / "runs/archcomp26_20261001/p3_quad_old_weighted256_40_nohash_20261003_001/run_001/data"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    runner = load("full_candidate_runner", ROOT / "runner.py")
    adapter = load("full_candidate_adapter", ROOT / "weighted_chunk256.py")
    assert (ROOT / "quad_author_resolved.yaml").read_bytes() == (BASE / "runs/archcomp_failure_20260923/quad_author_resolved.yaml").read_bytes()
    config = yaml.safe_load((ROOT / "quad_author_resolved.yaml").read_text())
    assert config["steps"] == 50 and config["sr_queue"] == 1000 and config["ode_order"] == 2
    assert config["model_dir"] == str(runner.MODEL) and runner.MODEL.is_file()
    assert (ROOT / "weighted_chunk256.py").read_bytes() == (BASE / "runs/archcomp26_20261001/p3_quad_old_weighted256_40_nohash_20261003_001/weighted_chunk256.py").read_bytes()
    assert all(path.is_file() for path in runner.EXTENSIONS.values())
    assert not (ROOT / "run_001").exists()
    old_result = json.loads((OLD / "RESULT.json").read_text())
    assert old_result["overall_horizon_completed"] and old_result["completed_step"] == 1000
    assert old_result["controller_calls"] == 50 and old_result["accepted_lane_steps"] == 1024000
    assert sum((OLD / f"observer_{step}.pt").is_file() for step in range(1, 1001)) == 1000
    old_config = yaml.safe_load((ROOT / "quad_author_resolved.yaml").read_text())
    short_config = yaml.safe_load((BASE / "runs/archcomp26_20261001/p3_quad_old_weighted256_40_nohash_20261003_001/quad_40.yaml").read_text())
    old_config["steps"] = 2
    old_config.pop("sr_queue", None)
    assert old_config == short_config
    rows = [json.loads(line) for line in (SHORT / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 40
    for step, row in enumerate(rows, 1):
        prior = torch.load(OLD / f"observer_{step}.pt", map_location="cpu", weights_only=True)
        bounds = prior["bounds"].numpy()
        pooled = np.stack((bounds[:, :, 0].min(0), bounds[:, :, 1].max(0),
                           bounds[:, :, 2].min(0), bounds[:, :, 3].max(0)), -1)
        saved = np.asarray(row["tube_endpoint_union_12x4"], dtype=np.float64)
        assert np.array_equal(pooled.view(np.uint64), saved.view(np.uint64)), step
        assert row["accepted_count"] == int(prior["accepted"].sum()) == 1024, step
    check = adapter.check_local()
    assert check["status"] == "passed_CPU_padding_only"
    gpu = subprocess.check_output(["nvidia-smi", "-i", "3", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
    assert gpu == "0", f"GPU3 occupied: {gpu} MiB"
    active = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        line = " ".join(process.info["cmdline"] or [])
        if ("runner.py" in line and "p3_quad_old" in line) or "run_fullbatch_p3_trig_continue.py" in line:
            active.append(process.info["pid"])
    assert not active, active
    receipt = {"schema": "quad-old-author-p3-weighted256-full1000-preflight-v1",
               "status": "passed", "archived_full_steps": 1000,
               "archived_full_accepted_box_steps": 1024000,
               "archived_full_controller_calls": 50,
               "archived_observer_pt_files": 1000,
               "short_candidate_matches_archived_pooled_steps": 40,
               "full_yaml_direct_equal_to_archived": True,
               "full_contract_same_as_short_except_steps_and_sr_queue": True,
               "weighted_adapter_direct_equal_to_40_step_candidate": True,
               "precompiled_extensions_present": len(runner.EXTENSIONS),
               "candidate_run_id_unused": True,
               "gpu3_used_mib": 0, "related_processes": active,
               "adapter_cpu_check": check,
               "qualification": "direct saved-value checks only; no full numerical qualification yet; no content digest"}
    (ROOT / "PREFLIGHT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "archived_observer_pt_files": 1000,
                      "short_pooled_steps_equal": 40, "gpu3_used_mib": 0}))


if __name__ == "__main__":
    main()
