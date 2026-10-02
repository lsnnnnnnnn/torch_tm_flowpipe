#!/usr/bin/env python3
"""On the server, build separate h=0.05 native TORA binaries from frozen h=0.1 sources."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


SERVER = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
RUNS = SERVER / "runs/archcomp26_20261001"
OLD = RUNS / "native_tora_remain_build_001"
NEW = RUNS / "native_tora_remain_h005_probe_20261002"
OLD_WORK = OLD / "archcomp/TORA"
WORK = NEW / "build/archcomp/TORA"
OLD_TEXT = "setting.setFixedStepsize(0.1, 3);"
NEW_TEXT = "setting.setFixedStepsize(0.05, 3);"


def main():
    if not NEW.is_dir() or not (NEW / "INTENT.json").is_file():
        raise RuntimeError("new isolated root and intent must be staged first")
    if (NEW / "build").exists():
        raise RuntimeError("build already exists; inspect it instead of repeating")
    old_build = json.loads((OLD / "BUILD.json").read_text())
    if [entry["periods"] for entry in old_build["builds"]] != [1, 20]:
        raise RuntimeError("unexpected frozen build declarations")
    WORK.mkdir(parents=True)
    (NEW / "build/flowstar").symlink_to((OLD / "flowstar").resolve(), target_is_directory=True)
    for name in ("arch_ranges.h", "matched_reach.h", "crown_paper.py"):
        shutil.copy2(OLD_WORK / name, WORK / name)
    old_model = RUNS / "tora_remain_prep_001/official_controllerTora_2026.onnx"
    model = NEW / "build/official_controllerTora_2026.onnx"
    shutil.copy2(old_model, model)
    shutil.copy2(RUNS / "run_archcomp26_tora_remain_native_pair.sh", NEW / "run_native_pair.sh")

    receipt = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": "native_tora_remain_h005_probe_20261002",
        "frozen_build": str(OLD / "BUILD.json"),
        "source_change": {"old": OLD_TEXT, "new": NEW_TEXT},
        "other_source_changes": [],
        "model_copy": str(model),
        "runner_copy": str(NEW / "run_native_pair.sh"),
        "builds": [],
    }
    for entry in old_build["builds"]:
        name = "smoke1" if entry["periods"] == 1 else "full20"
        old_source, old_binary = Path(entry["source"]), Path(entry["binary"])
        base = old_source.read_text()
        if base.count(OLD_TEXT) != 1 or NEW_TEXT in base:
            raise RuntimeError(f"unexpected frozen {name} fixed-step setting")
        if base.count(f"int steps = {entry['periods']};") != 1:
            raise RuntimeError(f"unexpected frozen {name} period count")
        source, binary = WORK / f"tora_remain_h005_{name}.cpp", WORK / f"tora_remain_h005_{name}"
        source.write_text(base.replace(OLD_TEXT, NEW_TEXT))
        # Exact string reconstruction is the scope check; no digest operation.
        if source.read_text().replace(NEW_TEXT, OLD_TEXT) != base:
            raise RuntimeError(f"{name}: more than the fixed-step line changed")
        old_command = entry["command"]
        if old_command.count(str(old_source)) != 1 or old_command.count(str(old_binary)) != 1:
            raise RuntimeError(f"unexpected frozen {name} compile command")
        command = [str(source) if value == str(old_source) else
                   str(binary) if value == str(old_binary) else value for value in old_command]
        log_path = NEW / f"{name}.build.log"
        with log_path.open("x") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        receipt["builds"].append({"periods": entry["periods"], "source": str(source),
                                  "source_bytes": source.stat().st_size, "binary": str(binary),
                                  "binary_bytes": binary.stat().st_size if binary.is_file() else None,
                                  "command": command, "returncode": completed.returncode,
                                  "build_log": str(log_path)})
        (NEW / "BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if completed.returncode:
            raise RuntimeError(f"{name} build failed; inspect the isolated build log")
    print(json.dumps({"status": "built", "root": str(NEW)}))


if __name__ == "__main__":
    main()
