#!/usr/bin/env python3
"""Build one isolated Airplane order-3, wider-remainder diagnostic."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
PREVIOUS = N / "runs/archcomp26_20261001/native_airplane_fullbox_build_001/archcomp/airplane"
SOURCE = PREVIOUS / "airplane_fullbox_order3_smoke1.cpp"
NEW = N / "runs/archcomp26_20261001/native_airplane_fullbox_build_002"
WORK = NEW / "archcomp/airplane"


def main():
    content = SOURCE.read_text()
    original = "Interval I(-0.01, 0.01);"
    if content.count(original) != 1 or content.count("setting.setFixedStepsize(0.01, 3);") != 1:
        raise RuntimeError("unexpected saved order-3 source")
    content = content.replace(original, "Interval I(-1, 1);")
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    for name in ("arch_ranges.h", "matched_reach.h", "crown_paper.py"):
        shutil.copy2(PREVIOUS / name, WORK / name)
    source = WORK / "airplane_fullbox_order3_remainder1_smoke1.cpp"
    binary = WORK / "airplane_fullbox_order3_remainder1_smoke1"
    source.write_text(content)
    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    library = OLD / "flowstar/flowstar-toolbox/libflowstar.a"
    command = ["taskset", "-c", "18", "/usr/bin/g++-15", "-O3", "-std=c++11",
               "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
               "-I", "/usr/include/jsoncpp", str(source), str(library),
               "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk",
               "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp", "-lcurl",
               "-ljsonrpccpp-common", "-ljsonrpccpp-client", "-l:libboost_thread.so.1.90.0",
               "-o", str(binary)]
    with (NEW / "build.log").open("x") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "new_build_root": str(NEW), "saved_source": str(SOURCE),
              "source_bytes": source.stat().st_size,
              "change": "only Flow* remainder estimate [-0.01,0.01] -> [-1,1]; order 3 and full 2026 box unchanged",
              "source_identity_policy": "path and byte size; no content digest",
              "source": str(source), "binary": str(binary),
              "command": command, "returncode": process.returncode,
              "binary_bytes": binary.stat().st_size if binary.exists() else None}
    (NEW / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
    if process.returncode:
        raise RuntimeError("compile failed; inspect build.log")
    print(json.dumps({"status": "built", "root": str(NEW)}))


if __name__ == "__main__":
    main()
