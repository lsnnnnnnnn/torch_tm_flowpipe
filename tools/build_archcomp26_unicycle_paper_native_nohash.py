#!/usr/bin/env python3
"""Build one isolated native paper Unicycle prefix without content digests."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
SOURCE = N / "tools/archcomp26_unicycle_paper_native_nohash.cpp"
PERIODS = 50 if sys.argv[1:] == ["--full50"] else 1
if sys.argv[1:] not in ([], ["--full50"]):
    raise SystemExit("usage: build_archcomp26_unicycle_paper_native_nohash.py [--full50]")
LABEL = "full50" if PERIODS == 50 else "smoke1"
NEW = N / ("runs/archcomp26_20261001/native_unicycle_paper_speed_build_full50_001"
           if PERIODS == 50 else
           "runs/archcomp26_20261001/native_unicycle_paper_speed_build_002")
WORK = NEW / "archcomp/unicycle"
MODEL = N / "runs/archcomp26_20261001/unicycle_prep_001/controllerB_2026.onnx"


def once(content, old, new):
    if content.count(old) != 1:
        raise RuntimeError(f"expected exactly one source marker: {old}")
    return content.replace(old, new)


def main():
    if not SOURCE.is_file() or not MODEL.is_file():
        raise FileNotFoundError((SOURCE, MODEL))
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    reference = OLD / "archcomp/unicycle"
    for name in ("arch_ranges.h", "matched_reach.h"):
        shutil.copy2(reference / name, WORK / name)
    source = WORK / SOURCE.name
    shutil.copy2(SOURCE, source)

    server = (reference / "observed_server.py").read_text()
    server = once(server, "('127.0.0.1', 5100)", "('127.0.0.1', 5107)")
    server = once(server,
                  'model_path = os.path.join(benchmark_dir, "controllerB.onnx")',
                  'model_path = os.environ["UNICYCLE_MODEL"]\nassert os.path.isfile(model_path)')
    (WORK / "crown_paper.py").write_text(server)

    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    library = OLD / "flowstar/flowstar-toolbox/libflowstar.a"
    binary = WORK / f"unicycle_{LABEL}"
    command = ["taskset", "-c", "24", "/usr/bin/g++-15", "-O3", "-std=c++11",
               "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
               "-I", "/usr/include/jsoncpp", f"-DUNICYCLE_PERIODS={PERIODS}", str(source),
               str(library), "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm",
               "-lglpk", "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp",
               "-lcurl", "-ljsonrpccpp-common", "-ljsonrpccpp-client",
               "-l:libboost_thread.so.1.90.0", "-o", str(binary)]
    with (NEW / f"{LABEL}.build.log").open("x") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    record = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(SOURCE), "source_bytes": SOURCE.stat().st_size,
              "build_root": str(NEW), "controller": str(MODEL),
              "profile": "2026 paper RHS; constant w in speed only; repaired full initial box",
              "periods": PERIODS,
              "command": command, "returncode": process.returncode,
              "binary_bytes": binary.stat().st_size if binary.exists() else None}
    (NEW / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
    if process.returncode:
        raise RuntimeError("Unicycle native compile failed; inspect build log")
    print(json.dumps({"status": "built", "binary": str(binary)}))


if __name__ == "__main__":
    main()
