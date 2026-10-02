#!/usr/bin/env python3
"""Build an isolated Flow* CartPole raw4 entry from the fixed repository."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
SOURCE = N / "tools/archcomp26_balancing_raw4_native_nohash.cpp"
NEW = N / "runs/archcomp26_20261001/native_balancing_raw4_build_002"
WORK = NEW / "archcomp/balancing_raw4"
MODEL = Path("/srv/local/shengenli/CROWN-Reach/ARCH-COMP2024/benchmarks/CartPole/model.onnx")


def once(content, old, new):
    if content.count(old) != 1:
        raise RuntimeError(f"expected exactly one source marker: {old[:90]}")
    return content.replace(old, new)


def main():
    if not SOURCE.is_file() or not MODEL.is_file():
        raise FileNotFoundError((SOURCE, MODEL))
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    reference = OLD / "archcomp/tora_homogeneous"
    for name in ("arch_ranges.h", "matched_reach.h"):
        shutil.copy2(reference / name, WORK / name)
    source = WORK / SOURCE.name
    shutil.copy2(SOURCE, source)

    server = (reference / "observed_server.py").read_text()
    server = once(server, "('127.0.0.1', 5100)", "('127.0.0.1', 5106)")
    server = once(server,
                  'model_path = os.path.join(benchmark_dir, "controllerTora.onnx")',
                  'model_path = os.environ["BALANCING_MODEL"]\nassert os.path.isfile(model_path)')
    server = once(server, "input_shape = (-1, 1, 1, 4)", "input_shape = (-1, 4)")
    # The CartPole controller is 4->1, matching the original server output layout.
    server = ('import hashlib\n'
              'def _digest_disabled(*args, **kwargs):\n'
              '    raise RuntimeError("content digest disabled for this experiment")\n'
              '_old_new = hashlib.new\n'
              'def _guarded_new(name, *args, **kwargs):\n'
              '    if name.lower().replace("-", "") == "sha256":\n'
              '        return _digest_disabled()\n'
              '    return _old_new(name, *args, **kwargs)\n'
              'hashlib.sha256 = _digest_disabled\n'
              'hashlib.new = _guarded_new\n' + server)
    (WORK / "crown_paper.py").write_text(server)

    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    library = OLD / "flowstar/flowstar-toolbox/libflowstar.a"
    prefix = ["taskset", "-c", "24", "/usr/bin/g++-15", "-O3", "-std=c++11",
              "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
              "-I", "/usr/include/jsoncpp"]
    libraries = [str(library), "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm",
                 "-lglpk", "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp",
                 "-lcurl", "-ljsonrpccpp-common", "-ljsonrpccpp-client",
                 "-l:libboost_thread.so.1.90.0"]
    record = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(SOURCE), "source_bytes": SOURCE.stat().st_size,
              "build_root": str(NEW), "model": str(MODEL),
              "source_identity_policy": "path and byte size; no content digest",
              "profile": "fixed-repository raw4 CartPole full initial box, 4->1 controller, 0.02 s hold, 0.005 s order 6 Flow*, [8,10] continuous-tube target checker",
              "builds": []}
    for periods, label in ((1, "smoke1"), (500, "full500")):
        binary = WORK / f"balancing_{label}"
        command = prefix + [f"-DBALANCING_PERIODS={periods}", str(source)] + libraries + ["-o", str(binary)]
        with (NEW / f"{label}.build.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record["builds"].append({"periods": periods, "binary": str(binary),
                                 "command": command, "returncode": process.returncode,
                                 "binary_bytes": binary.stat().st_size if binary.exists() else None})
        (NEW / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"{label} compile failed; inspect build log")
    print(json.dumps({"status": "built", "root": str(NEW)}))


if __name__ == "__main__":
    main()
