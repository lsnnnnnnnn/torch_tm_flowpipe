#!/usr/bin/env python3
"""Build isolated native TORA remain binaries for the 2026 paper contract."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
SOURCE = OLD / "archcomp/tora_homogeneous/matched_threads4.cpp"
NEW = N / "runs/archcomp26_20261001/native_tora_remain_build_001"
WORK = NEW / "archcomp/TORA"
MODEL = N / "runs/archcomp26_20261001/tora_remain_prep_001/official_controllerTora_2026.onnx"


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    for name in ("arch_ranges.h", "matched_reach.h"):
        shutil.copy2(SOURCE.parent / name, WORK / name)

    base = SOURCE.read_text()
    # This source already has the paper RHS, full initial set and per-segment safeSet.
    expected = ('"x2",', '"-x1 + 0.1 * sin(x3)"', '"x4",', '"u - 10"',
                'Interval init_x1(0.6, 0.7)', 'init_x2(-0.7, -0.6)',
                'init_x3(-0.4, -0.3)', 'init_x4(0.5, 0.6)',
                'initial_sets.size()', 'arch_ranges::record(results,4)')
    for fragment in expected:
        if fragment not in base:
            raise RuntimeError(f"unexpected saved TORA source: {fragment}")
    if base.count("int steps = 20;") != 1 or base.count("int final_result = 0;") != 1:
        raise RuntimeError("unexpected loop or property declaration")
    base = '#include <atomic>\n' + base.replace("int final_result = 0;", "std::atomic<int> final_result(0);")
    if base.count("    return 0;\n}") != 1:
        raise RuntimeError("unexpected final native exit")
    base = base.replace("    return 0;\n}", "    return final_result == 0 ? 0 : 2;\n}")

    server = (SOURCE.parent / "observed_server.py").read_text()
    old_model = 'model_path = os.path.join(benchmark_dir, "controllerTora.onnx")'
    if server.count(old_model) != 1:
        raise RuntimeError("unexpected TORA server model selection")
    server = server.replace(old_model, 'model_path = os.environ["TORA_MODEL"]\nassert os.path.isfile(model_path)')
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
    prefix = ["taskset", "-c", "18", "/usr/bin/g++-15", "-O3", "-std=c++11",
              "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
              "-I", "/usr/include/jsoncpp"]
    libraries = [str(library), "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm",
                 "-lglpk", "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp",
                 "-lcurl", "-ljsonrpccpp-common", "-ljsonrpccpp-client",
                 "-l:libboost_thread.so.1.90.0"]
    record = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "saved_source": str(SOURCE), "build_root": str(NEW),
              "model": str(MODEL), "source_identity_policy": "path and byte size; no content digest",
              "changes": ["atomic concurrent property status", "nonzero exit on incomplete property",
                          "explicit paper model path", "one-period and full-period binaries"],
              "builds": []}
    for periods, label in ((1, "smoke1"), (20, "full20")):
        source = WORK / f"tora_remain_{label}.cpp"
        binary = WORK / f"tora_remain_{label}"
        source.write_text(base.replace("int steps = 20;", f"int steps = {periods};"))
        command = prefix + [str(source)] + libraries + ["-o", str(binary)]
        with (NEW / f"{label}.build.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record["builds"].append({"periods": periods, "source": str(source),
                                 "source_bytes": source.stat().st_size,
                                 "binary": str(binary), "command": command,
                                 "returncode": process.returncode,
                                 "binary_bytes": binary.stat().st_size if binary.exists() else None})
        (NEW / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"{label} build failed; inspect build log")
    print(json.dumps({"status": "built", "root": str(NEW)}))


if __name__ == "__main__":
    main()
