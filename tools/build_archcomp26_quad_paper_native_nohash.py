#!/usr/bin/env python3
"""Build an isolated native Flow* QUAD using the 2026 paper equations."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/native_build"
SOURCE = OLD / "archcomp/Quadrotor/quad_matched_threads4.cpp"
NEW = N / "runs/archcomp26_20261001/native_quad_paper_build_001"
WORK = NEW / "archcomp/Quadrotor"

REPLACEMENTS = {
    'cos(x8)*sin(x9)*x4 + (sin(x7)*sin(x8)*sin(x9) - cos(x7)*cos(x9))*x5 + (cos(x7)*sin(x8)*sin(x9) + sin(x7)*cos(x9))*x6':
        'cos(x8)*sin(x9)*x4 + (sin(x7)*sin(x8)*sin(x9) + cos(x7)*cos(x9))*x5 + (cos(x7)*sin(x8)*sin(x9) - sin(x7)*cos(x9))*x6',
    'x12*x5 * x11*x6 - 9.81 *sin(x8)':
        'x12*x5 - x11*x6 - 9.81*sin(x8)',
    'x10*x6 - x11*x6 - 9.81 *sin(x8)':
        'x10*x6 - x12*x4 + 9.81*cos(x8)*sin(x7)',
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    for name in ("arch_ranges.h", "matched_reach.h"):
        shutil.copy2(SOURCE.parent / name, WORK / name)

    base = SOURCE.read_text()
    for old, new in REPLACEMENTS.items():
        if base.count(old) != 1:
            raise RuntimeError(f"expected one original equation: {old}")
        base = base.replace(old, new)
    if base.count("int steps = 50;") != 1:
        raise RuntimeError("unexpected control period declaration")

    server_source = (SOURCE.parent / "crown.py").read_text()
    old_model = 'model_path = os.path.join(benchmark_dir, "quad_controller_3_64_torch.onnx")'
    if server_source.count(old_model) != 1:
        raise RuntimeError("unexpected QUAD controller model selection")
    server_source = server_source.replace(old_model, 'model_path = os.environ["QUAD_MODEL"]\nassert os.path.isfile(model_path)')
    server_source = ('import hashlib\nimport json\n'
                     'def _no_digest(*args, **kwargs):\n'
                     '    raise RuntimeError("SHA-256 disabled for this new run")\n'
                     '_original_hash_new = hashlib.new\n'
                     'def _guarded_hash_new(name, *args, **kwargs):\n'
                     '    if name.lower().replace("-", "") == "sha256":\n'
                     '        return _no_digest()\n'
                     '    return _original_hash_new(name, *args, **kwargs)\n'
                     'hashlib.sha256 = _no_digest\n'
                     'hashlib.new = _guarded_hash_new\n'
                     + server_source)
    server_source = server_source.replace('    return coefficients\n',
        '    log_path = os.environ.get("QUAD_RPC_LOG")\n'
        '    if log_path:\n'
        '        with open(log_path, "a") as log:\n'
        '            log.write(json.dumps({"input_boxes": len(input_lb), "output_boxes": len(coefficients["T"])}) + "\\n")\n'
        '    return coefficients\n')
    (WORK / "crown_paper.py").write_text(server_source)

    library = OLD / "flowstar/flowstar-toolbox/libflowstar.a"
    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    command_prefix = ["taskset", "-c", "10", "/usr/bin/g++-15", "-O3", "-g", "-std=c++11",
        "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
        "-I", "/usr/include/jsoncpp"]
    libraries = [str(library), "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk",
                 "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp", "-lcurl",
                 "-ljsonrpccpp-common", "-ljsonrpccpp-client", "-l:libboost_thread.so.1.90.0"]
    record = {"built_utc": datetime.now(timezone.utc).isoformat(), "source_parent": str(SOURCE),
              "new_build_root": str(NEW), "equation_replacements": REPLACEMENTS,
              "controller_server": str(WORK / "crown_paper.py"),
              "source_identity_policy": "paths, modification times and byte sizes only; no content digest",
              "builds": []}
    for periods, label in ((1, "smoke1"), (50, "full50")):
        source = WORK / f"quad_paper_{label}.cpp"
        binary = WORK / f"quad_paper_{label}"
        source.write_text(base.replace("int steps = 50;", f"int steps = {periods};"))
        command = command_prefix + [str(source)] + libraries + ["-o", str(binary)]
        with (NEW / f"{label}.build.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        row = {"periods": periods, "source": str(source), "binary": str(binary),
               "command": command, "returncode": process.returncode,
               "source_bytes": source.stat().st_size,
               "binary_bytes": binary.stat().st_size if binary.is_file() else None}
        record["builds"].append(row)
        write_json(NEW / "BUILD.json", record)
        if process.returncode:
            raise RuntimeError(f"{label} build failed; see build log")
    print(json.dumps({"status": "built", "root": str(NEW), "binaries": [x["binary"] for x in record["builds"]]}))


if __name__ == "__main__":
    main()
