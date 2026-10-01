#!/usr/bin/env python3
"""Build isolated Attitude Control native binaries with the 2026 unsafe box."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
SOURCE = OLD / "archcomp/attitude_control/matched_threads4.cpp"
NEW = N / "runs/archcomp26_20261001/native_attitude_avoid_build_001"
WORK = NEW / "archcomp/attitude_control"
MODEL = N / "runs/archcomp26_20261001/attitude_prep_001/official_attitude_control_3_64_torch.onnx"


def once(content, old, new):
    if content.count(old) != 1:
        raise RuntimeError(f"expected exactly one source marker: {old[:90]}")
    return content.replace(old, new)


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    for name in ("arch_ranges.h", "matched_reach.h"):
        shutil.copy2(SOURCE.parent / name, WORK / name)

    base = SOURCE.read_text()
    expected = (
        '"0.25 * (u1 + x2 * x3)"',
        '"0.5 * (u2 - 3 * x1 * x3)"',
        '"u3 + 2 * x1 * x2"',
        '"0.5 * (x2 * (x4^2 + x5^2 + x6^2 - x6) + x3 * (x4^2 + x5^2 + x5 + x6^2) + x1 * (x4^2 + x5^2 + x6^2 + 1))"',
        '"0.5 * (x1 * (x4^2 + x5^2 + x6^2 + x6) + x3 * (x4^2 - x4 + x5^2 + x6^2) + x2 * (x4^2 + x5^2 + x6^2 + 1))"',
        '"0.5 * (x1 * (x4^2 + x5^2 - x5 + x6^2) + x2 * (x4^2 + x4 + x5^2 + x6^2) + x3 * (x4^2 + x5^2 + x6^2 + 1))"',
        'Interval init_x1(-0.45, -0.44)',
        'init_x2(-0.55, -0.54)',
        'init_x3(0.65, 0.66)',
        'init_x4(-0.75, -0.74)',
        'init_x5(0.85, 0.86)',
        'init_x6(-0.65, -0.64)',
        'arch_ranges::record(result,0,6)',
        'result.unsafetyChecking(unsafeSet, setting.tm_setting, setting.g_setting)',
    )
    for fragment in expected:
        if fragment not in base:
            raise RuntimeError(f"unexpected saved Attitude source: {fragment[:90]}")
    base = once(base, 'HttpClient httpclient("http://127.0.0.1:5100");',
                'HttpClient httpclient("http://127.0.0.1:5103");')
    base = once(base, '"-x4 - 0.4", "x4 + 0.6"',
                '"-x4 - 0.7", "x4 + 0.6"')
    base = once(base, "    int final_result = 0;",
                "    int final_result = 0;\n    int completed_periods = 0;")
    base = once(base, "            initial_set = result.fp_end_of_time;",
                "            initial_set = result.fp_end_of_time;\n            ++completed_periods;")
    base = once(base, "    // Check reachability\n",
                '    cout << "COMPLETED_PERIODS " << completed_periods << "/" << steps << endl;\n'
                '    cout << "FLOWPIPE_SEGMENTS " << result.flowpipes.size() << endl;\n'
                '    if (completed_periods != steps || result.flowpipes.size() != 2 * steps) final_result = 2;\n'
                "    // Check reachability\n")
    base = once(base, "    return 0;\n}",
                "    return final_result == 0 ? 0 : 2;\n}")

    server = (SOURCE.parent / "observed_server.py").read_text()
    server = once(server, "('127.0.0.1', 5100)", "('127.0.0.1', 5103)")
    server = once(server,
                  'model_path = os.path.join(benchmark_dir, "attitude_control_3_64_torch.onnx")',
                  'model_path = os.environ["ATTITUDE_MODEL"]\nassert os.path.isfile(model_path)')
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
              "old_source": str(SOURCE), "build_root": str(NEW), "model": str(MODEL),
              "property_fix": "x4 lower bound -0.7 replaces impossible -0.4",
              "source_identity_policy": "path and byte size, no content digest",
              "builds": []}
    for periods, label in ((1, "smoke1"), (30, "full30")):
        source = WORK / f"attitude_avoid_{label}.cpp"
        binary = WORK / f"attitude_avoid_{label}"
        content = once(base, "int steps = 30;", f"int steps = {periods};")
        if periods == 1:
            content = once(content, 'cout << "VERIFIED" << endl;',
                           'cout << "PREFIX_SAFE_NO_FULL_PROPERTY" << endl;')
            content = once(content, 'cout << "FALSIFIED" << endl;',
                           'cout << "PREFIX_UNSAFE_NO_FULL_PROPERTY" << endl;')
            content = once(content, 'cout << "UNKNOWN" << endl;',
                           'cout << "PREFIX_UNKNOWN_NO_FULL_PROPERTY" << endl;')
        source.write_text(content)
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
            raise RuntimeError(f"{label} compile failed; inspect build log")
    print(json.dumps({"status": "built", "root": str(NEW)}))


if __name__ == "__main__":
    main()
