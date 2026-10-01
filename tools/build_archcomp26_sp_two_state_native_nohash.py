#!/usr/bin/env python3
"""Build a separate native Single Pendulum with the 2026 closed time window."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
SOURCE = OLD / "archcomp/single_pendulum/matched_threads4.cpp"
NEW = N / "runs/archcomp26_20261001/native_sp_two_state_build_001"
WORK = NEW / "archcomp/single_pendulum"


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"expected one source marker: {old[:100]}")
    return text.replace(old, new)


def main():
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    for name in ("arch_ranges.h", "matched_reach.h"):
        shutil.copy2(SOURCE.parent / name, WORK / name)

    base = SOURCE.read_text()
    base = replace_one(base, 'HttpClient httpclient("http://127.0.0.1:5100");',
                       'HttpClient httpclient("http://127.0.0.1:5101");')
    base = replace_one(base, '    vector<Constraint> safeSet;\n    Result_of_Reachability result;',
                       '    const vector<Constraint> noSafety;\n'
                       '    vector<Constraint> windowSafety;\n'
                       '    windowSafety.emplace_back("-x1", vars);\n'
                       '    windowSafety.emplace_back("x1 - 1", vars);\n'
                       '    Result_of_Reachability result;\n'
                       '    int completedPeriods = 0;\n'
                       '    bool propertyUnsafe = false, propertyUnknown = false;')
    base = replace_one(base,
        '        author_matched::reach(dynamics, result, initial_set, 0.05, setting, safeSet, symbolic_remainder);',
        '        const auto & activeSafety = iter < 10 ? noSafety : windowSafety;\n'
        '        author_matched::reach(dynamics, result, initial_set, 0.05, setting, activeSafety, symbolic_remainder);')
    old_status = '''        if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
        {
            initial_set = result.fp_end_of_time;
        }
        else
        {
            cout << "Flow* terminated." << endl;
            break;
        }'''
    new_status = '''        if (result.status == COMPLETED_SAFE)
        {
            initial_set = result.fp_end_of_time;
            ++completedPeriods;
        }
        else
        {
            if (iter >= 10 && result.status == COMPLETED_UNSAFE)
                propertyUnsafe = true;
            else
                propertyUnknown = true;
            if (result.status != COMPLETED_UNSAFE && result.status != COMPLETED_UNKNOWN)
                cout << "Flow* terminated." << endl;
            break;
        }'''
    base = replace_one(base, old_status, new_status)
    start = base.index('    // Check reachability\n')
    end = base.index('    auto end = std::chrono::steady_clock::now();', start)
    base = base[:start] + '''    // Check only the official closed window [.5, 1] through periods 10..19.
    cout << "COMPLETED_PERIODS " << completedPeriods << "/" << steps << endl;
    if (propertyUnsafe)
        cout << "FALSIFIED" << endl;
    else if (propertyUnknown || completedPeriods != steps)
        cout << "UNKNOWN" << endl;
    else if (steps == 20)
        cout << "VERIFIED" << endl;
    else
        cout << "PLUMBING_COMPLETE_NO_PROPERTY" << endl;

''' + base[end:]
    if 'unsafetyChecking' in base:
        raise RuntimeError("old end-only checker remains")

    server = (SOURCE.parent / "observed_server.py").read_text()
    server = replace_one(server, "('127.0.0.1', 5100)", "('127.0.0.1', 5101)")
    server = replace_one(server,
        'model_path = os.path.join(benchmark_dir, "controller_single_pendulum.onnx")',
        'model_path = os.environ["SP_MODEL"]\nassert os.path.isfile(model_path)')
    server = ('import hashlib\n'
              'def _no_digest(*args, **kwargs):\n'
              '    raise RuntimeError("content digest disabled")\n'
              '_original_new = hashlib.new\n'
              'def _guarded_new(name, *args, **kwargs):\n'
              '    if name.lower().replace("-", "") == "sha256":\n'
              '        return _no_digest()\n'
              '    return _original_new(name, *args, **kwargs)\n'
              'hashlib.sha256 = _no_digest\n'
              'hashlib.new = _guarded_new\n' + server)
    (WORK / "observed_server_paper.py").write_text(server)

    library = OLD / "flowstar/flowstar-toolbox/libflowstar.a"
    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    prefix = ["taskset", "-c", "10", "/usr/bin/g++-15", "-O3", "-g", "-std=c++11",
              "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
              "-I", "/usr/include/jsoncpp"]
    libraries = [str(library), "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk",
                 "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp", "-lcurl",
                 "-ljsonrpccpp-common", "-ljsonrpccpp-client", "-l:libboost_thread.so.1.90.0"]
    receipt = {"built_utc": datetime.now(timezone.utc).isoformat(),
               "old_source": str(SOURCE), "new_build_root": str(NEW),
               "controller_server": str(WORK / "observed_server_paper.py"),
               "controller_env": "SP_MODEL", "client_port": 5101, "server_port": 5101,
               "property_window": [0.5, 1.0], "physical_states": ["x1", "x2"],
               "source_identity_policy": "path and byte size only; no content digest",
               "builds": []}
    for periods, label in ((1, "smoke1"), (20, "full20")):
        source = WORK / f"sp_two_state_{label}.cpp"
        binary = WORK / f"sp_two_state_{label}"
        source.write_text(replace_one(base, "int steps = 20;", f"int steps = {periods};"))
        command = prefix + [str(source)] + libraries + ["-o", str(binary)]
        with (NEW / f"{label}.build.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        receipt["builds"].append({"periods": periods, "source": str(source),
                                  "binary": str(binary), "command": command,
                                  "returncode": process.returncode,
                                  "source_bytes": source.stat().st_size,
                                  "binary_bytes": binary.stat().st_size if binary.is_file() else None})
        (NEW / "BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"{label} build failed; see build log")
    print(json.dumps({"status": "built", "root": str(NEW),
                      "binaries": [b["binary"] for b in receipt["builds"]]}))


if __name__ == "__main__":
    main()
