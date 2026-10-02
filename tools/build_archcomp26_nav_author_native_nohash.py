#!/usr/bin/env python3
"""Build isolated NAV standard native binaries from the saved author profile."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
SOURCE = OLD / "archcomp/nav_standard/matched_threads4.cpp"
NEW = N / "runs/archcomp26_20261001/native_nav_standard_author_build_001"
WORK = NEW / "archcomp/nav_standard"
BOXES = N / "runs/archcomp_review_20260923/contracts/nav_standard_boxes.json"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"saved NAV source marker differs: {old[:80]}")
    return source.replace(old, new)


def main():
    ledger = json.loads(BOXES.read_text())
    if len(ledger) != 640 or len(ledger[0]) != 7 or ledger[0][0] != [2.9, 2.9050000000000002] or ledger[0][1] != [2.9, 2.9125]:
        raise ValueError("saved NAV standard initial partition differs")
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    for name in ("arch_ranges.h", "matched_reach.h"):
        shutil.copy2(SOURCE.parent / name, WORK / name)
    (NEW / "first_box.json").write_text(json.dumps(ledger[:1]) + "\n")

    base = SOURCE.read_text()
    for fragment in ('"x3 * cos(x4)"', '"x3 * sin(x4)"', '"u1"', '"u2"',
                     'Interval init_x1(2.9, 3.1), init_x2(2.9, 3.1)',
                     'init_x1.split(list_x1, 40)', 'init_x2.split(list_x2, 16)',
                     '"-x1 + 1", "x1 - 2"', '"-x1 - 0.5", "x1 - 0.5"'):
        if fragment not in base:
            raise ValueError(f"saved NAV author contract differs: {fragment}")
    base = replace_once(base, 'HttpClient httpclient("http://127.0.0.1:5100");',
                        'HttpClient httpclient("http://127.0.0.1:5110");')
    base = replace_once(base, '    vector<Symbolic_Remainder> symbolic_remainders;',
                        '    author_matched::initial(initial_sets);\n'
                        '    vector<Symbolic_Remainder> symbolic_remainders;')
    base = replace_once(base, '    int steps = 30;',
                        '    int steps = NAV_PERIODS;\n    int completed_periods = 0;')
    base = replace_once(base,
        '        arch_ranges::record(results,4);\n        if (terminate.load())',
        '        arch_ranges::record(results,4);\n'
        '        for (const auto &r : results)\n'
        '            if (r.flowpipes.size() != static_cast<size_t>((iter + 1) * 20)) terminate.store(true);\n'
        '        if (!terminate.load()) ++completed_periods;\n'
        '        if (terminate.load())')
    base = replace_once(base, '    // Check obstacle\n',
        '    cout << "COMPLETED_PERIODS " << completed_periods << "/" << steps << endl;\n'
        '    if (steps == 1) {\n'
        '        cout << (completed_periods == 1 ? "PREFIX_COMPLETE_ONLY" : "PREFIX_INCOMPLETE") << endl;\n'
        '        return completed_periods == 1 ? 0 : 2;\n'
        '    }\n'
        '    if (completed_periods != steps) return 2;\n'
        '    // Check obstacle\n')

    server = (SOURCE.parent / "observed_server.py").read_text()
    server = replace_once(server, "('127.0.0.1', 5100)", "('127.0.0.1', 5110)")
    server = replace_once(server,
        'model_path = os.path.join(benchmark_dir, "networks/nn-nav-point.onnx")',
        'model_path = os.environ["NAV_MODEL"]\nassert os.path.isfile(model_path)')
    server = ('import hashlib\n'
              'def _no_digest(*args, **kwargs):\n'
              '    raise RuntimeError("content digest disabled for this run")\n'
              '_old_new = hashlib.new\n'
              'def _guarded_new(name, *args, **kwargs):\n'
              '    if name.lower().replace("-", "") == "sha256": return _no_digest()\n'
              '    return _old_new(name, *args, **kwargs)\n'
              'hashlib.sha256 = _no_digest\n'
              'hashlib.new = _guarded_new\n' + server)
    (WORK / "crown_author.py").write_text(server)

    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    library = OLD / "flowstar/flowstar-toolbox/libflowstar.a"
    prefix = ["taskset", "-c", "18", "/usr/bin/g++-15", "-O3", "-std=c++11",
              "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
              "-I", "/usr/include/jsoncpp"]
    libs = [str(library), "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm",
            "-lglpk", "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp",
            "-lcurl", "-ljsonrpccpp-common", "-ljsonrpccpp-client",
            "-l:libboost_thread.so.1.90.0"]
    record = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "saved_source": str(SOURCE), "official_profile_boxes": str(BOXES),
              "new_build_root": str(NEW), "source_identity": "path and size only",
              "builds": []}
    for label, periods, smoke in (("smoke1", 1, True), ("full30", 30, False)):
        content = base.replace("NAV_PERIODS", str(periods))
        if smoke:
            content = replace_once(content,
                'Interval init_x1(2.9, 3.1), init_x2(2.9, 3.1)',
                'Interval init_x1(2.9, 2.9050000000000002), init_x2(2.9, 2.9125)')
            content = replace_once(content, 'init_x1.split(list_x1, 40)',
                                   'init_x1.split(list_x1, 1)')
            content = replace_once(content, 'init_x2.split(list_x2, 16)',
                                   'init_x2.split(list_x2, 1)')
        source = WORK / f"nav_standard_{label}.cpp"
        binary = WORK / f"nav_standard_{label}"
        source.write_text(content)
        command = prefix + [str(source)] + libs + ["-o", str(binary)]
        with (NEW / f"{label}.build.log").open("x") as output:
            process = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        record["builds"].append({"label": label, "periods": periods,
                                 "expected_boxes": 1 if smoke else 640,
                                 "source": str(source), "source_bytes": source.stat().st_size,
                                 "binary": str(binary), "binary_bytes": binary.stat().st_size if binary.exists() else None,
                                 "command": command, "returncode": process.returncode})
        (NEW / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"{label} native build failed")
    print(json.dumps({"status": "built", "root": str(NEW)}))


if __name__ == "__main__":
    main()
