#!/usr/bin/env python3
"""Build an isolated ACC participant-order native binary with the saved VAR-tail fix."""

import json
from pathlib import Path
import subprocess
import sys


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
SOURCE = N / "runs/quad_residual_memory_20260923/native_var_tail_fix/acc_matched.cpp"
SERVER = N / "runs/archcomp_review_20260923/suite_build/archcomp/acc/observed_server.py"
FIXED = N / "runs/quad_residual_memory_20260923/native_var_tail_fix/flowstar"
MODEL = N / "runs/archcomp26_20261001/acc_prep_001/official_acc_controller_5_20.onnx"
OUT = N / "runs/archcomp26_20261001/acc_native_var_tail_build_001"
PORT = 5102


def replace_once(value, old, new):
    if value.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}")
    return value.replace(old, new)


def main():
    if not all(p.is_file() for p in (SOURCE, SERVER, FIXED / "libflowstar.a", MODEL)):
        raise FileNotFoundError("missing saved ACC source, RPC source, corrected library or fixed model")
    expression = (FIXED / "expression.h").read_text()
    for token in ("Interval input_remainder = result.remainder;", "intermediate_ranges.push_back(result.remainder);", "result += *iter;"):
        if token not in expression:
            raise RuntimeError(f"saved VAR-tail correction is missing {token!r}")
    source = replace_once(SOURCE.read_text(), "http://127.0.0.1:5100", f"http://127.0.0.1:{PORT}")
    if "tm_v_rel = initial_set.tmvPre.tms[v_lead_id] - initial_set.tmvPre.tms[v_ego_id]" not in source:
        raise RuntimeError("participant velocity feature sign changed")
    if "Constraint c_temp(\"-x_lead + x_ego + 1.4 * v_ego + 10\"" not in source:
        raise RuntimeError("ACC safe halfspace changed")
    server = SERVER.read_text()
    server = replace_once(server, "('127.0.0.1', 5100)", f"('127.0.0.1', {PORT})")
    server = replace_once(server, 'CROWN_DIR = "../../Verifier_Development/complete_verifier"', 'CROWN_DIR = "/srv/local/shengenli/auto_LiRPA"')
    server = replace_once(server, 'model_path = os.path.join(benchmark_dir, "controller_5_20.onnx")', f'model_path = "{MODEL}"')
    OUT.mkdir()
    src = OUT / "acc.cpp"
    srv = OUT / "observed_server.py"
    src.write_text(source)
    srv.write_text(server)
    binary = OUT / "acc_native_var_tail"
    command = [
        "/usr/bin/g++-15", "-O3", "-std=c++11", "-fpermissive", "-Wno-template-body", "-fopenmp",
        "-I", str(FIXED),
        "-I", str(N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"),
        "-I", "/usr/include/jsoncpp", str(src), str(FIXED / "libflowstar.a"),
        "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm", "-lglpk", "-lcolamd", "-lamd",
        "-lz", "-lltdl", "-ljsoncpp", "-lcurl", "-ljsonrpccpp-common", "-ljsonrpccpp-client",
        "-l:libboost_thread.so.1.90.0", "-o", str(binary),
    ]
    (OUT / "BUILD.json").write_text(json.dumps({
        "source": str(SOURCE), "corrected_flowstar_library": str(FIXED / "libflowstar.a"),
        "fixed_model": str(MODEL), "new_source": str(src), "new_server": str(srv),
        "new_binary": str(binary), "port": PORT, "command": command,
        "contract": "ACC saved participant feature order, corrected native VAR truncation-tail replay",
        "content_digest_performed": False,
    }, indent=2) + "\n")
    with (OUT / "build.log").open("wb") as log:
        result = subprocess.run(command, cwd=OUT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f"build failed: {result.returncode}; see {OUT / 'build.log'}", file=sys.stderr)
        return result.returncode
    print(json.dumps({"binary": str(binary), "server": str(srv), "port": PORT, "bytes": binary.stat().st_size}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
