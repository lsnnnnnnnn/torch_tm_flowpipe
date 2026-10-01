#!/usr/bin/env python3
"""Build isolated one-period Flow* Airplane full-box diagnostics."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
SOURCE = OLD / "archcomp/airplane/matched_threads4.cpp"
NEW = N / "runs/archcomp26_20261001/native_airplane_fullbox_build_001"
WORK = NEW / "archcomp/airplane"
MODEL = N / "runs/archcomp26_20261001/airplane_prep_001/controller_airplane.onnx"


def once(content, old, new):
    if content.count(old) != 1:
        raise RuntimeError(f"expected exactly one saved source marker: {old[:100]}")
    return content.replace(old, new)


def main():
    if not SOURCE.is_file() or not MODEL.is_file():
        raise FileNotFoundError((SOURCE, MODEL))
    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    for name in ("arch_ranges.h", "matched_reach.h"):
        shutil.copy2(SOURCE.parent / name, WORK / name)

    base = '#include <cmath>\n#include <fstream>\n#include <iomanip>\n#include <stdexcept>\n' + SOURCE.read_text()
    base = once(base, 'HttpClient httpclient("http://127.0.0.1:5100");',
                'HttpClient httpclient("http://127.0.0.1:5105");')
    for name in ("u", "v", "w"):
        base = once(base, f"init_{name}(1, 1)", f"init_{name}(0, 1)")
    for name in ("phi", "theta", "psi"):
        base = once(base, f"init_{name}(0.9, 0.9)", f"init_{name}(0, 1)")
    base = once(base, "int steps = 20;", "int steps = 1;")
    old_controls_start = "        Matrix<Real> T(num_nn_output, num_nn_input, Real(0));"
    old_controls_end = "        initial_set.tmvPre.tms[Fx_id] = tmv_output.tms[0];"
    if base.count(old_controls_start) != 1 or base.count(old_controls_end) != 1:
        raise RuntimeError("saved controller injection changed")
    start = base.index(old_controls_start)
    end = base.index(old_controls_end)
    new_controls = """        vector<Real> zeros(num_nn_output, Real(0));
        TaylorModelVec<Real> tmv_output(zeros, numVars);
        for (unsigned int j = 0; j < num_nn_output; ++j) {
            for (unsigned int i = 0; i < num_nn_input; ++i) {
                double slope = output_coefficients["T"][j][i].asDouble();
                if (!std::isfinite(slope)) throw std::runtime_error("nonfinite NN slope");
                tmv_output.tms[j] += initial_set.tmvPre.tms[i] * Real(slope);
            }
            double lo = output_coefficients["u_min"][j].asDouble();
            double hi = output_coefficients["u_max"][j].asDouble();
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
                throw std::runtime_error("invalid NN bias interval");
            tmv_output.tms[j].remainder += Interval(lo, hi);
        }

"""
    base = base[:start] + new_controls + base[end:]
    old_reach_start = "        // Flow*\n"
    old_reach_end = "    if (final_result == 0)\n"
    if base.count(old_reach_start) != 1 or base.count(old_reach_end) != 1:
        raise RuntimeError("saved reach/status loop changed")
    start = base.index(old_reach_start)
    end = base.index(old_reach_end)
    new_reach = """        // Flow*: keep the full local tube and stop on any non-safe or incomplete step.
        size_t prior = result.flowpipes.size();
        author_matched::reach(dynamics, result, initial_set, 0.1, setting, safeSet, symbolic_remainder);
        arch_ranges::record(result, 0, 12);
        bool independently_box_safe = true;
        size_t index = 0;
        const int safe_ids[] = {y_id, phi_id, theta_id, psi_id};
        for (const Flowpipe &fp : result.flowpipes) {
            ++index;
            if (index <= prior) continue;
            vector<Interval> tube;
            fp.tmvPre.intEval(tube, fp.domain);
            if (tube.size() < 12) throw std::runtime_error("short Airplane tube");
            Interval cosine = tube[theta_id].cos();
            checks << iter << '\\t' << (index - prior) << '\\t' << index;
            for (int id : safe_ids) {
                checks << '\\t' << tube[id].inf() << '\\t' << tube[id].sup();
                if (tube[id].inf() < -1 || tube[id].sup() > 1) independently_box_safe = false;
            }
            checks << '\\t' << cosine.inf() << '\\t' << cosine.sup() << '\\n';
            if (cosine.inf() <= 0) independently_box_safe = false;
        }
        checks.flush();
        if (!checks) throw std::runtime_error("Airplane check output failed");
        size_t produced = result.flowpipes.size() - prior;
        cout << "PERIOD " << iter << " FLOWPIPES " << produced
             << " STATUS " << result.status << " BOX_SAFE " << independently_box_safe << endl;
        if (result.status != COMPLETED_SAFE || produced != 10 || !independently_box_safe) {
            final_result = 2;
            break;
        }
        initial_set = result.fp_end_of_time;
        ++completed_periods;
    }
    cout << "COMPLETED_SAFE_PERIODS " << completed_periods << "/" << steps << endl;
    cout << "FLOWPIPE_SEGMENTS " << result.flowpipes.size() << endl;
"""
    base = base[:start] + new_reach + base[end:]
    base = once(base, "    int final_result = 0;",
                "    int final_result = 0;\n    int completed_periods = 0;\n"
                '    const char *check_path = std::getenv("AIRPLANE_CHECK_LOG");\n'
                '    if (!check_path) throw std::runtime_error("AIRPLANE_CHECK_LOG required");\n'
                '    std::ofstream checks(check_path);\n'
                '    if (!checks) throw std::runtime_error("check log unavailable");\n'
                '    checks << "period\\tlocal_substep\\tglobal_substep\\ty_lo\\ty_hi\\tphi_lo\\tphi_hi\\ttheta_lo\\ttheta_hi\\tpsi_lo\\tpsi_hi\\tcos_theta_lo\\tcos_theta_hi\\n" << std::setprecision(17);')
    base = once(base, "    return 0;\n}", "    return final_result == 0 ? 0 : 2;\n}")

    server = (SOURCE.parent / "observed_server.py").read_text()
    server = once(server, "('127.0.0.1', 5100)", "('127.0.0.1', 5105)")
    server = once(server,
                  'model_path = os.path.join(benchmark_dir, "controller_airplane.onnx")',
                  'model_path = os.environ["AIRPLANE_MODEL"]\nassert os.path.isfile(model_path)')
    server = once(server,
                  "    A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]",
                  "    A = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]\n"
                  "    if not torch.equal(A['lA'], A['uA']):\n"
                  "        raise RuntimeError('Airplane CROWN upper/lower slopes differ')")
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
              "saved_source": str(SOURCE), "source_bytes": SOURCE.stat().st_size,
              "build_root": str(NEW), "model": str(MODEL),
              "changes": ["full official initial box", "one-period smoke", "dedicated RPC port",
                          "direct NN bias interval and double slopes", "stop on any unknown/noncompletion",
                          "save all 12-state tubes and independent box safety/denominator checks"],
              "source_identity_policy": "path and byte size; no content digest", "builds": []}
    for order in (6, 3):
        source = WORK / f"airplane_fullbox_order{order}_smoke1.cpp"
        binary = WORK / f"airplane_fullbox_order{order}_smoke1"
        content = once(base, "setting.setFixedStepsize(0.01, 6);",
                       f"setting.setFixedStepsize(0.01, {order});")
        source.write_text(content)
        command = prefix + [str(source)] + libraries + ["-o", str(binary)]
        with (NEW / f"order{order}.build.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record["builds"].append({"order": order, "periods": 1,
                                 "source": str(source), "binary": str(binary),
                                 "command": command, "returncode": process.returncode,
                                 "binary_bytes": binary.stat().st_size if binary.exists() else None})
        (NEW / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"order{order} compile failed; inspect build log")
    print(json.dumps({"status": "built", "root": str(NEW)}))


if __name__ == "__main__":
    main()
