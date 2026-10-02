#!/usr/bin/env python3
"""Build an isolated native TORA reach-sigmoid official-2026 u=11f pair."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess


N = Path("/srv/local/shengenli/flowstar_acceleration_20260921T153643Z")
OLD = N / "runs/archcomp_review_20260923/suite_build"
FROZEN = Path("/srv/local/shengenli/CROWN-Reach/submit/CROWN-Reach/archcomp/TORA")
P3 = N / "runs/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_p3_full500_001"
NEW = N / "runs/archcomp26_20261001/native_tora_reach_sigmoid_u11_build_001"
WORK = NEW / "archcomp/TORA"


def once(content, before, after):
    if content.count(before) != 1:
        raise RuntimeError(f"expected exactly one frozen source marker: {before}")
    return content.replace(before, after)


def main():
    import numpy as np
    import onnx
    import onnx2pytorch
    import torch
    from onnx import numpy_helper

    for path in (FROZEN / "tora_sigmoid.cpp", FROZEN / "crown_sigmoid.py",
                 P3 / "nn_tora_sigmoid.mat", P3 / "controller_plant_u.onnx",
                 P3 / "controller_plant_u.onnx.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    receipt = json.loads((P3 / "controller_plant_u.onnx.json").read_text())
    if (receipt["activations"] != ["sigmoid"] * 4 or receipt["scale"] != 11 or
            receipt["offset"] != 0):
        raise ValueError("frozen official controller export differs")
    model = onnx.load(str(P3 / "controller_plant_u.onnx"))
    ops = [node.op_type for node in model.graph.node]
    if ops != ["Gemm", "Sigmoid"] * 4 + ["Mul", "Add"]:
        raise ValueError(f"official controller graph differs: {ops}")
    constants = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if (not np.array_equal(constants["control_scale"], np.array([11.0])) or
            not np.array_equal(constants["control_offset"], np.array([0.0]))):
        raise ValueError("controller output affine differs")
    net = onnx2pytorch.ConvertModel(model, experimental=False).to(torch.float64).eval()
    with torch.no_grad():
        center_u = float(net(torch.tensor([[-0.76, -0.44, 0.525, -0.29]],
                                          dtype=torch.float64)).reshape(-1)[0])
    if abs(center_u - receipt["forward"]["center_onnx"]) > 1e-12 or torch.cuda.is_initialized():
        raise ValueError("CPU controller preflight differs or initialized CUDA")

    source = (FROZEN / "tora_sigmoid.cpp").read_text()
    for marker in ('"x2"', '"-x1 + 0.1 * sin(x3)"', '"x4"', '"u"',
                   'setting.setFixedStepsize(0.01, 6);',
                   'Interval init_x1(-0.77, -0.75)', 'init_x2(-0.45, -0.43)',
                   'init_x3(0.51, 0.54)', 'init_x4(-0.3, -0.28)',
                   '"-x1 - 0.1"', '"x1 - 0.2"', '"-x2 - 0.9"', '"x2 + 0.6"'):
        if marker not in source:
            raise ValueError(f"frozen native source differs: {marker}")
    source = once(source, '#include "../../flowstar/flowstar-toolbox/Continuous.h"',
                  '#include "../../flowstar/flowstar-toolbox/Continuous.h"\n#include "arch_ranges.h"\n'
                  '#ifndef TORA_PERIODS\n#define TORA_PERIODS 10\n#endif')
    source = once(source, 'http://127.0.0.1:5000', 'http://127.0.0.1:5111')
    source = once(source, 'int steps = 10;', 'int steps = TORA_PERIODS;')
    source = once(source, 'X0.push_back(init_u);',
                  'X0.push_back(init_u);\n    if (argc == 2) {\n'
                  '        vector<vector<Interval>> boxes(1, X0);\n'
                  '        arch_ranges::boxes(boxes, argv[1]);\n        return 0;\n    }')
    source = once(source,
                  'dynamics.reach(result, initial_set, 0.5, setting, safeSet, symbolic_remainder);',
                  'dynamics.reach(result, initial_set, 0.5, setting, safeSet, symbolic_remainder);\n'
                  '        arch_ranges::record(result, 0, 4);\n'
                  '        if (result.flowpipes.size() < static_cast<size_t>((iter + 1) * 50))\n'
                  '        {\n            cout << "Flow* short numerical history." << endl;\n'
                  '            final_result = 2;\n            break;\n        }')
    source = once(source, '    // Check reachability',
                  '    if (final_result != 0)\n    {\n'
                  '        cout << "UNKNOWN (incomplete numerical horizon)" << endl;\n'
                  '        return 2;\n    }\n\n    // Check reachability')

    server = (FROZEN / "crown_sigmoid.py").read_text()
    server = once(server, "('127.0.0.1', 5000)", "('127.0.0.1', 5111)")
    server = once(server,
                  'model_path = os.path.join(benchmark_dir, "nn_tora_sigmoid.onnx")',
                  'model_path = os.environ["TORA_REACH_MODEL"]\nassert os.path.isfile(model_path)')
    server = once(server, 'output_scale = 22', 'output_scale = 1')
    server = once(server, 'output_offset = 0.5', 'output_offset = 0')
    server = once(server, '    return coefficients',
                  '    if os.environ.get("ARCH_RPC_LOG"):\n'
                  '        import json\n'
                  '        with open(os.environ["ARCH_RPC_LOG"], "a") as log:\n'
                  '            log.write(json.dumps({"input_lower": input_lb.detach().cpu().reshape(-1).tolist(),\n'
                  '                                  "input_upper": input_ub.detach().cpu().reshape(-1).tolist(),\n'
                  '                                  "response": coefficients}, allow_nan=False) + "\\n")\n'
                  '    return coefficients')
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

    NEW.mkdir(parents=True, exist_ok=False)
    WORK.mkdir(parents=True)
    (NEW / "flowstar").symlink_to(OLD / "flowstar", target_is_directory=True)
    shutil.copy2(OLD / "archcomp/tora_homogeneous/arch_ranges.h", WORK / "arch_ranges.h")
    shutil.copy2(P3 / "nn_tora_sigmoid.mat", WORK / "nn_tora_sigmoid.mat")
    shutil.copy2(P3 / "controller_plant_u.onnx", WORK / "controller_plant_u.onnx")
    (WORK / "tora_sigmoid_u11.cpp").write_text(source)
    (WORK / "crown_paper.py").write_text(server)
    preflight = {"controller_source": str(P3 / "nn_tora_sigmoid.mat"),
                 "controller_ops": ops, "control_scale_in_graph": 11,
                 "control_offset_in_graph": 0, "rpc_output_scale": 1,
                 "rpc_output_offset": 0, "center_plant_u": center_u,
                 "cuda_initialized": False, "initial_box_source": str(FROZEN / "tora_sigmoid.cpp"),
                 "ode": ["x2", "-x1 + 0.1 * sin(x3)", "x4", "u"],
                 "rpc_port": 5111, "step_s": 0.01, "order": 6}
    (NEW / "PREFLIGHT.json").write_text(json.dumps(preflight, indent=2) + "\n")

    include = N / "runs/author_nncs_reproduction_v1/native_build/deps/extracted/usr/include"
    library = OLD / "flowstar/flowstar-toolbox/libflowstar.a"
    prefix = ["taskset", "-c", "44", "/usr/bin/g++-15", "-O3", "-std=c++11",
              "-fpermissive", "-Wno-template-body", "-fopenmp", "-I", str(include),
              "-I", "/usr/include/jsoncpp"]
    libraries = [str(library), "-lmpfr", "-lgmp", "-lgsl", "-lgslcblas", "-lm",
                 "-lglpk", "-lcolamd", "-lamd", "-lz", "-lltdl", "-ljsoncpp",
                 "-lcurl", "-ljsonrpccpp-common", "-ljsonrpccpp-client",
                 "-l:libboost_thread.so.1.90.0"]
    record = {"built_utc": datetime.now(timezone.utc).isoformat(),
              "frozen_cpp": str(FROZEN / "tora_sigmoid.cpp"),
              "frozen_server": str(FROZEN / "crown_sigmoid.py"),
              "build_root": str(NEW), "profile": "official2026-mat-u11f",
              "controller": str(WORK / "controller_plant_u.onnx"), "builds": []}
    for periods, name in ((1, "smoke1"), (10, "full10")):
        binary = WORK / f"tora_sigmoid_u11_{name}"
        command = prefix + [f"-DTORA_PERIODS={periods}", str(WORK / "tora_sigmoid_u11.cpp")]
        command += libraries + ["-o", str(binary)]
        with (NEW / f"{name}.build.log").open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record["builds"].append({"periods": periods, "command": command,
                                 "returncode": process.returncode,
                                 "binary_bytes": binary.stat().st_size if binary.exists() else None})
        (NEW / "BUILD.json").write_text(json.dumps(record, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(f"native TORA {name} compile failed")
    boxes_path = NEW / "initial_boxes.json"
    subprocess.run([str(WORK / "tora_sigmoid_u11_smoke1"), str(boxes_path)], check=True)
    boxes = json.loads(boxes_path.read_text())
    expected = [[-0.77, -0.75], [-0.45, -0.43], [0.51, 0.54],
                [-0.3, -0.28], [0, 0], [0, 0]]
    if (len(boxes) != 1 or len(boxes[0]) != 6 or
            any(not (actual[0] <= requested[0] <= requested[1] <= actual[1] and
                     abs(actual[0] - requested[0]) < 1e-12 and
                     abs(actual[1] - requested[1]) < 1e-12)
                for actual, requested in zip(boxes[0], expected))):
        raise ValueError("compiled native initial box differs from official full box")
    preflight["compiled_initial_box"] = boxes[0]
    (NEW / "PREFLIGHT.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps({"status": "built", "build_root": str(NEW)}))


if __name__ == "__main__":
    main()
