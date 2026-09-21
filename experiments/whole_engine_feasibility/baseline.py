"""Frozen inputs and thin wrappers around existing Gr/Flow* runners."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "benchmarks/review/fixed_profiles.json"
PARTITION = ROOT / "artifacts/runs/range_batch_device_20260909T030609Z/PARTITION_PLAN.json"
PYTHON = Path("/srv/local/shengenli/miniforge3/envs/py11/bin/python")
FLOWSTAR_ROOT = Path("/srv/local/shengenli/xiangru_adoption_20260907T032448Z/flowstar_native")
FLOWSTAR_SHA = "722a5611c0b33564b11916b13474d0d21d74bc0c"


def _read(path):
    return json.loads(Path(path).read_text())


def frozen_case(plant, batch):
    """B1 is the original box; B2 is lanes 0/31; B32 is the frozen 8x4 grid."""
    if plant not in ("van_der_pol", "brusselator") or batch not in (1, 2, 32):
        raise ValueError("only the two frozen plants and B1/B2/B32 are supported")
    profile = _read(PROFILE)
    matched_path = ROOT / profile["shared_matched_contracts_path"]
    matched_bytes = matched_path.read_bytes()
    if hashlib.sha256(matched_bytes).hexdigest() != profile["shared_matched_contracts_sha256"]:
        raise ValueError("frozen matched contract changed")
    matched = json.loads(matched_bytes)["plants"][plant]
    fixed = profile["plants"][plant]
    if batch == 1:
        lane_ids = [0]
        exact_boxes = [matched["initial_decimal_box"]]
        box_hex = [[axis["enclosing_binary64_hex"] for axis in matched["initial_representation"]]]
    else:
        plan = _read(PARTITION)
        lane_ids = [0, 31] if batch == 2 else plan["subsets"]["32"]
        tasks = plan["plants"][plant]["tasks"]
        exact_boxes = [tasks[lane]["exact"] for lane in lane_ids]
        box_hex = [tasks[lane]["outward_hex"] for lane in lane_ids]
    boxes = [[[float.fromhex(v) for v in axis] for axis in box] for box in box_hex]
    for exact_box, box in zip(exact_boxes, boxes):
        for exact, bounds in zip(exact_box, box):
            if not (Fraction(bounds[0]) <= Fraction(exact[0]) <= Fraction(exact[1]) <= Fraction(bounds[1])):
                raise ValueError("frozen binary64 box does not enclose its exact target")
    return dict(plant=plant, batch=batch, lane_ids=lane_ids, boxes=boxes,
                box_hex=box_hex, exact_boxes=exact_boxes,
                original=batch == 1, order=matched["order"],
                h=float.fromhex(fixed["fixed_h_hex"]), h_hex=fixed["fixed_h_hex"],
                sr_capacity=matched["sr_capacity"],
                cutoff=float.fromhex(matched["cutoff"]["nearest_binary64_hex"]),
                remainder_estimation=float.fromhex(matched["target_remainder_radius"]["nearest_binary64_hex"]),
                rhs=matched["rhs_expression_strings"],
                rhs_expression_strings=matched["rhs_expression_strings"],
                refinement_limit=fixed["config"]["refinement_replay_limit"],
                stop_ratio=float.fromhex(fixed["config"]["stop_ratio_hex"]),
                state_names=matched["state_names"],
                requested_steps=fixed["steps"],
                initial_semantics=matched["initial_semantics"],
                input_box_semantics="minimal outward binary64 enclosure of the exact target; normalization may widen the internal state range",
                original_cpu_normalized_state_range_hex=(fixed["original_outward_binary64_box_hex"] if batch == 1 else None),
                profile_path=str(PROFILE), matched_path=str(matched_path),
                partition_path=str(PARTITION),
                partition_sha256=hashlib.sha256(PARTITION.read_bytes()).hexdigest())


def gr_command(plant, batch, steps, output, *, core=2):
    """Return (argv, env), without executing; run with cwd=ROOT.

    The original py11 baseline environment is intentional and must be disclosed
    separately from the Huan environment. --warm excludes cold CUDA startup;
    the existing wall_s also excludes result serialization and physical export.
    """
    case = frozen_case(plant, batch)
    if type(steps) is not int or steps < 1:
        raise ValueError("positive step count required")
    command = ["taskset", "-c", str(core), str(PYTHON), "-m",
               "experiments.live_range_solver.runner", "--plant", plant,
               "--route", "Gr", "--batch", str(batch), "--steps", str(steps),
               "--ids", ",".join(map(str, case["lane_ids"])),
               "--max-wait-s", ".020", "--max-group", "32", "--warm",
               "--output", str(Path(output).resolve())]
    if case["original"]:
        command.append("--original")
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1")
    return command, environment


def flowstar_identity():
    """Require the already inspected accessor commit and unchanged tracked code."""
    head = subprocess.check_output(["git", "-C", str(FLOWSTAR_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if head != FLOWSTAR_SHA:
        raise ValueError(f"unexpected Flow* source commit: {head}")
    for args in (("diff", "--exit-code"), ("diff", "--cached", "--exit-code")):
        subprocess.run(["git", "-C", str(FLOWSTAR_ROOT), *args], check=True, capture_output=True)
    archive = FLOWSTAR_ROOT / "flowstar-toolbox/libflowstar.a"
    source_record = ROOT / ("artifacts/runs/resident_tm_block_20260914T032650Z/"
                            "raw_minimal/matched_flowstar/flowstar_identity.json")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != _read(source_record)["flowstar_static_archive_sha256"]:
        raise ValueError("Flow* library differs from the recorded build")
    return dict(root=str(FLOWSTAR_ROOT), source_sha=head, library_sha256=digest,
                stock_base_sha="b85a3211748cb77b736fe4ad42ee02d8d2b81148",
                tracked_source_clean=True, untracked_build_products_preserved=True)


def run_flowstar(plant, output, *, steps=20, core=2, binary=None):
    """Run only one B32 comparison, reusing the old build/run functions.

    No campaign, old five-pair gate, library rebuild, or source mutation occurs.
    A caller may reuse the first receipt's binary in another fresh output dir.
    """
    from experiments.resident_tm_block.run_flowstar_matched import build, expected_boxes, run_plant
    frozen_case(plant, 32)
    if type(steps) is not int or steps < 1:
        raise ValueError("positive step count required")
    identity = flowstar_identity()
    output = Path(output).resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("baseline outputs must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw_minimal"
    raw.mkdir()
    build_command, build_s = None, 0.0
    if binary is None:
        binary = raw / "flowstar_matched_b32"
        driver = ROOT / "experiments/resident_tm_block/flowstar_matched_b32.cpp"
        build_command, build_s = build(Path("/usr/bin/g++-15"), FLOWSTAR_ROOT,
                                      driver, binary, raw / "build.log")
    else:
        binary = Path(binary).resolve()
        if not binary.is_file():
            raise FileNotFoundError(binary)
    summary, receipt = run_plant(binary, output, plant, steps, core,
                                 expected_boxes(_read(PARTITION), plant))
    result = dict(plant=plant, batch=32, steps=steps, summary=summary, receipt=receipt,
                  identity=identity, binary=str(binary), build_command=build_command,
                  build_seconds_outside_timing=build_s)
    (output / "baseline.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _self_check():
    for plant in ("van_der_pol", "brusselator"):
        for batch in (1, 2, 32):
            case = frozen_case(plant, batch)
            assert len(case["boxes"]) == batch and len(set(case["lane_ids"])) == batch
            assert case["refinement_limit"] == 491 and case["stop_ratio"] == 0.99
            command, environment = gr_command(plant, batch, 20, ROOT.parent / "not_run")
            assert ("--original" in command) is (batch == 1)
            assert "--diagnostic" not in command
            assert environment.get("CUDA_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES")
    assert frozen_case("van_der_pol", 2)["lane_ids"] == [0, 31]
    assert frozen_case("van_der_pol", 1)["exact_boxes"][0][0] == ["1.1", "1.4"]
    print("frozen inputs and baseline command self-check passed; no solver executed")


if __name__ == "__main__":
    _self_check()
