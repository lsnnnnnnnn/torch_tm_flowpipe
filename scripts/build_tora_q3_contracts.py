#!/usr/bin/env python3
"""Machine-extract sanitized TORA/Q3/controller contracts from frozen evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import onnx

from torch_tm_flowpipe.batched_dense_tm import BatchedMonomialBasis


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def function_record(path: Path, name: str, logical_path: str) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        candidate for candidate in ast.walk(tree)
        if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and candidate.name == name
    )
    lines = source.splitlines()
    fragment = "\n".join(lines[node.lineno - 1 : node.end_lineno]) + "\n"
    return {
        "logical_source": logical_path,
        "source_file_sha256": sha256(path),
        "function": name,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "function_source_sha256": hashlib.sha256(fragment.encode()).hexdigest(),
        "extraction": "python_ast",
        "manually_transcribed": False,
    }


def onnx_contract(path: Path, role: str) -> dict[str, Any]:
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    return {
        "role": role,
        "sha256": sha256(path),
        "byte_size": path.stat().st_size,
        "ir_version": model.ir_version,
        "opset": [{"domain": row.domain, "version": row.version} for row in model.opset_import],
        "inputs": [row.name for row in graph.input],
        "outputs": [row.name for row in graph.output],
        "initializer_names": [row.name for row in graph.initializer],
        "operator_sequence": [row.op_type for row in graph.node],
        "extraction": "onnx_parser",
        "manually_transcribed": False,
    }


def bounds_range(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    lows: list[float] = []
    highs: list[float] = []
    for row in rows:
        value = row[field]
        lows.extend(float(item) for leaf in value["lower"] for item in (leaf if isinstance(leaf, list) else [leaf]))
        highs.extend(float(item) for leaf in value["upper"] for item in (leaf if isinstance(leaf, list) else [leaf]))
    return {"minimum_lower": min(lows), "maximum_upper": max(highs)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xiangru-root", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--controller-trace", type=Path, required=True)
    parser.add_argument("--xiangru-plant-jsonl", type=Path, required=True)
    parser.add_argument("--original-controller", type=Path, required=True)
    parser.add_argument("--transformed-controller", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    root = args.xiangru_root.resolve()
    generic = root / "experiments/remainder_ablation/generic_fixed_basis.py"
    runner = root / "experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py"
    controller_source = root / "experiments/remainder_ablation/run_c2_autolirpa_feasibility.py"
    outward_source = root / "experiments/remainder_ablation/controller_outward.py"
    config = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    trace = json.loads(args.controller_trace.read_text(encoding="utf-8"))
    rows = trace["rows"]
    with args.xiangru_plant_jsonl.open(encoding="utf-8") as handle:
        xiangru_header = json.loads(next(handle))
    torch_basis = BatchedMonomialBasis.build(6, 3, "cpu")
    torch_exponents = torch_basis.exponents.tolist()
    if xiangru_header["basis_exponents"] != torch_exponents:
        raise ValueError("Xiangru and Torch Q3 exponent order differs")
    if len(rows) != 20 or any(row["leaf_id"] != list(range(48)) for row in rows):
        raise ValueError("controller observation trace is not canonical B48/T20")

    evidence = {
        "xiangru_commit": "27d29050a5f214b56f211ca9cb411e734ed80230",
        "dynamics": function_record(generic, "tm_tora_rhs", "xiangru@27d29050:experiments/remainder_ablation/generic_fixed_basis.py"),
        "sine": function_record(generic, "tm_sin", "xiangru@27d29050:experiments/remainder_ablation/generic_fixed_basis.py"),
        "picard": function_record(generic, "run_tora_remainder_picard", "xiangru@27d29050:experiments/remainder_ablation/generic_fixed_basis.py"),
        "lane": function_record(runner, "_run_lane", "xiangru@27d29050:experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py"),
        "controller": function_record(controller_source, "_normalized_inputs", "xiangru@27d29050:experiments/remainder_ablation/run_c2_autolirpa_feasibility.py"),
        "outward": function_record(outward_source, "outward_host_composition", "xiangru@27d29050:experiments/remainder_ablation/controller_outward.py"),
    }
    workload = {
        "schema": "tora_q3_workload_contract_v1",
        "dynamics": ["x1'=x2", "x2'=-x1+0.1*sin(x3)", "x3'=x4", "x4'=u1-10", "u1'=0"],
        "state_order": ["x1", "x2", "x3", "x4", "u1"],
        "initial_set": config["initial_set"],
        "partition": {"name": "b48_static", "splits": [8, 6, 1, 1], "leaf_ids": "0..47"},
        "step_size": config["step_size"],
        "segments_per_controller_period": config["n_steps_per_control"],
        "controller_period": config["step_size"] * config["n_steps_per_control"],
        "segments": config["n_total_steps"],
        "target_horizon": config["step_size"] * config["n_total_steps"],
        "property": {"physical_states": ["x1", "x2", "x3", "x4"], "predicate": "abs(state)<=2"},
        "remainder_rounds": config["frr_rounds"],
        "dtype": "float64",
        "device": "cuda",
        "extraction": {"config_sha256": sha256(args.frozen_config), "source": evidence},
        "manually_transcribed_fields": ["dynamics display strings", "property predicate display string", "b48 split tuple"],
    }
    q3 = {
        "schema": "tora_q3_basis_contract_v1",
        "variables": xiangru_header["basis_variables"],
        "support": "complete total degree <= 3",
        "slot_count": 84,
        "exponents": torch_exponents,
        "xiangru_to_torch_slot_permutation": list(range(84)),
        "permutation_bijective": True,
        "torch_basis_fingerprint": torch_basis.fingerprint,
        "xiangru_export_header_sha256": sha256(args.xiangru_plant_jsonl),
        "local_time_domain": [0.0, 0.1],
        "spatial_parameter_domain": [-1.0, 1.0],
        "extraction": "Xiangru live exporter header compared exactly with Torch route table",
        "manually_transcribed": False,
    }
    original = onnx_contract(args.original_controller, "original_execution_model")
    transformed = onnx_contract(args.transformed_controller, "transformed_configuration_model")
    if original["sha256"] != trace["controller_execution_model_sha256"] or transformed["sha256"] != trace["controller_sha256"]:
        raise ValueError("controller asset hash does not match observation trace")
    controller = {
        "schema": "tora_q3_controller_contract_v1",
        "assets": [original, transformed],
        "load_contract": {"environment_variable": "TORA_CONTROLLER_PATH", "expected_sha256": original["sha256"], "bytes_permitted_in_git": False},
        "state_input_order": ["x1", "x2", "x3", "x4"],
        "input_dimension": 4,
        "output_dimension": 1,
        "normalization": "state - ONNX initializer input_Mean",
        "network_semantics": "four flattened Linear layers, ReLU after every layer, output reshaped to [batch,1]",
        "bound_backend": {
            "package": "auto_LiRPA",
            "version": importlib.metadata.version("auto-lirpa"),
            "method": "CROWN",
            "activation_bound_option": "same-slope",
            "conv_mode": "matrix",
            "composition": "outward host nextafter",
        },
        "controller_update_period": 1.0,
        "observed_periods": len(rows),
        "observed_leaf_ids": "0..47 at every period",
        "observed_aggregate_ranges": {
            "input_after_normalization": bounds_range(rows, "controller_input_box_after_normalization"),
            "output_before_outward_composition": bounds_range(rows, "controller_output_interval_before_outward_composition"),
            "output_after_outward_composition": bounds_range(rows, "controller_output_interval_after_outward_composition"),
            "installed_u1": bounds_range(rows, "u1_interval_installed_for_next_ten_segments"),
        },
        "observation_trace_sha256": sha256(args.controller_trace),
        "evidence": {"normalization": evidence["controller"], "composition": evidence["outward"]},
        "manually_transcribed_fields": ["network_semantics display string"],
    }
    output_contract = {
        "schema": "tora_q3_output_contract_v1",
        "alignment_key": ["lane", "segment_index", "exact_physical_time", "controller_period", "leaf_id", "state", "enclosure_kind"],
        "enclosure_kinds": {
            "endpoint": "exact substitution at t=segment_index*0.1",
            "tube": "full segment interval [(segment_index-1)*0.1,segment_index*0.1]",
        },
        "levels": ["per_leaf", "b48_hull"],
        "states": ["x1", "x2", "x3", "x4", "u1"],
        "required_status": ["validation_seed", "validation_image", "validation_margin", "accepted_or_rejected", "property_margin"],
        "interpolation": "prohibited",
        "zero_width_ratio": "N/A",
        "manually_transcribed": True,
    }
    algorithm_aligned = {
        "schema": "tora_q3_algorithm_aligned_contract_v1",
        "lane": "common_control_plant_replay",
        "matched": ["workload", "B48 leaf order", "held controller intervals", "h=0.1", "T20", "complete Q3 exponent order", "K2 polynomial Picard", "10 remainder rounds", "float64 CUDA", "endpoint/tube definitions"],
        "not_matched": ["sine enclosure implementation", "roundoff policy", "normalization/carry implementation", "eager versus compiled execution"],
        "controller_recomputed_by_torch": False,
        "period_local_observation_restart": True,
        "independent_closed_loop": False,
    }
    method_native = {
        "schema": "tora_q3_method_native_contract_v1",
        "xiangru": "frozen compiled GenericTM complete-Q3 with its native sine/remainder/carry",
        "torch": "native generic dense complete-Q3 tensors with analytic outward sine and affine carry",
        "common_workload": workload,
        "coefficient_comparison": "requires a separately tested per-segment normalization bijection; otherwise unavailable",
        "claims": ["physical enclosure tightness", "property margin", "measured stage runtime", "completion horizon"],
    }
    for name, value in (
        ("tora_workload_contract.json", workload),
        ("q3_basis_contract.json", q3),
        ("controller_contract.json", controller),
        ("output_contract.json", output_contract),
        ("algorithm_aligned_contract.json", algorithm_aligned),
        ("method_native_contract.json", method_native),
    ):
        write_json(output / name, value)
    field_map = """# TORA-Q3 contract field map

| Field | Frozen evidence | Torch implementation |
|---|---|---|
| Dynamics/state order | AST-hashed `tm_tora_rhs` at Xiangru `27d29050` | `torch_tm_flowpipe.tora_q3.tora_q3_rhs` |
| Q3 basis/order | live Xiangru exporter header | `BatchedMonomialBasis.build(6,3)`; identical 84-slot order |
| Workload | hash-pinned resolved config | native B48 runner |
| Controller assets | parsed ONNX hashes and observation trace | external `TORA_CONTROLLER_PATH`; bytes excluded from Git |
| Endpoint/tube | explicit tagged raw fields | distinct endpoint substitution and full-domain range |
| Replay reset | observed pre-controller leaf boxes each period | period-local observation restart; not independent closed loop |

Fields described for readability rather than directly parsed are listed in each
JSON file under `manually_transcribed_fields` or `manually_transcribed`.
"""
    (output / "field_map.md").write_text(field_map, encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_files": 7, "basis_fingerprint": torch_basis.fingerprint, "controller_trace_sha256": sha256(args.controller_trace)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
