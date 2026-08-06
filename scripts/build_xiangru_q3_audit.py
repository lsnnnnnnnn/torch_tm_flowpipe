#!/usr/bin/env python3
"""Build deterministic Q3/order/contract/gate evidence from frozen inputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from torch_tm_flowpipe.protocol.q3_audit import (
    REQUIRED_CONTRACT_FIELDS,
    complete_total_degree_exponents,
    contract_field,
    deterministic_json_bytes,
    formal_match_decision,
    parse_xiangru_runtime,
    total_degree_retained,
    validate_contract,
)


OUT = ROOT / "outputs/xiangru_q3_matched_audit_20260806"
REP = OUT / "xiangru_reproduction"
FRESH_DIR = REP / "fresh_q3_b48_t20"
FRESH_JSON = FRESH_DIR / "raw_outputs/s3r_q3_b48_rep1.json"
AUTHOR_JSON = Path(
    "/srv/local/shengenli/CROWN-Reach_Development/experiments/reachability/"
    "results/s3c_fair_timing_raw/rep1_q3/s3r_q3_b48_rep1.json"
)
XIANGRU = Path("/srv/local/shengenli/CROWN-Reach_Development_native_27d2905")
FLOWSTAR = Path("/srv/local/shengenli/flowstar")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(deterministic_json_bytes(value))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def field(
    value: Any,
    matched: bool | str,
    source_file: str,
    source_line: str | int,
    evidence: str,
    reason: str,
) -> dict[str, Any]:
    return contract_field(value, matched, source_file, source_line, evidence, reason)


def reproduction_artifacts() -> None:
    fresh = json.loads(FRESH_JSON.read_text(encoding="utf-8"))
    author = json.loads(AUTHOR_JSON.read_text(encoding="utf-8"))
    fresh_cell = fresh["cells"]["b48_static"]["complete_q3"]
    author_cell = author["cells"]["b48_static"]["complete_q3"]
    columns = [
        "run", "source", "status", "cell_status", "segments", "horizon",
        "first_failure", "retry_segments", "leaves", "solver_seconds_excluding_validation",
        "total_seconds_including_validation", "compile_warm_seconds_excluded",
        "maximum_endpoint_width_at_t20", "maximum_tube_width_at_t20",
    ]
    rows = []
    for name, source, payload, cell in (
        ("author_reference", str(AUTHOR_JSON), author, author_cell),
        ("fresh_exact_commit", str(FRESH_JSON.relative_to(ROOT)), fresh, fresh_cell),
    ):
        timing = cell["timing"]
        last = cell["segments"][-1]
        rows.append(
            {
                "run": name,
                "source": source,
                "status": payload["status"],
                "cell_status": cell["status"],
                "segments": cell["segments_attempted"],
                "horizon": cell["certified_horizon"],
                "first_failure": json.dumps(cell["first_failure"], sort_keys=True),
                "retry_segments": json.dumps(cell["retry_segments"], sort_keys=True),
                "leaves": last["leaves"],
                "solver_seconds_excluding_validation": timing["solver_wall_seconds_excluding_validation"],
                "total_seconds_including_validation": timing["total_wall_seconds_including_validation"],
                "compile_warm_seconds_excluded": timing["implementation_compile_and_warm_seconds_excluded"],
                "maximum_endpoint_width_at_t20": last["maximum_endpoint_width"],
                "maximum_tube_width_at_t20": last["maximum_composed_tube_width"],
            }
        )
    path = REP / "reproduction_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    command = FRESH_DIR / "original_command.txt"
    write_text(
        REP / "original_commands.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\n\n# Frozen exact command captured from the fresh run.\n"
        + command.read_text(encoding="utf-8").strip(),
    )
    snapshot = REP / "original_config_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for source in ("diffreach_config.json", "controller_transformed.onnx"):
        shutil.copyfile(FRESH_DIR / "original_config_snapshot" / source, snapshot / source)

    artifact_rows = []
    for key, value in sorted(fresh["artifacts"].items()):
        if key.endswith("_sha256"):
            continue
        artifact_rows.append(
            {
                "name": key,
                "declared_path": value,
                "declared_sha256": fresh["artifacts"].get(f"{key}_sha256"),
                "role": "frozen source/input named by the upstream result",
            }
        )
    artifact_rows.extend(
        {
            "name": path.name,
            "declared_path": str(path.relative_to(ROOT)),
            "declared_sha256": sha256(path),
            "role": "fresh raw reproduction artifact",
        }
        for path in sorted((FRESH_DIR / "raw_outputs").iterdir())
    )
    write_json(
        REP / "original_artifact_inventory.json",
        {
            "schema": "xiangru_q3_original_artifact_inventory_v1",
            "author_reference": str(AUTHOR_JSON),
            "author_reference_sha256": sha256(AUTHOR_JSON),
            "fresh_raw_sha256": sha256(FRESH_JSON),
            "artifacts": artifact_rows,
        },
    )

    write_text(
        REP / "q3_entrypoint_map.md",
        """# Xiangru complete-Q3 entrypoint map

| Question | Authoritative path | Evidence |
|---|---|---|
| Where is Q3 selected? | `experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py:341-342` | Method `complete_q3` calls `complete_total_degree_support(3)`. |
| What receives it? | `run_s0...py:345-348,456-462` | The support builds static routes, then enters `step_closed_loop`. |
| What mathematical operation changes? | `generic_fixed_basis.py:94-108`; `tensor_fixed_basis.py:149-162,605-648` | Six-variable complete total-degree support; integration/products outside support are intervalized. |
| What completed baseline is used here? | `s3c_fair_timing_raw/rep1_q3/s3r_q3_b48_rep1.json` | Homogeneous TORA, B48 static partition, complete-Q3, 200 segments. |
| What does “completed” mean? | raw cell fields plus `run_s0...py:688-715` | Cell `VERIFIED`, no first failure/retries, 200/200 segments, certified horizon 20.0; top-level `PARTIAL_VALIDATION` only records selected matrix scope. |

The authoritative frozen command is in `original_commands.sh`; its config and controller bytes are in `original_config_snapshot/`.
""",
    )
    write_text(
        REP / "q3_source_callgraph.md",
        """# Xiangru complete-Q3 source callgraph

```text
run_s0_tora_static_partition_sweep.main
  -> _run_lane(method="complete_q3")                         [run_s0:327-345]
     -> complete_total_degree_support(3, variables=6)        [generic_fixed_basis:94-108]
     -> StaticRouteTables.build(support)                     [run_s0:345]
     -> step_closed_loop                                     [run_s0:456-462]
        -> TensorFixedBasisKernel.run_tora_remainder_picard   [tensor_fixed_basis:838-940]
           -> polynomial_rhs / integrate, exactly twice      [tensor_fixed_basis:745-772,847-858]
           -> seed interval remainder                        [tensor_fixed_basis:860-873]
           -> tm_rhs / integrate and 10 DR-RP rounds         [tensor_fixed_basis:874-927]
           -> evaluate full tube and endpoint separately     [tensor_fixed_basis:928-940]
        -> property margins 2 - abs(bound)                    [tensor_closed_loop:521-522]
     -> boundary_for_controller / worker.control every 10 h  [run_s0:407-451]
```

Variable slot 0 is local time. Slots 1-5 parameterize `x1,x2,x3,x4,u1`; the complete degree-3 support therefore contains `C(6+3,3)=84` monomials.
""",
    )


def order_artifacts() -> dict[str, Any]:
    labels = {
        "1": (0, 0), "x": (0, 1), "t": (1, 0), "x^2": (0, 2),
        "xt": (1, 1), "t^2": (2, 0), "x^3": (0, 3), "x^2t": (1, 2),
        "xt^2": (2, 1), "t^3": (3, 0), "x^4": (0, 4),
    }
    rows = []
    for label, exponent in labels.items():
        xiangru_retained = exponent in complete_total_degree_exponents(3, 2)
        torch_retained = total_degree_retained(exponent, 3)
        rows.append(
            {
                "monomial": label,
                "exponent_local_time_x": list(exponent),
                "xiangru_complete_q3_retained": xiangru_retained,
                "torch_order3_retained": torch_retained,
                "predicate_equal": xiangru_retained == torch_retained,
            }
        )
    result = {
        "schema": "q3_order_semantics_tests_v1",
        "degree": 3,
        "xiangru_variable_count": 6,
        "xiangru_support_size_observed": len(complete_total_degree_exponents(3, 6)),
        "xiangru_support_size_expected": math.comb(9, 3),
        "retention_predicate_equivalent": all(row["predicate_equal"] for row in rows),
        "full_algorithmic_order_semantics_equivalent": False,
        "algorithmic_blockers": [
            "Xiangru complete-Q3 uses a dense fixed six-variable support; the current Torch VDP lane uses a sparse three-variable representation.",
            "Xiangru performs exactly two polynomial Picard iterations plus ten DR-RP remainder rounds; Torch defaults polynomial Picard iterations to order (three at order 3).",
            "The available models are different (closed-loop TORA versus plant-only Van der Pol).",
        ],
        "monomials": rows,
    }
    write_json(OUT / "contract/order_semantics_tests.json", result)
    write_text(
        OUT / "contract/order_semantics.md",
        """# Q3/order semantics

`complete_q3` is the complete total-degree-3 support over six variables, with local time in slot 0. It has 84 slots. Products or time integrations whose exponent is absent are evaluated into interval overflow. The checked monomial retention predicate is exactly `sum(exponent) <= 3`, so the small predicate test agrees with Torch `Polynomial.truncate(3)`.

That predicate agreement is necessary but not sufficient for algorithmic equivalence. Xiangru uses a dense fixed basis, exactly two polynomial Picard iterates, a seed remainder, and ten DR-RP remainder rounds. Torch uses sparse polynomial maps in its current VDP lane and defaults polynomial Picard iterations to `order`, hence three iterations at order 3. The plant/controller contracts also differ. Therefore this audit records `retention_predicate_equivalent=true` but `full_algorithmic_order_semantics_equivalent=false`.

Sources: Xiangru `generic_fixed_basis.py:94-108`, `tensor_fixed_basis.py:149-162,605-648,838-928`; Torch `polynomial.py:235-250`, `flowpipe.py:1783-1805`; Flow* `Continuous.cpp:2781,2843,2920-2940,2962`.
""",
    )
    path = OUT / "contract/order_field_map.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        columns = ["tool", "field", "value", "source_file", "source_line", "comparison"]
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [
                {"tool": "Xiangru", "field": "support", "value": "complete total degree <=3, six variables, 84 slots", "source_file": "experiments/remainder_ablation/generic_fixed_basis.py", "source_line": "94-108", "comparison": "retention predicate agrees; basis arity/algorithm differ"},
                {"tool": "Xiangru", "field": "Picard", "value": "K2 polynomial + 10 DR-RP", "source_file": "experiments/remainder_ablation/tensor_fixed_basis.py", "source_line": "838-928", "comparison": "not Torch default order-3 K3"},
                {"tool": "Torch", "field": "support", "value": "sparse total degree <= order", "source_file": "src/torch_tm_flowpipe/polynomial.py", "source_line": "235-250", "comparison": "predicate agrees"},
                {"tool": "Torch", "field": "Picard", "value": "iterations defaults to order", "source_file": "src/torch_tm_flowpipe/flowpipe.py", "source_line": "1783-1805", "comparison": "order-3 defaults K3"},
                {"tool": "Flowstar", "field": "order", "value": "insertion/truncation order and order iterations", "source_file": "flowstar-toolbox/Continuous.cpp", "source_line": "2781,2843,2920-2940", "comparison": "independent VDP reference; no Q3 TORA match"},
            ]
        )
    return result


def xiangru_contract() -> dict[str, Any]:
    src = "experiments/remainder_ablation/"
    fields = {
        "dynamics": field(["x1'=x2", "x2'=-x1+0.1*sin(x3)", "x3'=x4", "x4'=u1-10", "u1'=0 between controller updates"], True, src + "tensor_fixed_basis.py", "745-836", "Polynomial and TM RHS concatenate these five derivatives.", "Reference baseline field."),
        "state_order": field(["x1", "x2", "x3", "x4", "u1"], True, "fresh_q3_b48_t20/config_resolved.json", "variable_names", "state_dim=4 and action_dim=1; u1 is held as a fifth state.", "Reference baseline field."),
        "coordinate_transform": field("48 uniform Cartesian leaves [8,6,1,1]; normalized local variables with local time first", True, "fresh raw JSON", "controls.grid_splits", "B48 partition metadata passes ordering, hull, adjacency and containment checks.", "Reference baseline field."),
        "plant_parameters": field({"sine_gain": 0.1, "control_offset": -10.0}, True, src + "tensor_fixed_basis.py", "754-760", "Constants appear in x2' and x4'.", "Reference baseline field."),
        "controller": field({"kind": "homogeneous ReLU TORA NN, exact native flat MLP reconstruction of ONNX Conv/ReLU", "bounder": "auto_LiRPA same-slope outward", "update_period": 1.0, "source_sha256": "52a50c6bc6b1b45b89319edc809cd4d3baca06c32f9fd2ddceb2e95007414418", "transformed_sha256": "bb80479ce51b6f2558ac4a47cae2831ff3f49275ffaf7b1b874adf3c3b14703e"}, True, src + "run_s0_tora_static_partition_sweep.py", "407-451,1080-1090", "Controller boundary is reprojected and bounded every ten 0.1 s segments.", "Reference baseline field."),
        "initial_set": field([[0.6, 0.7], [-0.7, -0.6], [-0.4, -0.3], [0.5, 0.6], [0.0, 0.0]], True, "fresh_q3_b48_t20/config_resolved.json", "initial_set", "Frozen configuration SHA256 13b28a...", "Reference baseline field."),
        "property_set": field("full tube and endpoint satisfy abs(x_i)<=2 for i=1..4", True, src + "tensor_closed_loop.py", "521-522", "Margins are 2.0 - abs(tube/endpoint); certificate requires nonnegative tube margins.", "Reference baseline field."),
        "target_horizon": field(20.0, True, "fresh raw JSON", "cells.b48_static.complete_q3.certified_horizon", "200 fixed segments at h=0.1 reach 20.0.", "Reference baseline field."),
        "order_semantics": field({"q": 3, "support": "complete total degree <=3", "variables": 6, "slots": 84, "polynomial_picard_rounds": 2, "remainder_rounds": 10}, True, src + "generic_fixed_basis.py", "94-108", "Direct monomial test and support count pass.", "Reference baseline field."),
        "local_time_interval": field([0.0, 0.1], True, src + "tensor_fixed_basis.py", "149-162", "Integration increments exponent slot 0; configured step is 0.1.", "Reference baseline field."),
        "step_policy": field({"kind": "fixed", "h": 0.1, "h_min": "not_applicable", "h_max": "not_applicable", "controller_period_segments": 10}, True, "fresh raw JSON", "controls.step_size/controller_period", "No runtime step branching; 200 segments.", "Reference baseline field."),
        "remainder_policy": field({"initial_seed": 0.01, "failure_retry_seed": 0.1, "rounds": 10, "stop_ratio": 0.95, "range": "natural", "symbolic_capacity": 200}, True, src + "tensor_fixed_basis.py", "860-927", "Seeded interval remainder and componentwise shrink rounds.", "Reference baseline field."),
        "cutoff_truncation": field("fixed support; every absent product/integration route is conservatively intervalized", True, src + "tensor_fixed_basis.py", "149-162,605-648", "Overflow route values are accumulated as intervals.", "Reference baseline field."),
        "picard_validation": field("K2 polynomial Picard, then seeded DR-RP componentwise self-map/shrink for 10 rounds", True, src + "tensor_fixed_basis.py", "838-928", "Code constructs first and second iterates, then performs subseteq_elem updates.", "Reference baseline field."),
        "transition_lifecycle": field("closed-loop boundary projection; controller imposed every 10 segments; fixed-basis current state carried", True, src + "run_s0_tora_static_partition_sweep.py", "407-462", "Control is refreshed before the next segment and step_closed_loop consumes pre_state.", "Reference baseline field."),
        "interval_backend": field({"backend": "PyTorch tensor interval arithmetic with outward controller composition", "dtype": "float64"}, True, "fresh raw JSON", "controls", "CPU/CUDA replay maximum error <=1.776e-15 and validity passes.", "Reference baseline field."),
        "output_semantics": field({"endpoint": "local time fixed at h", "tube": "local time over [0,h]", "property_uses": "full tube"}, True, src + "tensor_fixed_basis.py", "928-940", "Tube and endpoint are evaluated separately and retained as separate fields.", "Reference baseline field."),
        "included_stages": field(["initial NN bound", "controller updates", "plant propagation", "validation", "scheduler", "compile/warm separately excluded from solver time"], True, src + "run_s0_tora_static_partition_sweep.py", "698-715", "Measured timing dictionary exposes controller, dynamics, validation, compile/warm and totals.", "Reference baseline field."),
        "device_threads": field({"device": "CUDA GPU 0 (V100 16GB)", "dtype": "float64", "threads": "captured environment; not forced"}, True, "fresh_q3_b48_t20/environment.json", "cuda_visible_devices/thread_environment", "Fresh run used CUDA_VISIBLE_DEVICES=0.", "Reference baseline field."),
        "success_predicate": field("all 48 leaves finite, DR-RP certified, full-tube property holds, 200 segments completed to t=20", True, src + "run_s0_tora_static_partition_sweep.py", "470-478,688-715", "Fresh cell is VERIFIED with first_failure=null and no retries.", "Reference baseline field."),
    }
    return {"schema": "q3_benchmark_contract_v1", "tool": "xiangru", "role": "reference baseline", "fields": fields}


def torch_contract() -> dict[str, Any]:
    cfg = "experiments/three_way_comparison_repair/benchmark_spec.yaml"
    mismatch = "Available Torch candidate is plant-only Van der Pol, not Xiangru closed-loop TORA."
    fields = {
        "dynamics": field(["x'=y", "y'=(1-x^2)y-x"], False, "src/torch_tm_flowpipe/ode_examples.py", "21-23", "The current candidate uses the two-state polynomial Van der Pol RHS.", mismatch),
        "state_order": field(["x", "y"], False, cfg, "systems.van_der_pol.state_names", "Two states rather than TORA x1..x4 plus held u1.", mismatch),
        "coordinate_transform": field("normalized insertion with local time plus two uncertainty variables", False, "src/torch_tm_flowpipe/flowpipe.py", "normalized insertion path", "Current lifecycle is a three-variable sparse VDP map.", "Different state dimension and physical coordinates."),
        "plant_parameters": field({"mu": 1.0}, False, "src/torch_tm_flowpipe/ode_examples.py", "21-23", "Van der Pol parameter mu defaults to one.", "No mapping to TORA sine gain/control offset."),
        "controller": field("none (plant-only)", False, "experiments/run_vdp_dense_backend.py", "48-78", "Loaded contract contains no NN controller/checkpoint.", "Xiangru baseline contains the exact TORA NN and periodic control update."),
        "initial_set": field([[1.1, 1.4], [2.35, 2.45]], False, cfg, "systems.van_der_pol.initial_box", "Authoritative VDP box.", "Different dimension and values from TORA B48 initial set."),
        "property_set": field("VDP y<=2.75 in stock reference; Torch candidate primarily propagates enclosures", False, "/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp", "56-58", "Independent VDP property is not abs(TORA states)<=2.", "Properties and models differ."),
        "target_horizon": field(10.0, False, cfg, "flowstar.original_vanderpol.horizon", "VDP target is 10 seconds.", "Xiangru Q3 target is 20 seconds."),
        "order_semantics": field({"candidate_order": 4, "order3_predicate": "sparse total degree <=3", "order3_default_polynomial_picard_rounds": 3}, False, "src/torch_tm_flowpipe/polynomial.py", "235-250", "Retention predicate matches at order 3, but current candidate is order 4 and algorithm/basis differ.", "Same digit alone is insufficient; full algorithmic order semantics do not match."),
        "local_time_interval": field("[0,h] per adaptive accepted segment", False, "src/torch_tm_flowpipe/flowpipe.py", "1783-1805", "Local time is integrated and truncated per step.", "Step sizes and model differ from fixed h=0.1 TORA."),
        "step_policy": field({"kind": "adaptive", "initial_h": 0.1, "h_min": 0.002, "h_max": 0.1, "growth": 1.1, "reject": "halve"}, False, cfg, "flowstar.original_vanderpol.step_policy", "Torch VDP compatibility lane follows adaptive policy.", "Xiangru baseline is fixed h=0.1 without runtime branching."),
        "remainder_policy": field({"initial_radius": 0.0001, "validation": "raw-remainder compatibility"}, False, cfg, "flowstar.original_vanderpol.candidate_remainder", "VDP candidate remainder differs.", "Not Xiangru seed 0.01/10-round DR-RP."),
        "cutoff_truncation": field({"cutoff": 1e-10, "truncate": "total degree <= order with dropped range added to interval"}, False, "src/torch_tm_flowpipe/polynomial.py", "235-256", "Sparse truncation/cutoff operation.", "Different cutoff and fixed-support overflow lifecycle."),
        "picard_validation": field("polynomial iterations default to order; interval self-map validation and adaptive rejection", False, "src/torch_tm_flowpipe/flowpipe.py", "1783-1805", "Order-3 would use K3 by default.", "Xiangru Q3 uses K2 plus ten componentwise DR-RP rounds."),
        "transition_lifecycle": field("right map -> insertion input/output -> normalized reset input/output", False, "src/torch_tm_flowpipe/flowpipe.py", "FlowstarTransitionLifecycle", "Actual transition objects are now captured by identity.", "Lifecycle is repaired but belongs to a different VDP algorithm/model."),
        "interval_backend": field({"backend": "torch.float64 interval arithmetic", "device": "CPU for VDP candidate"}, False, "experiments/run_vdp_dense_backend.py", "58-71", "Candidate contract fixes float64 CPU.", "Xiangru reproduction is native CUDA closed-loop."),
        "output_semantics": field({"endpoint": "tmv_right/raw endpoint", "tube": "accepted segment TM over [0,h]"}, False, "src/torch_tm_flowpipe/audit_trace.py", "transition trace schema v2", "Endpoint and tube remain distinct.", "No common dynamics/time interval on which to compare them."),
        "included_stages": field(["plant propagation", "validation/scheduler", "serialization optional"], False, "experiments/run_vdp_dense_backend.py", "234-270", "No NN bound or controller update.", "Plant-only runtime cannot be ranked against Xiangru closed-loop total."),
        "device_threads": field({"device": "CPU", "dtype": "float64", "threads": "captured, not matched to GPU"}, False, "experiments/run_vdp_dense_backend.py", "58-71", "Candidate device is CPU.", "Device and end-to-end stages differ."),
        "success_predicate": field("validated adaptive VDP flowpipe reaches requested horizon; failed horizon retained", False, "experiments/run_vdp_dense_backend.py", "541-558", "Completion requires current_time >= requested_horizon.", "Not the 48-leaf TORA full-tube certificate."),
    }
    return {"schema": "q3_benchmark_contract_v1", "tool": "torch_tm_flowpipe", "role": "available candidate", "fields": fields}


def flowstar_contract() -> dict[str, Any]:
    source = "/srv/local/shengenli/flowstar/benchmarks/continuous/vanderpol/vanderpol.cpp"
    mismatch = "Stock Flowstar candidate is an independent Van der Pol reference, not the Xiangru TORA benchmark."
    values = {
        "dynamics": (["x'=y", "y'=(1-x^2)y-x", "t'=1"], "25-26"),
        "state_order": (["x", "y", "t"], "18-26"),
        "coordinate_transform": ("Flowstar preconditioned Taylor-model flowpipe", "42-53"),
        "plant_parameters": ({"mu": 1.0}, "25-26"),
        "controller": ("none", "23-26"),
        "initial_set": ([[1.1, 1.4], [2.35, 2.45]], "42-53"),
        "property_set": ("y<=2.75", "56-58"),
        "target_horizon": (10.0, "74-89"),
        "order_semantics": ({"order": 4, "polynomial_picard_loops": 4, "total-order ctrunc": True}, "33-35"),
        "local_time_interval": ("[0,h] per accepted flowpipe", "23-26"),
        "step_policy": ({"kind": "adaptive", "h_min": 0.002, "h_max": 0.1}, "32-35"),
        "remainder_policy": ({"candidate": 0.0001, "symbolic_queue": 100}, "67-74"),
        "cutoff_truncation": ({"cutoff": 1e-10, "order": 4}, "32-35"),
        "picard_validation": ("order polynomial iterations, ctrunc Picard, subset validation", "32-35"),
        "transition_lifecycle": ("insert_ctrunc_normal and symbolic J/Phi_L carry", "67-89"),
        "interval_backend": ({"backend": "Flowstar stock interval", "precision": "double default"}, "9-11"),
        "output_semantics": ({"tube": "flowpipe", "plot": "transformed Taylor models"}, "114-129"),
        "included_stages": (["plant propagation", "validation", "symbolic remainder", "CPU clock timing"], "64-93"),
        "device_threads": ({"device": "CPU", "dtype": "double", "threads": "stock default"}, "9-11"),
        "success_predicate": ("result.isCompleted plus safety classification", "96-111"),
    }
    fields = {
        name: field(value, False, source, line, f"Stock source directly declares {name}.", mismatch)
        for name, (value, line) in values.items()
    }
    return {"schema": "q3_benchmark_contract_v1", "tool": "flowstar", "role": "independent reference", "fields": fields}


def contract_artifacts(order: dict[str, Any]) -> dict[str, Any]:
    contract_dir = OUT / "contract"
    xiangru = xiangru_contract()
    torch = torch_contract()
    flowstar = flowstar_contract()
    for name, contract in (("xiangru_native_contract", xiangru), ("torch_candidate_contract", torch), ("flowstar_candidate_contract", flowstar)):
        errors = validate_contract(contract)
        if errors:
            raise ValueError(f"{name}: {errors}")
        write_json(contract_dir / f"{name}.json", contract)
    decision = formal_match_decision([xiangru, torch])
    matched = {
        "schema": "q3_matched_contract_decision_v1",
        "reference": "xiangru",
        "candidate": "torch_tm_flowpipe",
        "case": "B",
        "retention_predicate_equivalent": order["retention_predicate_equivalent"],
        "full_algorithmic_order_semantics_equivalent": order["full_algorithmic_order_semantics_equivalent"],
        **decision,
        "first_failed_gate": "Gate 1: initial set, dimension, state order, coordinates, and dynamics",
        "unique_decisive_blocker": "No existing Torch lane runs the exact homogeneous-TORA plant, 5-state held-control representation, frozen NN controller, B48 initial partition, and 20 s closed-loop contract.",
        "flowstar_role": "independent stock Van der Pol reference; excluded from formal three-way ranking",
    }
    write_json(contract_dir / "matched_contract.json", matched)
    write_text(
        contract_dir / "matched_contract.md",
        """# Matched contract decision

**FORMAL MATCHED COMPARISON NOT AUTHORIZED.** The first gate fails before numerical comparison: Xiangru is a five-variable held-control homogeneous-TORA NNCS at T=20, while the existing Torch and stock Flowstar candidates are two-state plant-only Van der Pol at T=10. No adapter or approximation was added.

The small total-degree retention predicate agrees at degree 3, but the dense basis, Picard iteration count, remainder validation, controller, state set, horizon, device, and output workload do not form a common contract. See `matched_contract.json` for fail-closed blockers.
""",
    )
    write_text(
        contract_dir / "field_map.md",
        """# Contract field map

| Field | Xiangru native Q3 | Torch candidate | Flowstar reference | Matched? |
|---|---|---|---|---|
| Model | closed-loop homogeneous TORA | plant-only Van der Pol | plant-only Van der Pol | no |
| State | x1,x2,x3,x4,u1 | x,y | x,y,t | no |
| Controller | frozen ReLU ONNX, auto_LiRPA, period 1 s | none | none | no |
| Initial set | 4D box + held u, B48 | [1.1,1.4]x[2.35,2.45] | same VDP box | no |
| Horizon | 20 | 10 | 10 | no |
| Q/order | dense total-degree Q3, K2+10 DR-RP | current VDP order 4; order-3 sparse K3 | order 4 | predicate only |
| Step | fixed 0.1 | adaptive 0.002-0.1 | adaptive 0.002-0.1 | no |
| Device/workload | CUDA closed-loop | CPU plant-only | CPU plant-only | no |

Full evidence and per-field reasons are in the three machine-readable candidate contracts.
""",
    )
    return matched


def gate_and_runtime_artifacts(matched: dict[str, Any]) -> None:
    selection = {
        "schema": "q3_benchmark_selection_v1",
        "selected_native_reproduction": "Xiangru homogeneous-TORA B48 complete-Q3 T20",
        "formal_cross_tool_selection": None,
        "candidates": [
            {"candidate": "Xiangru homogeneous-TORA B48 Q3", "decision": "selected_native_only", "reason": "Only completed existing Q3 baseline with frozen raw result."},
            {"candidate": "Torch Van der Pol", "decision": "rejected_formal_match", "reason": "Different plant, state, controller, initial set, horizon, order lane, step policy and workload."},
            {"candidate": "stock Flowstar Van der Pol", "decision": "independent_reference", "reason": "Matches the Torch VDP family but not Xiangru TORA; no three-way Q3 contract."},
        ],
        "adapter_added": False,
    }
    write_json(OUT / "benchmark_selection/benchmark_selection.json", selection)
    write_text(
        OUT / "benchmark_selection/benchmark_selection.md",
        """# Benchmark selection

The existing Xiangru homogeneous-TORA B48 complete-Q3 T20 run is selected for native reproduction. No formal Torch–Xiangru benchmark is selected: Torch has no existing exact TORA+controller lane, and the goal prohibits adding a sine/controller adapter that changes the contract. Stock Flowstar Van der Pol remains an independent reference only.
""",
    )
    gates = {
        "schema": "q3_matched_gate_status_v1",
        "formal_comparison_authorized": False,
        "case": "B",
        "gates": [
            {"gate": 1, "name": "initial set and coordinates", "status": "FAIL", "reason": "TORA 4D+B48+held control versus 2D VDP box; dynamics differ."},
            {"gate": 2, "name": "order semantics", "status": "FAIL", "reason": "Retention predicate agrees, but dense/sparse basis, K2/K3 and remainder algorithms do not."},
            {"gate": 3, "name": "one step", "status": "NOT_RUN", "reason": "No identical pre-state, dynamics, h and controller contract after Gate 1 failure."},
            {"gate": 4, "name": "short horizon", "status": "NOT_RUN", "reason": "Formal cross-tool run prohibited; Torch VDP trace repair smoke is diagnostic only."},
            {"gate": 5, "name": "common time grid", "status": "NOT_RUN", "reason": "No common model/output contract to align."},
        ],
        "tightness": "N/A",
        "runtime_ranking": "N/A",
        "blocker": matched["unique_decisive_blocker"],
    }
    for relative in (
        "matched_runs/one_step/gate_status.json",
        "matched_runs/short_horizon/gate_status.json",
        "matched_runs/full_horizon/gate_status.json",
    ):
        write_json(OUT / relative, gates)
    write_text(OUT / "figures/README.md", "# Figures\n\nNo formal overlay or width plot was generated because Gate 1 failed. Plotting different state spaces would be misleading.")
    fresh = json.loads(FRESH_JSON.read_text(encoding="utf-8"))
    runtime = {
        "schema": "q3_runtime_classification_v1",
        "formal_matched_runtime_authorized": False,
        "native_xiangru_fresh": parse_xiangru_runtime(fresh, "b48_static", "complete_q3"),
        "process_wall_seconds_cold": float((FRESH_DIR / "wall_time.txt").read_text()),
        "resource_usage_file": str((FRESH_DIR / "resource_usage.txt").relative_to(ROOT)),
        "comparison": "N/A: GPU closed-loop runtime is not ranked against CPU plant-only runtime.",
        "repeat_policy": "One required exact fresh reproduction was run. Its native timing is evidence, not a formal repeated performance benchmark.",
    }
    write_json(OUT / "runtime/native_runtime_only.json", runtime)
    write_text(OUT / "runtime/README.md", "# Runtime\n\nOnly the fresh Xiangru native closed-loop timing is reported. Formal cross-tool runtime ranking is N/A because the model, controller workload, device and horizon contracts do not match.")


def trace_artifacts() -> None:
    trace_root = OUT / "trace_repair"
    short_on = json.loads((trace_root / "short_horizon/instrumented_run/summary.json").read_text())
    short_off = json.loads((trace_root / "short_horizon/uninstrumented_run/summary.json").read_text())
    numerical_fields = [
        "status", "requested_horizon", "completed_horizon", "completed_requested_horizon",
        "accepted_steps", "rejected_attempts", "full_tube", "last_segment",
        "raw_endpoint", "range_leaf_evaluations", "range_subdivision_invocations",
        "segment_boundary_conversion_count", "sample_sanity_status",
        "sample_sanity_max_violation",
    ]
    comparison = []
    for name in numerical_fields:
        left = short_on.get(name)
        right = short_off.get(name)
        comparison.append({"field": name, "instrumented": left, "uninstrumented": right, "equal": left == right})
    equivalence = {
        "schema": "trace_instrumentation_equivalence_v1",
        "horizon": 0.2,
        "fields": comparison,
        "numerically_equivalent": all(row["equal"] for row in comparison),
        "runtime_intentionally_excluded": True,
        "reason": "Trace/export overhead is observational and timing is not a numerical output.",
    }
    write_json(trace_root / "instrumentation_equivalence.json", equivalence)

    transitions_path = trace_root / "short_horizon/instrumented_trace/transitions.jsonl"
    rows = [json.loads(line) for line in transitions_path.read_text().splitlines() if line]
    stages = ["right_map_input", "right_map_output", "insertion_input", "insertion_output", "normalized_reset_input", "normalized_reset_output"]
    counts = {stage: 0 for stage in stages}
    missing = []
    for row in rows:
        stage = row.get("stage")
        if stage in counts:
            counts[stage] += 1
            if not row.get("object_content_sha256") and not row.get("unavailable_reason"):
                missing.append({"stage": stage, "attempt_index": row.get("attempt_index")})
    identity = {
        "schema": "transition_lifecycle_identity_summary_v1",
        "trace_schema": json.loads((trace_root / "short_horizon/instrumented_trace/trace_schema.json").read_text())["schema"],
        "stage_counts": counts,
        "missing_content_hashes": missing,
        "all_required_stages_observed": all(counts[stage] > 0 for stage in stages),
        "all_recorded_objects_hashed": not missing,
        "declared_unavailable_rows": sum(
            1 for row in rows if row.get("stage") in stages and row.get("unavailable_reason")
        ),
        "identity_tests": "tests/test_vdp_cross_step_carry_lineage_audit.py",
    }
    write_json(trace_root / "lifecycle_identity.json", identity)
    write_text(
        trace_root / "pre_fix_post_fix_field_map.md",
        """# Pre-fix/post-fix trace field map

| Field | Pre-fix behavior | Post-fix behavior |
|---|---|---|
| right_map_input | adjacent/pre-state substitute | exact previous `tmv_right` object |
| insertion_input/output | adjacent stage labels | exact `endpoint_without_constants` / `inserted` objects |
| normalized_reset_input/output | reset aliases/substitutes | exact `inserted_for_reset` / `reset_tm` objects |
| object identity | Python-adjacent assumptions | stable full-content SHA256 plus exact in-process object checks |
| rejected attempt index | could look like an accepted step | `accepted_step_index=null`, with accepted-count-before-attempt |
| call-44 component | roots could be mislabeled x | y roots and y result are component 1; x-only aggregations component 0 |
| ancestry | implied trajectory-wide | explicitly terminal-local; cross-step completeness false |

The post-fix schema is `vdp_transition_trace_schema_v2`. Missing lifecycle objects fail closed to null with a reason; adjacent objects are never substituted.
""",
    )


def manifest() -> None:
    path = OUT / "manifest.sha256"
    rows = []
    for item in sorted(OUT.rglob("*")):
        if item.is_file() and item != path:
            rows.append(f"{sha256(item)}  {item.relative_to(OUT)}")
    write_text(path, "\n".join(rows))


def main() -> int:
    reproduction_artifacts()
    order = order_artifacts()
    matched = contract_artifacts(order)
    gate_and_runtime_artifacts(matched)
    trace_artifacts()
    manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
