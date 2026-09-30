import copy
import hashlib
import json
from pathlib import Path

from torch_tm_flowpipe.archcomp26_preflight import (
    _contract_type_errors,
    preflight_reasons,
    validate_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
METHOD_FIELDS = {
    "arithmetic.mode",
    "arithmetic.controller_domain",
    "arithmetic.relaxation",
    "arithmetic.dtype",
    "arithmetic.transport",
    "runtime.hardware",
    "runtime.cpu_threads",
    "runtime.resource_limits",
    "runtime.timeout",
    "commands",
    "source_identity",
    "binary_identity",
}


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def inputs():
    manifest = load("benchmarks/archcomp26/manifest.json")
    matrix = load(manifest["execution_matrix"]["path"])
    return manifest, matrix


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_dotted(target, dotted, value):
    current = target
    parts = dotted.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def synthetic_inputs(tmp_path: Path, instance_id: str, profile: str):
    manifest, matrix = inputs()
    manifest = copy.deepcopy(manifest)
    matrix = copy.deepcopy(matrix)
    for link in (
        manifest["official_sources"]["asset_inventory"],
        manifest["official_sources"]["asset_inventory"]["audit_receipt"],
        manifest["instance_contract_record"], matrix["result_record_contract"]
    ):
        source = ROOT / link["path"]
        destination = tmp_path / link["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    evidence_path = tmp_path / "evidence.txt"
    evidence_path.write_text("synthetic contract evidence\n", encoding="utf-8")
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    required = manifest["unresolved_field_profiles"][profile]
    shared = [path for path in required if path not in METHOD_FIELDS]
    fields = {}
    for dotted in shared:
        if dotted.endswith(".source"):
            value = "evidence.txt"
        elif dotted.endswith(".sha256"):
            value = evidence_sha
        elif dotted in {"variable_order", "controller.input_output_order"}:
            value = ["x"]
        elif dotted == "initial_set.partitions":
            value = [{"x": [0.0, 1.0]}]
        elif dotted == "remainder.cutoff":
            value = 1e-6
        elif dotted == "remainder.cap":
            value = 0.1
        elif dotted in {
            "integration.step_size", "integration.solution_order",
            "integration.validation_order", "integration.horizon",
            "controller_update.period", "controller_update.nn_calls",
            "remainder.symbolic_queue",
            "discrete.transition_count", "discrete.sample_period",
            "controller_update.period_steps",
        }:
            value = 1
        elif dotted == "discrete.index_set":
            value = [0, 1]
        else:
            value = "synthetic-resolved"
        set_dotted(fields, dotted, value)
    fields["initial_set"]["boxes_sha256"] = canonical_sha256(
        fields["initial_set"]["partitions"]
    )
    record = {
        "schema_version": "archcomp26-instance-contract-v1",
        "instance_id": instance_id,
        "profile": profile,
        "fields": fields,
        "evidence": [{
            "path": "evidence.txt",
            "sha256": evidence_sha,
            "supports": shared,
        }],
    }
    record_path = Path("contracts") / f"{instance_id}.json"
    record_sha = write_json(tmp_path / record_path, record)
    row = next(row for row in manifest["instances"] if row["id"] == instance_id)
    row["contract"] = {
        "status": "resolved",
        "unresolved_field_profile": profile,
        "record": {
            "path": record_path.as_posix(),
            "sha256": record_sha,
            "schema_version": "archcomp26-instance-contract-v1",
        },
    }

    method = "pytorch_gpu"
    cell = copy.deepcopy(matrix["cell_defaults"])
    cell["support"] = {"status": "supported", "blockers": []}
    cell["command"] = {"argv": ["synthetic-run"], "cwd": "."}
    cell["source_identity"] = {
        "kind": "synthetic",
        "locator": "evidence.txt",
        "revision": None,
        "sha256": evidence_sha,
    }
    cell["binary_identity"] = {"path": "evidence.txt", "sha256": evidence_sha}
    cell["arithmetic"] = {
        "mode": "synthetic",
        "controller_domain": "synthetic",
        "relaxation": "synthetic",
        "dtype": "float64",
        "transport": "synthetic",
    }
    cell["runtime"] = {
        "hardware": "synthetic",
        "cpu_threads": 1,
        "gpu": "none",
        "resource_limits": {"memory": "synthetic"},
        "timeout_s": 1.0,
    }
    matrix["cells"][instance_id][method] = cell
    return manifest, matrix, record, cell, evidence_sha


def completed_result_record(manifest, cell, instance_id, evidence_sha):
    row = next(row for row in manifest["instances"] if row["id"] == instance_id)
    extent_kind = (
        "time_s"
        if row["contract"]["unresolved_field_profile"] == "full_execution_contract_v1"
        else "steps"
    )
    extent = {"kind": extent_kind, "value": 1}
    artifact = {"path": "evidence.txt", "sha256": evidence_sha}
    timing = {
        "process_total": 1.0,
        "driver_total": 0.9,
        "solver_core": 0.8,
        "compile": 0.0,
        "validation": 0.1,
        "observer_output": 0.0,
        "plot_report": 0.0,
    }
    samples = []
    for role, count in (
        ("cold", cell["measurement_plan"]["cold_runs"]),
        ("steady", cell["measurement_plan"]["steady_runs"]),
    ):
        for index in range(count):
            samples.append({
                "role": role,
                "index": index,
                "outcome": "completed",
                "timing_s": timing,
                "peak_memory_bytes": {"host": 1, "device": 0},
                "validated_extent": copy.deepcopy(extent),
                "accepted_steps": 1,
                "rejected_steps": 0,
                "nn_calls": 1,
                "failure": None,
                "artifact": artifact,
            })
    width = {
        "status": "complete",
        "domain": {"kind": extent_kind, "start": 0, "end": 1},
        "per_coordinate": [{
            "coordinate": "x",
            "union": {"lo": 0.0, "hi": 1.0, "width": 1.0},
            "per_partition_width": {"mean": 1.0, "max": 1.0},
        }],
        "artifact": artifact,
    }
    plan_fields = (
        "support", "command", "source_identity", "binary_identity",
        "arithmetic", "runtime", "measurement_plan",
    )
    return {
        "schema_version": "archcomp26-cell-result-v1",
        "instance_id": instance_id,
        "method": "pytorch_gpu",
        "contract_identity": {
            "instance_contract_sha256": row["contract"]["record"]["sha256"],
            "cell_plan_sha256": canonical_sha256({
                name: cell[name] for name in plan_fields
            }),
        },
        "measurement_plan": cell["measurement_plan"],
        "run": {
            "status": "completed",
            "outcome": "completed",
            "requested_extent": copy.deepcopy(extent),
            "validated_extent": copy.deepcopy(extent),
            "requested_horizon_completed": True,
            "accepted_steps": 1,
            "rejected_steps": 0,
            "nn_calls": 1,
            "first_failure": None,
        },
        "property": {
            "status": "passed",
            "checker": "synthetic-checker",
            "certificate_status": "not_applicable",
            "certificate_semantics": "synthetic explicit non-certificate",
            "artifact": artifact,
        },
        "eligibility": {
            "mathematical_contract_known": True,
            "requested_horizon_completed": True,
            "certificate_semantics_passed": False,
            "finite_outputs": True,
            "numerical_soundness_class": "empirically sampled only",
            "soundness_scope": "fixed workload",
            "formal_claim_eligible": False,
            "performance_measurement_eligible": True,
            "cross_tool_ranking_eligible": False,
        },
        "samples": samples,
        "widths": {
            "status": "complete",
            "common_prefix": copy.deepcopy(extent),
            "coordinate_order": ["x"],
            "coordinate_units": ["arbitrary"],
            "aggregation_semantics": "union_and_per_partition",
            "endpoint": {
                **copy.deepcopy(width),
                "domain": {"kind": extent_kind, "start": 1, "end": 1},
            },
            "last_segment_tube": copy.deepcopy(width),
            "full_horizon_tube": copy.deepcopy(width),
            "trajectory_artifact": artifact,
        },
        "artifacts": [
            {"role": role, "path": "evidence.txt", "sha256": evidence_sha}
            for role in ("command", "run_log", "result")
        ],
    }


def bind_completed_result(tmp_path, matrix, cell, instance_id, result):
    result_path = Path("results") / f"{instance_id}-pytorch-gpu.json"
    cell["result_record"] = {
        "schema_version": "archcomp26-cell-result-v1",
        "path": result_path.as_posix(),
        "sha256": write_json(tmp_path / result_path, result),
    }
    matrix["cells"][instance_id]["pytorch_gpu"] = cell
    matrix["status"] = "in_progress"
    return result_path


def test_current_matrix_is_structurally_valid_and_paused():
    manifest, matrix = inputs()
    assert validate_matrix(manifest, matrix) == []
    reasons = preflight_reasons(manifest, matrix, "quad-reach", "pytorch_gpu")
    assert {
        "experiments_paused",
        "contract_unresolved",
        "support_not_supported",
        "command_argv_missing",
        "command_cwd_missing",
        "source_identity_missing",
        "source_revision_or_sha_missing",
        "binary_identity_missing",
        "runtime_timeout_missing",
        "runtime_budget_missing",
    } <= set(reasons)


def test_partial_nested_override_is_rejected():
    manifest, matrix = inputs()
    broken = copy.deepcopy(matrix)
    broken["cells"]["quad-reach"]["pytorch_gpu"] = {
        "run": {"status": "completed"}
    }
    errors = validate_matrix(manifest, broken)
    assert any("incomplete override" in error for error in errors)


def test_preflight_does_not_treat_unknown_targets_as_launchable():
    manifest, matrix = inputs()
    assert "unknown_instance" in preflight_reasons(
        manifest, matrix, "missing", "pytorch_gpu"
    )
    assert "unknown_method" in preflight_reasons(
        manifest, matrix, "quad-reach", "missing"
    )


def test_status_flip_without_bound_contract_record_is_rejected():
    manifest, matrix = inputs()
    broken = copy.deepcopy(manifest)
    row = next(row for row in broken["instances"] if row["id"] == "quad-reach")
    row["contract"]["status"] = "resolved"
    errors = validate_matrix(broken, matrix)
    assert any("resolved contract" in error or "contract.record" in error
               for error in errors)
    assert "contract_invalid" in preflight_reasons(
        broken, matrix, "quad-reach", "pytorch_gpu"
    )


def test_missing_contract_record_is_the_only_launch_gate_in_valid_fixture(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    assert preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path
    ) == []
    row = next(row for row in manifest["instances"] if row["id"] == "acc-safe-distance")
    row["contract"].pop("record")
    reasons = preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path
    )
    assert "contract_invalid" in reasons
    assert "experiments_paused" not in reasons
    assert "contract_unresolved" not in reasons
    assert not any(reason.endswith("_missing") for reason in reasons)


def test_official_asset_inventory_and_receipt_are_hash_bound(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    inventory = manifest["official_sources"]["asset_inventory"]
    for link, label in (
        (inventory, "official_sources.asset_inventory"),
        (inventory["audit_receipt"], "official_sources.asset_inventory.audit_receipt"),
    ):
        path = tmp_path / link["path"]
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(label in error and "SHA-256" in error for error in errors)
        path.write_bytes(original)

    broken = copy.deepcopy(manifest)
    broken["official_sources"]["asset_inventory"]["schema_version"] = "wrong"
    assert any("inventory has the wrong schema_version" in error for error in
               validate_matrix(broken, matrix, root=tmp_path))
    broken = copy.deepcopy(manifest)
    broken["official_sources"]["asset_inventory"]["audit_receipt"][
        "schema_version"
    ] = "wrong"
    assert any("audit receipt has the wrong schema_version" in error for error in
               validate_matrix(broken, matrix, root=tmp_path))


def test_contract_evidence_supports_reject_duplicates(tmp_path):
    manifest, matrix, record, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    row = next(row for row in manifest["instances"] if row["id"] == "acc-safe-distance")
    record_path = tmp_path / row["contract"]["record"]["path"]
    record["evidence"][0]["supports"].append(record["evidence"][0]["supports"][0])
    row["contract"]["record"]["sha256"] = write_json(record_path, record)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("supports: duplicate fields" in error for error in errors)


def test_fractional_remainder_contract_values_and_invalid_boundaries(tmp_path):
    manifest, matrix, record, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    assert record["fields"]["remainder"] == {
        "cutoff": 1e-6, "cap": 0.1, "symbolic_queue": 1,
    }
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    zero_remainder = copy.deepcopy(record["fields"])
    zero_remainder["remainder"].update({"cutoff": 0.0, "cap": 0.0})
    assert _contract_type_errors(
        zero_remainder, "full_execution_contract_v1", "acc-safe-distance"
    ) == []
    for dotted, invalid in (
        ("remainder.cutoff", True),
        ("remainder.cutoff", -1e-6),
        ("remainder.cutoff", float("nan")),
        ("remainder.cap", True),
        ("remainder.cap", -0.1),
        ("remainder.cap", float("inf")),
        ("remainder.symbolic_queue", 1.5),
        ("remainder.symbolic_queue", True),
        ("remainder.symbolic_queue", -1),
    ):
        broken = copy.deepcopy(record)
        set_dotted(broken["fields"], dotted, invalid)
        errors = _contract_type_errors(
            broken["fields"], "full_execution_contract_v1", "acc-safe-distance"
        )
        assert any(dotted in error for error in errors), (dotted, invalid)


def test_synthetic_continuous_contract_and_cell_complete_full_profile(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    assert preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path
    ) == []


def test_synthetic_discrete_contract_and_cell_complete_full_profile(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "airplane-discrete", "discrete_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    assert preflight_reasons(
        manifest, matrix, "airplane-discrete", "pytorch_gpu", root=tmp_path
    ) == []


def test_every_shared_contract_field_is_fail_closed(tmp_path):
    manifest, matrix, record, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    row = next(row for row in manifest["instances"] if row["id"] == "acc-safe-distance")
    record_path = tmp_path / row["contract"]["record"]["path"]
    for dotted in record["evidence"][0]["supports"]:
        broken = copy.deepcopy(record)
        current = broken["fields"]
        parts = dotted.split(".")
        for part in parts[:-1]:
            current = current[part]
        current.pop(parts[-1])
        row["contract"]["record"]["sha256"] = write_json(record_path, broken)
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(dotted in error and "missing" in error for error in errors), dotted


def test_terminal_cell_without_result_record_is_rejected(tmp_path):
    manifest, matrix, _, cell, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None
    }
    matrix["cells"]["acc-safe-distance"]["pytorch_gpu"] = cell
    matrix["status"] = "in_progress"
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("terminal cell lacks a result record" in error for error in errors)


def test_hash_bound_completed_result_requires_full_samples_and_widths(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = Path("results/acc-pytorch-gpu.json")
    result_sha = write_json(tmp_path / result_path, result)
    cell["result_record"] = {
        "schema_version": "archcomp26-cell-result-v1",
        "path": result_path.as_posix(),
        "sha256": result_sha,
    }
    matrix["cells"]["acc-safe-distance"]["pytorch_gpu"] = cell
    matrix["status"] = "in_progress"
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    missing_sample = copy.deepcopy(result)
    missing_sample["samples"].pop()
    cell["result_record"]["sha256"] = write_json(
        tmp_path / result_path, missing_sample
    )
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("lacks every planned sample" in error for error in errors)

    missing_width = copy.deepcopy(result)
    missing_width["widths"]["endpoint"] = {
        "status": "unavailable",
        "domain": {"kind": "steps", "start": 0, "end": 1},
        "per_coordinate": [],
        "artifact": None,
    }
    cell["result_record"]["sha256"] = write_json(
        tmp_path / result_path, missing_width
    )
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("all three complete views" in error for error in errors)

    cell["result_record"]["sha256"] = result_sha
    write_json(tmp_path / result_path, result)
    (tmp_path / "evidence.txt").write_text("tampered\n", encoding="utf-8")
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("SHA-256 does not match current file" in error for error in errors)


def test_completed_extent_is_bound_to_continuous_and_discrete_contracts(tmp_path):
    for instance_id, profile, expected_kind in (
        ("acc-safe-distance", "full_execution_contract_v1", "time_s"),
        ("airplane-discrete", "discrete_execution_contract_v1", "steps"),
    ):
        manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
            tmp_path, instance_id, profile
        )
        cell["run"] = {
            "status": "completed", "failure_category": None, "failure_detail": None
        }
        result = completed_result_record(manifest, cell, instance_id, evidence_sha)
        result_path = bind_completed_result(
            tmp_path, matrix, cell, instance_id, result
        )
        assert result["run"]["requested_extent"]["kind"] == expected_kind
        assert validate_matrix(manifest, matrix, root=tmp_path) == []

        result["run"]["requested_extent"]["value"] = 2
        cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(
            "requested extent does not match resolved instance contract" in error
            for error in errors
        )


def test_completed_samples_match_run_extent_and_counts(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None
    }
    original = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", original
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    mutations = (
        ("validated_extent", {"kind": "time_s", "value": 0}, "extent disagrees"),
        ("accepted_steps", 2, "accepted_steps: disagrees"),
        ("rejected_steps", 1, "rejected_steps: disagrees"),
        ("nn_calls", 2, "nn_calls: disagrees"),
    )
    for field, value, message in mutations:
        result = copy.deepcopy(original)
        result["samples"][0][field] = value
        cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(message in error for error in errors), field


def test_completed_work_counts_are_bound_to_resolved_contract(tmp_path):
    cases = (
        (
            "acc-safe-distance",
            "full_execution_contract_v1",
            "nn_calls",
            999,
            "NN calls do not match contract",
        ),
        (
            "airplane-discrete",
            "discrete_execution_contract_v1",
            "accepted_steps",
            0,
            "accepted steps do not match contract",
        ),
    )
    for instance_id, profile, field, value, message in cases:
        manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
            tmp_path, instance_id, profile
        )
        cell["run"] = {
            "status": "completed", "failure_category": None, "failure_detail": None
        }
        result = completed_result_record(manifest, cell, instance_id, evidence_sha)
        result["run"][field] = value
        for sample in result["samples"]:
            sample[field] = value
        bind_completed_result(tmp_path, matrix, cell, instance_id, result)
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(message in error for error in errors), instance_id


def test_completed_widths_are_bound_to_contract_and_full_run(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None
    }
    original = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", original
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    result = copy.deepcopy(original)
    result["widths"]["coordinate_order"] = ["not-x"]
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("coordinate order does not match contract" in error for error in errors)

    result = copy.deepcopy(original)
    result["widths"]["common_prefix"]["value"] = 0
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("completed common prefix must be positive" in error for error in errors)

    for view in ("endpoint", "last_segment_tube", "full_horizon_tube"):
        result = copy.deepcopy(original)
        result["widths"][view]["domain"]["start"] = 0.5
        cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(
            f"widths.{view}: domain disagrees with completed run" in error
            for error in errors
        ), view

    result = copy.deepcopy(original)
    for view in ("endpoint", "last_segment_tube", "full_horizon_tube"):
        partition = result["widths"][view]["per_coordinate"][0][
            "per_partition_width"
        ]
        partition["mean"] = 100.0
        partition["max"] = 100.0
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("max exceeds union width" in error for error in errors)


def test_eligibility_is_constrained_by_certificate_run_and_finiteness(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None
    }
    original = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", original
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    result = copy.deepcopy(original)
    result["property"]["status"] = "failed"
    result["property"]["certificate_status"] = "passed"
    result["eligibility"]["certificate_semantics_passed"] = True
    result["eligibility"]["numerical_soundness_class"] = (
        "independently outward replayed for exact benchmark workload"
    )
    result["eligibility"]["cross_tool_ranking_eligible"] = True
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    result = copy.deepcopy(original)
    result["property"]["certificate_status"] = "failed"
    result["eligibility"]["cross_tool_ranking_eligible"] = True
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("ranking eligibility lacks prerequisites" in error for error in errors)

    result = copy.deepcopy(original)
    result["property"]["certificate_status"] = "failed"
    result["eligibility"]["certificate_semantics_passed"] = True
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("certificate eligibility disagrees with property" in error for error in errors)

    result = copy.deepcopy(original)
    result["eligibility"]["finite_outputs"] = False
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("completed run must have finite outputs" in error for error in errors)

    for soundness, scope in (
        ("unsound/ineligible on a demonstrated counterexample", "fixed workload"),
        ("empirically sampled only", "fixed workload"),
        ("formally outward by construction", "one step"),
    ):
        result = copy.deepcopy(original)
        result["property"]["certificate_status"] = "passed"
        result["eligibility"]["certificate_semantics_passed"] = True
        result["eligibility"]["numerical_soundness_class"] = soundness
        result["eligibility"]["soundness_scope"] = scope
        result["eligibility"]["cross_tool_ranking_eligible"] = True
        cell["result_record"]["sha256"] = write_json(
            tmp_path / result_path, result
        )
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(
            "ranking eligibility lacks prerequisites" in error for error in errors
        ), (soundness, scope)


def test_terminal_cells_require_the_shared_executable_plan_gate(tmp_path):
    cases = (
        ("support", lambda cell: cell.__setitem__(
            "support", {"status": "unassessed", "blockers": []}
        ), "support_not_supported"),
        ("command", lambda cell: cell["command"].__setitem__("argv", None),
         "command_argv_missing"),
        ("source", lambda cell: cell["source_identity"].__setitem__("locator", None),
         "source_identity_missing"),
        ("binary", lambda cell: cell["binary_identity"].__setitem__("path", None),
         "binary_identity_missing"),
        ("runtime", lambda cell: cell["runtime"].__setitem__("hardware", None),
         "runtime_budget_missing"),
        ("arithmetic", lambda cell: cell["arithmetic"].__setitem__("mode", None),
         "arithmetic_mode_missing"),
    )
    for label, mutate, reason in cases:
        manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
            tmp_path, "acc-safe-distance", "full_execution_contract_v1"
        )
        mutate(cell)
        cell["run"] = {
            "status": "completed", "failure_category": None, "failure_detail": None
        }
        result = completed_result_record(
            manifest, cell, "acc-safe-distance", evidence_sha
        )
        bind_completed_result(tmp_path, matrix, cell, "acc-safe-distance", result)
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(
            "terminal cell lacks executable plan" in error and reason in error
            for error in errors
        ), label


def test_evidence_backed_unsupported_skip_does_not_require_executable_plan(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["support"] = {
        "status": "unsupported",
        "blockers": ["implementation is unavailable; see the bound result record"],
    }
    for name in ("command", "source_identity", "binary_identity", "arithmetic", "runtime"):
        cell[name] = copy.deepcopy(matrix["cell_defaults"][name])
    cell["run"] = {
        "status": "skipped",
        "failure_category": "missing_dependency",
        "failure_detail": "No applicable implementation exists for this method.",
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result["run"] = {
        "status": "skipped",
        "outcome": "missing_dependency",
        "requested_extent": {"kind": "time_s", "value": 1},
        "validated_extent": {"kind": "time_s", "value": 0},
        "requested_horizon_completed": False,
        "accepted_steps": 0,
        "rejected_steps": 0,
        "nn_calls": 0,
        "first_failure": {
            "stage": "support_assessment",
            "reason_code": "missing_dependency",
            "step": None,
            "time_s": None,
            "detail": "No applicable implementation exists for this method.",
            "exit_code": None,
            "signal": None,
        },
    }
    result["property"] = {
        "status": "not_checked",
        "checker": None,
        "certificate_status": "not_checked",
        "certificate_semantics": None,
        "artifact": None,
    }
    result["eligibility"] = {
        "mathematical_contract_known": True,
        "requested_horizon_completed": False,
        "certificate_semantics_passed": False,
        "finite_outputs": True,
        "numerical_soundness_class": "unknown",
        "soundness_scope": "fixed workload",
        "formal_claim_eligible": False,
        "performance_measurement_eligible": False,
        "cross_tool_ranking_eligible": False,
    }
    result["samples"] = []
    unavailable = {
        "status": "unavailable",
        "domain": {"kind": "time_s", "start": 0, "end": 0},
        "per_coordinate": [],
        "artifact": None,
    }
    result["widths"] = {
        "status": "unavailable",
        "common_prefix": {"kind": "time_s", "value": 0},
        "coordinate_order": ["x"],
        "coordinate_units": ["arbitrary"],
        "aggregation_semantics": "union_and_per_partition",
        "endpoint": copy.deepcopy(unavailable),
        "last_segment_tube": copy.deepcopy(unavailable),
        "full_horizon_tube": copy.deepcopy(unavailable),
        "trajectory_artifact": None,
    }
    original = copy.deepcopy(result)
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    mutations = (
        (
            lambda value: value["run"].__setitem__(
                "requested_extent", {"kind": "steps", "value": 999}
            ),
            "requested extent does not match resolved instance contract",
        ),
        (
            lambda value: value["run"].__setitem__(
                "validated_extent", {"kind": "time_s", "value": 9999}
            ),
            "validated extent is outside requested extent",
        ),
        (
            lambda value: value["widths"].__setitem__(
                "common_prefix", {"kind": "steps", "value": 9999}
            ),
            "common prefix disagrees with validated extent",
        ),
        (
            lambda value: value["widths"]["endpoint"].__setitem__("domain", None),
            "widths.endpoint.domain: expected an object",
        ),
        (
            lambda value: value["run"]["first_failure"].__setitem__(
                "reason_code", "timeout"
            ),
            "first failure reason disagrees with outcome",
        ),
        (
            lambda value: value["run"]["first_failure"].__setitem__(
                "detail", "different detail"
            ),
            "first failure detail disagrees with cell",
        ),
        (
            lambda value: value["run"].__setitem__("accepted_steps", 1),
            "skipped run must have zero work counts",
        ),
    )
    for mutate, message in mutations:
        result = copy.deepcopy(original)
        mutate(result)
        cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
        errors = validate_matrix(manifest, matrix, root=tmp_path)
        assert any(message in error for error in errors), message


def test_malformed_enum_values_report_errors_instead_of_raising(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None
    }
    original = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", original
    )
    mutations = (
        lambda result: result["run"].__setitem__("outcome", []),
        lambda result: result["samples"][0].__setitem__("role", []),
        lambda result: result["samples"][0].__setitem__("outcome", []),
        lambda result: result["run"]["requested_extent"].__setitem__("kind", []),
        lambda result: result["property"].__setitem__("certificate_status", []),
        lambda result: result["eligibility"].__setitem__(
            "numerical_soundness_class", []
        ),
        lambda result: (
            result["eligibility"].__setitem__("cross_tool_ranking_eligible", True),
            result["eligibility"].__setitem__("soundness_scope", []),
        ),
        lambda result: result["widths"].__setitem__("status", []),
        lambda result: result["widths"]["endpoint"]["domain"].__setitem__(
            "kind", []
        ),
        lambda result: result["widths"]["endpoint"]["per_coordinate"][0].__setitem__(
            "union", []
        ),
        lambda result: result["widths"]["endpoint"]["per_coordinate"][0][
            "union"
        ].pop("width"),
    )
    for mutate in mutations:
        result = copy.deepcopy(original)
        mutate(result)
        cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
        assert validate_matrix(manifest, matrix, root=tmp_path)

    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, original)
    for dotted in ("support.status", "run.status", "run.failure_category"):
        broken = copy.deepcopy(matrix)
        set_dotted(
            broken["cells"]["acc-safe-distance"]["pytorch_gpu"], dotted, []
        )
        assert validate_matrix(manifest, broken, root=tmp_path)

    broken = copy.deepcopy(matrix)
    broken["cells"]["acc-safe-distance"]["pytorch_gpu"]["measurement_plan"][
        "cold_runs"
    ] = []
    assert validate_matrix(manifest, broken, root=tmp_path)

    broken = copy.deepcopy(matrix)
    broken["cells"]["acc-safe-distance"]["pytorch_gpu"]["run"] = []
    assert validate_matrix(manifest, broken, root=tmp_path)


def test_nested_artifact_may_carry_optional_role(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result["property"]["artifact"] = {
        **result["property"]["artifact"], "role": "property_certificate"
    }
    bind_completed_result(tmp_path, matrix, cell, "acc-safe-distance", result)
    assert validate_matrix(manifest, matrix, root=tmp_path) == []


def test_current_repository_has_no_resolved_contract_fixture():
    manifest, _ = inputs()
    assert all(
        row["contract"]["status"] == "unresolved"
        and "record" not in row["contract"]
        for row in manifest["instances"]
    )
