import copy
import hashlib
import json
from pathlib import Path

from torch_tm_flowpipe.archcomp26_preflight import (
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
    inventory = load("benchmarks/archcomp26/official_assets.json")
    manifest["official_sources"]["asset_inventory"]["path"] = "official_assets.json"
    write_json(tmp_path / "official_assets.json", inventory)
    for link in (
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
        elif dotted in {
            "integration.step_size", "integration.solution_order",
            "integration.validation_order", "integration.horizon",
            "controller_update.period", "controller_update.nn_calls",
            "remainder.cutoff", "remainder.cap", "remainder.symbolic_queue",
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
                "validated_extent": {"kind": "steps", "value": 1},
                "accepted_steps": 1,
                "rejected_steps": 0,
                "nn_calls": 1,
                "failure": None,
                "artifact": artifact,
            })
    width = {
        "status": "complete",
        "domain": {"kind": "steps", "start": 0, "end": 1},
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
            "requested_extent": {"kind": "steps", "value": 1},
            "validated_extent": {"kind": "steps", "value": 1},
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
            "common_prefix": {"kind": "steps", "value": 1},
            "coordinate_order": ["x"],
            "coordinate_units": ["arbitrary"],
            "aggregation_semantics": "union_and_per_partition",
            "endpoint": copy.deepcopy(width),
            "last_segment_tube": copy.deepcopy(width),
            "full_horizon_tube": copy.deepcopy(width),
            "trajectory_artifact": artifact,
        },
        "artifacts": [
            {"role": role, "path": "evidence.txt", "sha256": evidence_sha}
            for role in ("command", "run_log", "result")
        ],
    }


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


def test_current_repository_has_no_resolved_contract_fixture():
    manifest, _ = inputs()
    assert all(
        row["contract"]["status"] == "unresolved"
        and "record" not in row["contract"]
        for row in manifest["instances"]
    )
