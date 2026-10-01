import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import zlib

import pytest

from torch_tm_flowpipe.archcomp26_preflight import (
    _validate_controller_update_trace,
    _validate_widths,
    preflight_reasons,
    validate_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
KNOWN_NATIVE_RESULT = (
    ROOT
    / "research/gpu_verified_20260930/evidence/native_terminal_20260930/RESULT.json"
)
SYNTHETIC_LAUNCH_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
METHOD_FIELDS = {
    "integration.step_size",
    "integration.solution_order",
    "integration.point_order",
    "integration.validation_order",
    "controller_update.nn_calls",
    "remainder.cutoff",
    "remainder.cap",
    "remainder.symbolic_queue",
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
    "property.checker",
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


def json_bytes(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def synthetic_schedule(manifest):
    methods = list(manifest["methods"])
    entries = []
    for instance_index, row in enumerate(manifest["instances"]):
        for method_index, method in enumerate(methods):
            for role, count in (("cold", 1), ("steady", 5)):
                for index in range(count):
                    round_index = index if role == "steady" else 0
                    offset = (instance_index + round_index) % len(methods)
                    entries.append({
                        "instance_id": row["id"],
                        "method": method,
                        "role": role,
                        "index": index,
                        "round_index": round_index,
                        "sequence_position": (method_index - offset) % len(methods),
                    })
    return {
        "schema_version": "archcomp26-campaign-schedule-v1",
        "campaign_id": "synthetic-four-way-campaign",
        "entries": entries,
    }


def synthetic_launch_identity():
    server_root = "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z"
    run_directory = (
        server_root
        + "/runs/native_quad_matched_20260929/initial_affine_cover_variant/"
        "full1000_v1"
    )
    return {
        "schema_version": "archcomp26-native-job-launch-v1",
        "server_research_root": server_root,
        "run_directory": run_directory,
        "watch_directory": run_directory + "_watch",
        "job_id": "native-quad-matched-20260929-full1000-v1",
        "command_sha256": (
            "db3f45544bf0cd4da63f90cf3973a7b76347f585809121783b87d709d818c62e"
        ),
        "created_at_utc": "2025-12-30T00:00:00Z",
    }


def synthetic_terminal_evidence():
    server_root = "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z"
    run_directory = (
        server_root
        + "/runs/native_quad_matched_20260929/initial_affine_cover_variant/"
        "full1000_v1"
    )
    result_payload = KNOWN_NATIVE_RESULT.read_bytes()
    return {
        "schema_version": "archcomp26-native-job-terminal-v1",
        "server_research_root": server_root,
        "run_directory": run_directory,
        "job_id": "native-quad-matched-20260929-full1000-v1",
        "terminal_status": "timeout",
        "checked_at_utc": "2025-12-31T23:58:00Z",
        "result_artifact": {
            "path": "campaign/original-result.json",
            "sha256": hashlib.sha256(result_payload).hexdigest(),
        },
    }


def synthetic_process_scan():
    server_root = "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z"
    run_directory = (
        server_root
        + "/runs/native_quad_matched_20260929/initial_affine_cover_variant/"
        "full1000_v1"
    )
    return {
        "schema_version": "archcomp26-process-scan-v1",
        "campaign_id": "synthetic-four-way-campaign",
        "server_research_root": server_root,
        "checked_at_utc": "2025-12-31T23:59:00Z",
        "active_owned_processes": [],
        "observed_run_directories": [run_directory],
    }


def synthetic_prelaunch_audit():
    server_root = "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z"
    run_directory = (
        server_root
        + "/runs/native_quad_matched_20260929/initial_affine_cover_variant/"
        "full1000_v1"
    )
    launch_payload = json_bytes(synthetic_launch_identity())
    terminal_payload = json_bytes(synthetic_terminal_evidence())
    scan_payload = json_bytes(synthetic_process_scan())
    return {
        "schema_version": "archcomp26-prelaunch-audit-v1",
        "campaign_id": "synthetic-four-way-campaign",
        "checked_at_utc": "2025-12-31T23:59:00Z",
        "valid_until_utc": "2026-01-01T23:59:00Z",
        "server_research_root": server_root,
        "process_scan_artifact": {
            "path": "campaign/process-scan.json",
            "sha256": hashlib.sha256(scan_payload).hexdigest(),
        },
        "original_native_quad": {
            "job_id": "native-quad-matched-20260929-full1000-v1",
            "run_directory": run_directory,
            "watch_directory": run_directory + "_watch",
            "launch_identity_artifact": {
                "path": "campaign/original-launch.json",
                "sha256": hashlib.sha256(launch_payload).hexdigest(),
            },
            "terminal_status": "timeout",
            "terminal_evidence_artifact": {
                "path": "campaign/original-terminal.json",
                "sha256": hashlib.sha256(terminal_payload).hexdigest(),
            },
        },
        "replacement_jobs": [],
        "duplicate_launch_absent": True,
    }


def synthetic_campaign(manifest):
    schedule_bytes = (
        json.dumps(synthetic_schedule(manifest), indent=2) + "\n"
    ).encode()
    audit_bytes = (
        json.dumps(synthetic_prelaunch_audit(), indent=2) + "\n"
    ).encode()
    return {
        "schema_version": "archcomp26-comparison-campaign-v1",
        "campaign_id": "synthetic-four-way-campaign",
        "host_identity": "synthetic-host",
        "hardware_identity": "synthetic",
        "cpu_thread_budget": 1,
        "gpu_device_budget": "none",
        "timeout_s": 1.0,
        "resource_limits": {
            "exclusive_host": True,
            "exclusive_gpu_device": True,
            "max_host_memory_bytes": 1024,
            "max_device_memory_bytes": 0,
        },
        "prelaunch_audit": {
            "path": "campaign/prelaunch-audit.json",
            "sha256": hashlib.sha256(audit_bytes).hexdigest(),
        },
        "active_run_receipt": {"path": None, "sha256": None},
        "launch_guard": {
            "schema_version": "archcomp26-atomic-launch-guard-v1",
            "protocol": "posix-flock-exclusive-nonblocking-v1",
            "lock_path": (
                "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/"
                ".archcomp26/launch.lock"
            ),
            "hold_scope": "fresh_audit_through_terminal_fsync_v1",
            "wrapper_module": "torch_tm_flowpipe.archcomp26_launch",
            "wrapper_path": "src/torch_tm_flowpipe/archcomp26_launch.py",
            "wrapper_sha256": hashlib.sha256(
                b"# synthetic atomic launch guard fixture\n"
            ).hexdigest(),
        },
        "timing_boundary": {
            "version": "total_configuration_v2",
            "start_event": "immediately_before_fresh_process_spawn",
            "stop_event": "after_result_and_width_artifacts_are_durable",
            "phase_fields": [
                "driver_total", "compile", "controller_nn", "solver_core",
                "validation", "observer", "output", "plot_report",
            ],
        },
        "rotation": {
            "policy": "balanced_round_robin_by_instance_and_steady_round_v1",
            "steady_rounds": 5,
            "schedule_artifact": {
                "path": "campaign/schedule.json",
                "sha256": hashlib.sha256(schedule_bytes).hexdigest(),
            },
        },
    }


def attempt_artifact(method, role, index):
    payload = f"{method}:{role}:{index}\n".encode()
    return {
        "path": f"attempts/{method}-{role}-{index}.txt",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


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
    launch_guard_path = tmp_path / "src/torch_tm_flowpipe/archcomp26_launch.py"
    launch_guard_path.parent.mkdir(parents=True, exist_ok=True)
    launch_guard_path.write_bytes(b"# synthetic atomic launch guard fixture\n")
    schedule_path = tmp_path / "campaign/schedule.json"
    write_json(schedule_path, synthetic_schedule(manifest))
    write_json(
        tmp_path / "campaign/prelaunch-audit.json", synthetic_prelaunch_audit()
    )
    write_json(tmp_path / "campaign/original-launch.json", synthetic_launch_identity())
    write_json(
        tmp_path / "campaign/original-terminal.json", synthetic_terminal_evidence()
    )
    write_json(tmp_path / "campaign/process-scan.json", synthetic_process_scan())
    (tmp_path / "campaign/original-result.json").write_bytes(
        KNOWN_NATIVE_RESULT.read_bytes()
    )
    matrix["comparison_campaign"] = synthetic_campaign(manifest)
    for method_name in matrix["methods"]:
        for role, count in (("cold", 1), ("steady", 5)):
            for index in range(count):
                link = attempt_artifact(method_name, role, index)
                attempt_path = tmp_path / link["path"]
                attempt_path.parent.mkdir(parents=True, exist_ok=True)
                attempt_path.write_bytes(
                    f"{method_name}:{role}:{index}\n".encode()
                )
    required = manifest["unresolved_field_profiles"][profile]
    shared = list(required)
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
        elif dotted == "dynamics.equations":
            value = ["x'=u"]
        elif dotted == "transition.state_update":
            value = ["x_next=x+u"]
        elif dotted == "disturbance":
            value = {"mode": "none", "bounds": {}}
        elif dotted == "width_comparison.coordinate_units":
            value = ["arbitrary"]
        elif dotted == "width_comparison.sample_points":
            value = [0, 1]
        elif dotted == "width_comparison.aggregation_semantics":
            value = "union_and_per_partition"
        elif dotted == "controller_update.schedule_points":
            value = [0]
        elif dotted in {
            "integration.horizon", "controller_update.period",
            "controller_update.scheduled_updates",
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
    if profile == "full_execution_contract_v1":
        cell["numerics"] = {
            "integration": {
                "step_size": {"mode": "fixed", "value": 1.0},
                "solution_order": {"mode": "configured", "value": 2},
                "point_order": {"mode": "configured", "value": 1},
                "validation_order": {"mode": "configured", "value": 3},
                "semantics": "fixed plant step; configured method orders",
            },
            "remainder": {
                "cutoff": {"mode": "configured", "value": 1e-6},
                "cap": {"mode": "configured", "value": 0.1},
                "symbolic_queue": {"mode": "configured", "value": 1},
                "semantics": "synthetic remainder and symbolic-remainder policy",
            },
        }
    else:
        cell["numerics"] = {
            "integration": {
                "step_size": {"mode": "not_applicable", "value": None},
                "solution_order": {"mode": "not_applicable", "value": None},
                "point_order": {"mode": "not_applicable", "value": None},
                "validation_order": {"mode": "not_applicable", "value": None},
                "semantics": "discrete transition; ODE integration is not applicable",
            },
            "remainder": {
                "cutoff": {"mode": "not_applicable", "value": None},
                "cap": {"mode": "not_applicable", "value": None},
                "symbolic_queue": {"mode": "not_applicable", "value": None},
                "semantics": "discrete transition; ODE remainder is not applicable",
            },
        }
    cell["controller_execution"] = {
        "scheduled_updates": 1,
        "nn_calls": {"mode": "exact", "value": 1},
        "nn_call_semantics": "one bound evaluation per scheduled update",
    }
    cell["property_checker"] = {
        "mode": "configured",
        "identity": "synthetic-checker",
        "semantics": "evaluate the shared property after the full run",
        "certificate_semantics": "synthetic explicit non-certificate",
        "early_stop_policy": "never",
    }
    cell["runtime"] = {
        "hardware": "synthetic",
        "cpu_threads": 1,
        "gpu": "none",
        "resource_limits": {
            "exclusive_host": True,
            "exclusive_gpu_device": True,
            "max_host_memory_bytes": 1024,
            "max_device_memory_bytes": 0,
        },
        "timeout_s": 1.0,
    }
    matrix["cells"][instance_id][method] = cell
    return manifest, matrix, record, cell, evidence_sha


def completed_result_record(
    manifest, cell, instance_id, evidence_sha, method="pytorch_gpu"
):
    row = next(row for row in manifest["instances"] if row["id"] == instance_id)
    extent_kind = (
        "time_s"
        if row["contract"]["unresolved_field_profile"] == "full_execution_contract_v1"
        else "steps"
    )
    extent = {"kind": extent_kind, "value": 1}
    if extent_kind == "time_s":
        step = cell["numerics"]["integration"]["step_size"]["value"]
        if isinstance(step, (int, float)) and not isinstance(step, bool) and step > 0:
            accepted_steps = round(1 / step)
            last_segment_start = (accepted_steps - 1) * step
        else:
            accepted_steps = 0
            last_segment_start = 0
    else:
        accepted_steps = 1
        last_segment_start = 0
    nn_plan = cell["controller_execution"]["nn_calls"]
    nn_calls = nn_plan["value"] if isinstance(nn_plan["value"], int) else 0
    artifact = {"path": "evidence.txt", "sha256": evidence_sha}
    coverage = {
        "requested_partitions": 1,
        "completed_partitions": 1,
        "failed_partitions": 0,
        "unattempted_partitions": 0,
        "boxes_sha256": row["contract"]["record"]["sha256"],
        "ledger_artifact": artifact,
    }
    # Replace the record SHA placeholder with the bound boxes SHA from the
    # deterministic one-box synthetic contract.
    coverage["boxes_sha256"] = canonical_sha256([{"x": [0.0, 1.0]}])
    timing = {
        "process_total": 1.0,
        "driver_total": 0.9,
        "compile": 0.0,
        "controller_nn": 0.1,
        "solver_core": 0.7,
        "validation": 0.1,
        "observer": 0.0,
        "output": 0.0,
        "plot_report": 0.0,
    }
    samples = []
    for role, count in (
        ("cold", cell["measurement_plan"]["cold_runs"]),
        ("steady", cell["measurement_plan"]["steady_runs"]),
    ):
        for index in range(count):
            round_index = index if role == "steady" else 0
            sequence_position = (
                list(manifest["methods"]).index(method)
                - (
                    [item["id"] for item in manifest["instances"]].index(instance_id)
                    + round_index
                ) % len(manifest["methods"])
            ) % len(manifest["methods"])
            started_second = (
                (0 if role == "cold" else index + 1) * 10
                + 2 * sequence_position
            )
            samples.append({
                "role": role,
                "index": index,
                "attempt_index": 0,
                "included_in_timing": True,
                "campaign": {
                    "campaign_id": "synthetic-four-way-campaign",
                    "round_index": round_index,
                    "sequence_position": sequence_position,
                },
                "process_identity": {
                    "invocation_id": f"{instance_id}:{method}:{role}:{index}",
                    "pid": 1000 + len(samples),
                    "started_at_utc": (
                        f"2026-01-01T00:00:{started_second:02d}Z"
                    ),
                    "finished_at_utc": (
                        f"2026-01-01T00:00:{started_second + 1:02d}Z"
                    ),
                },
                "outcome": "completed",
                "timing_s": timing,
                "peak_memory_bytes": {"host": 1, "device": 0},
                "validated_extent": copy.deepcopy(extent),
                "accepted_steps": accepted_steps,
                "rejected_steps": 0,
                "controller_updates": cell["controller_execution"]["scheduled_updates"],
                "controller_update_trace": {
                    "kind": extent_kind,
                    "points": [0],
                    "boundary_update_status": "not_scheduled",
                    "artifact": attempt_artifact(method, role, index),
                },
                "nn_calls": nn_calls,
                "partition_coverage": copy.deepcopy(coverage),
                "failure": None,
                "artifact": attempt_artifact(method, role, index),
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
        "arithmetic", "numerics", "controller_execution", "property_checker",
        "runtime", "measurement_plan",
    )
    return {
        "schema_version": "archcomp26-cell-result-v4",
        "instance_id": instance_id,
        "method": method,
        "contract_identity": {
            "instance_contract_sha256": row["contract"]["record"]["sha256"],
            "cell_plan_sha256": canonical_sha256({
                name: cell[name] for name in plan_fields
            }),
            "campaign_sha256": canonical_sha256(synthetic_campaign(manifest)),
        },
        "measurement_plan": cell["measurement_plan"],
        "run": {
            "status": "completed",
            "outcome": "completed",
            "requested_extent": copy.deepcopy(extent),
            "validated_extent": copy.deepcopy(extent),
            "requested_horizon_completed": True,
            "accepted_steps": accepted_steps,
            "rejected_steps": 0,
            "controller_updates": cell["controller_execution"]["scheduled_updates"],
            "controller_update_trace": {
                "kind": extent_kind,
                "points": [0],
                "boundary_update_status": "not_scheduled",
                "artifact": artifact,
            },
            "nn_calls": nn_calls,
            "partition_coverage": copy.deepcopy(coverage),
            "first_failure": None,
        },
        "property": {
            "status": "passed",
            "checker": cell["property_checker"]["identity"],
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
        },
        "samples": samples,
        "attempt_ledger": artifact,
        "widths": {
            "status": "complete",
            "validated_prefix": copy.deepcopy(extent),
            "coordinate_order": ["x"],
            "coordinate_units": ["arbitrary"],
            "aggregation_semantics": "union_and_per_partition",
            "endpoint": {
                **copy.deepcopy(width),
                "domain": {"kind": extent_kind, "start": 1, "end": 1},
            },
            "last_segment_tube": {
                **copy.deepcopy(width),
                "domain": {
                    "kind": extent_kind,
                    "start": last_segment_start,
                    "end": 1,
                },
            },
            "full_horizon_tube": copy.deepcopy(width),
            "series": [
                {
                    "extent": {"kind": extent_kind, "value": point},
                    "per_coordinate": copy.deepcopy(width["per_coordinate"]),
                }
                for point in (0, 1)
            ],
            "trajectory_artifact": artifact,
        },
        "artifacts": [
            {"role": role, "path": "evidence.txt", "sha256": evidence_sha}
            for role in ("command", "run_log", "result")
        ],
    }


def bind_completed_result(
    tmp_path, matrix, cell, instance_id, result, method="pytorch_gpu"
):
    partitions = [{"x": [0.0, 1.0]}]

    def bind_ledger(coverage, scope, invocation_id, suffix):
        if coverage["ledger_artifact"] is None:
            return
        outcomes = (
            ["completed"] * coverage["completed_partitions"]
            + ["failed"] * coverage["failed_partitions"]
            + ["unattempted"] * coverage["unattempted_partitions"]
        )
        ledger = {
            "schema_version": "archcomp26-partition-ledger-v1",
            "instance_id": instance_id,
            "scope": scope,
            "invocation_id": invocation_id,
            "boxes_sha256": coverage["boxes_sha256"],
            "entries": [
                {
                    "index": index,
                    "box_sha256": canonical_sha256(partition),
                    "outcome": outcomes[index],
                }
                for index, partition in enumerate(partitions)
            ],
        }
        ledger_path = Path("partition-ledgers") / (
            f"{instance_id}-{method}-{suffix}.json"
        )
        coverage["ledger_artifact"] = {
            "path": ledger_path.as_posix(),
            "sha256": write_json(tmp_path / ledger_path, ledger),
        }

    bind_ledger(
        result["run"]["partition_coverage"], "run_summary", None, "run"
    )
    for sample in result["samples"]:
        bind_ledger(
            sample["partition_coverage"], "attempt",
            sample["process_identity"]["invocation_id"],
            f"{sample['role']}-{sample['index']}-{sample['attempt_index']}",
        )
    previous_hash = None
    entries = []
    for sequence, sample in enumerate(result["samples"]):
        entry = {
            "sequence": sequence,
            "previous_entry_sha256": previous_hash,
            "role": sample["role"],
            "index": sample["index"],
            "attempt_index": sample["attempt_index"],
            "included_in_timing": sample["included_in_timing"],
            "invocation_id": sample["process_identity"]["invocation_id"],
            "outcome": sample["outcome"],
            "artifact": copy.deepcopy(sample["artifact"]),
        }
        entries.append(entry)
        previous_hash = canonical_sha256(entry)
    attempt_ledger = {
        "schema_version": "archcomp26-attempt-ledger-v1",
        "campaign_id": matrix["comparison_campaign"]["campaign_id"],
        "instance_id": instance_id,
        "method": method,
        "entries": entries,
        "head_sha256": previous_hash,
    }
    attempt_ledger_path = Path("attempt-ledgers") / f"{instance_id}-{method}.json"
    result["attempt_ledger"] = {
        "path": attempt_ledger_path.as_posix(),
        "sha256": write_json(tmp_path / attempt_ledger_path, attempt_ledger),
    }
    result_path = Path("results") / f"{instance_id}-{method}.json"
    cell["result_record"] = {
        "schema_version": "archcomp26-cell-result-v4",
        "path": result_path.as_posix(),
        "sha256": write_json(tmp_path / result_path, result),
    }
    matrix["cells"][instance_id][method] = cell
    matrix["status"] = "in_progress"
    return result_path


def bind_active_run_receipt(tmp_path, matrix, instance_id, method="pytorch_gpu"):
    invocation_id = f"{instance_id}:{method}:active"
    started_at = "2026-01-01T00:00:00Z"
    lock = {
        "schema_version": "archcomp26-campaign-lock-v1",
        "campaign_id": matrix["comparison_campaign"]["campaign_id"],
        "host_identity": matrix["comparison_campaign"]["host_identity"],
        "instance_id": instance_id,
        "method": method,
        "invocation_id": invocation_id,
        "acquired_at_utc": "2025-12-31T23:59:59Z",
        "lock_path": matrix["comparison_campaign"]["launch_guard"]["lock_path"],
        "protocol": matrix["comparison_campaign"]["launch_guard"]["protocol"],
        "wrapper_pid": 4241,
    }
    lock_path = Path("campaign/active.lock.json")
    lock_link = {
        "path": lock_path.as_posix(),
        "sha256": write_json(tmp_path / lock_path, lock),
    }
    receipt = {
        "schema_version": "archcomp26-active-run-v1",
        "campaign_id": matrix["comparison_campaign"]["campaign_id"],
        "instance_id": instance_id,
        "method": method,
        "role": "cold",
        "index": 0,
        "attempt_index": 0,
        "invocation_id": invocation_id,
        "pid": 4242,
        "started_at_utc": started_at,
        "prelaunch_audit_sha256": matrix["comparison_campaign"][
            "prelaunch_audit"
        ]["sha256"],
        "lock_artifact": lock_link,
    }
    receipt_path = Path("campaign/active-run.json")
    matrix["comparison_campaign"]["active_run_receipt"] = {
        "path": receipt_path.as_posix(),
        "sha256": write_json(tmp_path / receipt_path, receipt),
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


def test_v4_default_shape_cannot_be_weakened_or_crash_validation():
    manifest, matrix = inputs()
    broken = copy.deepcopy(matrix)
    broken["required_cell_fields"].remove("numerics")
    broken["cell_defaults"].pop("numerics")
    errors = validate_matrix(manifest, broken)
    assert "required_cell_fields do not match the v5 cell contract" in errors

    broken = copy.deepcopy(matrix)
    broken["cell_defaults"]["numerics"]["integration"].pop("step_size")
    errors = validate_matrix(manifest, broken)
    assert any(
        "cell_defaults.numerics.integration: incomplete override" in error
        for error in errors
    )


def test_non_string_applicability_tags_fail_closed_without_exceptions():
    manifest, matrix = inputs()
    broken = copy.deepcopy(matrix)
    numerics = copy.deepcopy(matrix["cell_defaults"]["numerics"])
    numerics["integration"]["step_size"]["mode"] = []
    broken["cells"]["quad-reach"]["pytorch_gpu"] = {"numerics": numerics}
    errors = validate_matrix(manifest, broken)
    assert any("invalid applicability tag" in error for error in errors)
    reasons = preflight_reasons(
        manifest, broken, "quad-reach", "pytorch_gpu"
    )
    assert any("invalid applicability tag" in reason for reason in reasons)


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


def test_fixed_instance_profile_rejects_continuous_discrete_switch(tmp_path):
    manifest, matrix = inputs()
    for instance_id, wrong_profile in (
        ("airplane-discrete", "full_execution_contract_v1"),
        ("acc-safe-distance", "discrete_execution_contract_v1"),
    ):
        broken = copy.deepcopy(manifest)
        row = next(row for row in broken["instances"] if row["id"] == instance_id)
        row["contract"]["unresolved_field_profile"] = wrong_profile
        errors = validate_matrix(broken, matrix)
        assert any(
            "does not match the fixed instance profile" in error for error in errors
        )

    resolved_manifest, resolved_matrix, _, _, _ = synthetic_inputs(
        tmp_path, "airplane-discrete", "full_execution_contract_v1"
    )
    resolved_manifest["execution_policy"]["experiments_paused"] = False
    errors = validate_matrix(resolved_manifest, resolved_matrix, root=tmp_path)
    assert any("does not match the fixed instance profile" in error for error in errors)
    assert "contract_invalid" in preflight_reasons(
        resolved_manifest,
        resolved_matrix,
        "airplane-discrete",
        "pytorch_gpu",
        root=tmp_path,
        now_utc=SYNTHETIC_LAUNCH_NOW,
    )


def test_missing_contract_record_is_the_only_launch_gate_in_valid_fixture(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    assert preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path,
        now_utc=SYNTHETIC_LAUNCH_NOW,
    ) == []
    row = next(row for row in manifest["instances"] if row["id"] == "acc-safe-distance")
    row["contract"].pop("record")
    reasons = preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path,
        now_utc=SYNTHETIC_LAUNCH_NOW,
    )
    assert "contract_invalid" in reasons
    assert "experiments_paused" not in reasons
    assert "contract_unresolved" not in reasons
    assert not any(reason.endswith("_missing") for reason in reasons)


def test_prelaunch_audit_receipt_is_required_before_resume(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    matrix["comparison_campaign"]["prelaunch_audit"] = {
        "path": None, "sha256": None,
    }
    reasons = preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path,
        now_utc=SYNTHETIC_LAUNCH_NOW,
    )
    assert "comparison_campaign_prelaunch_audit_missing" in reasons


def test_launch_stays_fail_closed_without_atomic_wrapper(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    matrix["comparison_campaign"]["launch_guard"]["wrapper_sha256"] = None
    reasons = preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path,
        now_utc=SYNTHETIC_LAUNCH_NOW,
    )
    assert "comparison_campaign_atomic_launcher_unavailable" in reasons


def test_prelaunch_audit_binds_readable_evidence_and_precedes_first_run(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    terminal_path = tmp_path / "campaign/original-terminal.json"
    original_terminal = terminal_path.read_bytes()
    terminal_path.write_bytes(original_terminal + b" ")
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "terminal_evidence_artifact" in error and "SHA-256" in error
        for error in errors
    )
    terminal_path.write_bytes(original_terminal)

    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    audit_link = matrix["comparison_campaign"]["prelaunch_audit"]
    audit_path = tmp_path / audit_link["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["checked_at_utc"] = "2026-01-01T00:00:00Z"
    audit_link["sha256"] = write_json(audit_path, audit)
    result["contract_identity"]["campaign_sha256"] = canonical_sha256(
        matrix["comparison_campaign"]
    )
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("audit must precede the first formal launch" in error for error in errors)


def test_launch_preflight_rejects_expired_or_semantically_false_audit(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    expired = datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
    reasons = preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path,
        now_utc=expired,
    )
    assert "comparison_campaign_prelaunch_audit_stale_or_invalid" in reasons

    audit_link = matrix["comparison_campaign"]["prelaunch_audit"]
    audit_path = tmp_path / audit_link["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    terminal_link = audit["original_native_quad"]["terminal_evidence_artifact"]
    terminal_path = tmp_path / terminal_link["path"]
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["terminal_status"] = "completed"
    terminal_link["sha256"] = write_json(terminal_path, terminal)
    audit_link["sha256"] = write_json(audit_path, audit)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("terminal identity mismatch" in error for error in errors)

    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    audit_link = matrix["comparison_campaign"]["prelaunch_audit"]
    audit_path = tmp_path / audit_link["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    terminal_link = audit["original_native_quad"]["terminal_evidence_artifact"]
    terminal_path = tmp_path / terminal_link["path"]
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    audit["original_native_quad"]["terminal_status"] = "completed"
    terminal["terminal_status"] = "completed"
    terminal_link["sha256"] = write_json(terminal_path, terminal)
    audit_link["sha256"] = write_json(audit_path, audit)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("independently frozen job" in error for error in errors)

    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    audit_link = matrix["comparison_campaign"]["prelaunch_audit"]
    audit_path = tmp_path / audit_link["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    terminal_link = audit["original_native_quad"]["terminal_evidence_artifact"]
    terminal_path = tmp_path / terminal_link["path"]
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    result_path = tmp_path / terminal["result_artifact"]["path"]
    result_path.write_bytes(b'{"rewritten":"terminal result"}\n')
    terminal["result_artifact"]["sha256"] = hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    terminal_link["sha256"] = write_json(terminal_path, terminal)
    audit_link["sha256"] = write_json(audit_path, audit)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("result identity" in error for error in errors)


def test_prelaunch_process_scan_and_run_directory_are_fail_closed(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    audit_link = matrix["comparison_campaign"]["prelaunch_audit"]
    audit_path = tmp_path / audit_link["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    scan_link = audit["process_scan_artifact"]
    scan_path = tmp_path / scan_link["path"]
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    scan["active_owned_processes"] = [{"pid": 42}]
    scan_link["sha256"] = write_json(scan_path, scan)
    audit_link["sha256"] = write_json(audit_path, audit)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("owned benchmark process is still active" in error for error in errors)

    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    audit_link = matrix["comparison_campaign"]["prelaunch_audit"]
    audit_path = tmp_path / audit_link["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["original_native_quad"]["run_directory"] = (
        "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/../escape/"
        "full1000_v1"
    )
    audit_link["sha256"] = write_json(audit_path, audit)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("run directory is outside root" in error for error in errors)

    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    audit_link = matrix["comparison_campaign"]["prelaunch_audit"]
    audit_path = tmp_path / audit_link["path"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    alternate = (
        "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/"
        "runs/not-the-original/full1000_v1"
    )
    audit["original_native_quad"]["run_directory"] = alternate
    audit["original_native_quad"]["watch_directory"] = alternate + "_watch"
    audit_link["sha256"] = write_json(audit_path, audit)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("independently frozen job" in error for error in errors)


def test_production_scope_rejects_synchronized_instance_removal(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    removed = "unicycle-reach"
    manifest["instances"] = [
        row for row in manifest["instances"] if row["id"] != removed
    ]
    manifest["scope"]["expected_instance_count"] = 15
    matrix["cells"].pop(removed)
    schedule_link = matrix["comparison_campaign"]["rotation"]["schedule_artifact"]
    schedule_path = tmp_path / schedule_link["path"]
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["entries"] = [
        entry for entry in schedule["entries"]
        if entry["instance_id"] != removed
    ]
    schedule_link["sha256"] = write_json(schedule_path, schedule)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert (
        "manifest instances do not match the frozen 16-instance ARCH-COMP scope"
        in errors
    )


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


def test_resolved_contract_rejects_duplicate_method_field_ownership(tmp_path):
    manifest, matrix, record, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    row = next(row for row in manifest["instances"] if row["id"] == "acc-safe-distance")
    record["fields"]["integration"]["step_size"] = 999.0
    record_path = tmp_path / row["contract"]["record"]["path"]
    row["contract"]["record"]["sha256"] = write_json(record_path, record)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "method-specific field integration.step_size must live in the execution cell"
        in error for error in errors
    )


def test_fractional_remainder_cell_values_and_invalid_boundaries(tmp_path):
    manifest, matrix, _, cell, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    assert cell["numerics"]["remainder"] == {
        "cutoff": {"mode": "configured", "value": 1e-6},
        "cap": {"mode": "configured", "value": 0.1},
        "symbolic_queue": {"mode": "configured", "value": 1},
        "semantics": "synthetic remainder and symbolic-remainder policy",
    }
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    cell["numerics"]["remainder"]["cutoff"]["value"] = 0.0
    cell["numerics"]["remainder"]["cap"]["value"] = 0.0
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    for dotted, invalid in (
        ("cutoff", True),
        ("cutoff", -1e-6),
        ("cutoff", float("nan")),
        ("cap", True),
        ("cap", -0.1),
        ("cap", float("inf")),
        ("symbolic_queue", 1.5),
        ("symbolic_queue", True),
        ("symbolic_queue", -1),
    ):
        broken = copy.deepcopy(matrix)
        broken["cells"]["acc-safe-distance"]["pytorch_gpu"]["numerics"][
            "remainder"
        ][dotted]["value"] = invalid
        errors = validate_matrix(manifest, broken, root=tmp_path)
        assert any(f"remainder.{dotted}.value" in error for error in errors), (
            dotted, invalid,
        )


def test_synthetic_continuous_contract_and_cell_complete_full_profile(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    assert preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path,
        now_utc=SYNTHETIC_LAUNCH_NOW,
    ) == []


def test_synthetic_discrete_contract_and_cell_complete_full_profile(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "airplane-discrete", "discrete_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    assert preflight_reasons(
        manifest, matrix, "airplane-discrete", "pytorch_gpu", root=tmp_path,
        now_utc=SYNTHETIC_LAUNCH_NOW,
    ) == []


def test_discrete_numerics_are_explicit_na_and_schedule_matches_contract(tmp_path):
    manifest, matrix, _, cell, _ = synthetic_inputs(
        tmp_path, "airplane-discrete", "discrete_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    assert preflight_reasons(
        manifest, matrix, "airplane-discrete", "pytorch_gpu", root=tmp_path,
        now_utc=SYNTHETIC_LAUNCH_NOW,
    ) == []

    cell["numerics"]["integration"]["step_size"]["mode"] = "unresolved"
    reasons = preflight_reasons(
        manifest, matrix, "airplane-discrete", "pytorch_gpu", root=tmp_path
    )
    assert "numerics_step_size_not_marked_not_applicable" in reasons
    cell["numerics"]["integration"]["step_size"]["mode"] = "not_applicable"

    cell["controller_execution"]["scheduled_updates"] = 2
    reasons = preflight_reasons(
        manifest, matrix, "airplane-discrete", "pytorch_gpu", root=tmp_path
    )
    assert "controller_scheduled_updates_contract_mismatch" in reasons


def test_tagged_na_cannot_carry_a_hidden_value(tmp_path):
    manifest, matrix, _, cell, _ = synthetic_inputs(
        tmp_path, "airplane-discrete", "discrete_execution_contract_v1"
    )
    cell["numerics"]["remainder"]["cap"]["value"] = 0.1
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "numerics.remainder.cap.value: must be null" in error
        for error in errors
    )


def test_adaptive_and_early_stop_modes_are_representable_but_fail_closed(tmp_path):
    manifest, matrix, _, cell, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    cell["numerics"]["integration"]["step_size"] = {
        "mode": "adaptive", "value": None,
    }
    cell["controller_execution"]["nn_calls"] = {
        "mode": "adaptive", "value": None,
    }
    cell["property_checker"]["early_stop_policy"] = "on_decisive"
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    reasons = preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path
    )
    assert {
        "adaptive_step_policy_not_executable",
        "adaptive_nn_call_policy_not_executable",
    } <= set(reasons)
    assert "property_early_stop_not_executable" not in reasons


def test_supported_nncs_cell_cannot_mark_nn_calls_or_checker_na(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    cell["controller_execution"]["nn_calls"] = {
        "mode": "not_applicable", "value": None,
    }
    cell["property_checker"] = copy.deepcopy(
        matrix["cell_defaults"]["property_checker"]
    )
    cell["property_checker"]["mode"] = "not_applicable"
    reasons = preflight_reasons(
        manifest, matrix, "acc-safe-distance", "pytorch_gpu", root=tmp_path
    )
    assert {
        "controller_nn_calls_not_applicable",
        "property_checker_not_applicable",
    } <= set(reasons)

    cell["property_checker"] = {
        "mode": "configured",
        "identity": "synthetic-checker",
        "semantics": "evaluate the shared property after the full run",
        "certificate_semantics": "synthetic explicit non-certificate",
        "early_stop_policy": "never",
    }
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    bind_completed_result(tmp_path, matrix, cell, "acc-safe-distance", result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "terminal cell lacks executable plan" in error
        and "controller_nn_calls_not_applicable" in error
        for error in errors
    )


@pytest.mark.parametrize(
    ("instance_id", "profile"),
    (
        ("acc-safe-distance", "full_execution_contract_v1"),
        ("airplane-discrete", "discrete_execution_contract_v1"),
    ),
)
def test_every_shared_contract_field_is_fail_closed(
    tmp_path, instance_id, profile
):
    manifest, matrix, record, _, _ = synthetic_inputs(
        tmp_path, instance_id, profile
    )
    row = next(row for row in manifest["instances"] if row["id"] == instance_id)
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


@pytest.mark.parametrize(
    ("profile", "instance_id", "dotted", "bad_value"),
    (
        (
            "full_execution_contract_v1",
            "acc-safe-distance",
            "controller_update.schedule_semantics",
            ["not text"],
        ),
        (
            "full_execution_contract_v1",
            "acc-safe-distance",
            "property.time_semantics",
            {"not": "text"},
        ),
        (
            "discrete_execution_contract_v1",
            "airplane-discrete",
            "property.step_semantics",
            ["not text"],
        ),
        (
            "discrete_execution_contract_v1",
            "airplane-discrete",
            "property.pass_condition",
            {"not": "text"},
        ),
    ),
)
def test_contract_semantics_fields_require_text(
    tmp_path, profile, instance_id, dotted, bad_value
):
    manifest, matrix, record, _, _ = synthetic_inputs(
        tmp_path, instance_id, profile
    )
    row = next(row for row in manifest["instances"] if row["id"] == instance_id)
    record_path = tmp_path / row["contract"]["record"]["path"]
    set_dotted(record["fields"], dotted, bad_value)
    row["contract"]["record"]["sha256"] = write_json(record_path, record)

    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(dotted in error and "non-empty text" in error for error in errors)


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
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    result_sha = cell["result_record"]["sha256"]
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    missing_sample = copy.deepcopy(result)
    missing_sample["samples"].pop()
    cell["result_record"]["sha256"] = write_json(
        tmp_path / result_path, missing_sample
    )
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("lacks exactly one included sample" in error for error in errors)

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


def test_long_run_shortfall_is_reportable_and_cannot_self_declare_ranking(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["measurement_plan"]["steady_runs"] = 2
    cell["measurement_plan"]["shortfall_reason"] = (
        "predeclared long-run wall-time budget"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    assert len(result["samples"]) == 3
    assert result["eligibility"]["performance_measurement_eligible"] is True
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    result["eligibility"]["cross_tool_ranking_eligible"] = True
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "result_record.eligibility: fields differ" in error
        and "cross_tool_ranking_eligible" in error
        for error in errors
    )


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


def test_two_methods_bind_distinct_numerical_cell_plans(tmp_path):
    manifest, matrix, _, base_cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cells = {
        "pytorch_gpu": copy.deepcopy(base_cell),
        "huan": copy.deepcopy(base_cell),
    }
    huan = cells["huan"]
    huan["numerics"]["integration"]["step_size"]["value"] = 0.5
    huan["numerics"]["integration"]["solution_order"]["value"] = 3
    huan["numerics"]["integration"]["point_order"]["value"] = 2
    huan["numerics"]["integration"]["validation_order"]["value"] = 4
    huan["numerics"]["remainder"]["symbolic_queue"]["value"] = 2
    huan["controller_execution"]["nn_calls"]["value"] = 2
    huan["controller_execution"]["nn_call_semantics"] = (
        "two bound evaluations across the shared update schedule"
    )
    huan["property_checker"]["identity"] = "synthetic-huan-checker"

    results = {}
    for method, cell in cells.items():
        cell["run"] = {
            "status": "completed", "failure_category": None,
            "failure_detail": None,
        }
        result = completed_result_record(
            manifest, cell, "acc-safe-distance", evidence_sha, method
        )
        bind_completed_result(
            tmp_path, matrix, cell, "acc-safe-distance", result, method
        )
        results[method] = result

    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    pytorch_identity = results["pytorch_gpu"]["contract_identity"]
    huan_identity = results["huan"]["contract_identity"]
    assert (
        pytorch_identity["instance_contract_sha256"]
        == huan_identity["instance_contract_sha256"]
    )
    assert (
        pytorch_identity["cell_plan_sha256"]
        != huan_identity["cell_plan_sha256"]
    )

    matrix["cells"]["acc-safe-distance"]["huan"]["numerics"][
        "integration"
    ]["step_size"]["value"] = 0.25
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("cell plan identity mismatch" in error for error in errors)

    matrix["cells"]["acc-safe-distance"]["huan"]["numerics"][
        "integration"
    ]["step_size"]["value"] = 0.5
    original_nn_semantics = matrix["cells"]["acc-safe-distance"]["huan"][
        "controller_execution"
    ]["nn_call_semantics"]
    matrix["cells"]["acc-safe-distance"]["huan"]["controller_execution"][
        "nn_call_semantics"
    ] = "tampered call semantics"
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("cell plan identity mismatch" in error for error in errors)
    matrix["cells"]["acc-safe-distance"]["huan"]["controller_execution"][
        "nn_call_semantics"
    ] = original_nn_semantics
    original_checker_semantics = matrix["cells"]["acc-safe-distance"]["huan"][
        "property_checker"
    ]["semantics"]
    matrix["cells"]["acc-safe-distance"]["huan"]["property_checker"][
        "semantics"
    ] = "tampered checker semantics"
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("cell plan identity mismatch" in error for error in errors)
    matrix["cells"]["acc-safe-distance"]["huan"]["property_checker"][
        "semantics"
    ] = original_checker_semantics
    huan_result = results["huan"]
    path = tmp_path / matrix["cells"]["acc-safe-distance"]["huan"][
        "result_record"
    ]["path"]
    huan_result["property"]["certificate_semantics"] = "tampered certificate"
    matrix["cells"]["acc-safe-distance"]["huan"]["result_record"][
        "sha256"
    ] = write_json(path, huan_result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("certificate semantics disagree" in error for error in errors)
    huan_result["property"]["certificate_semantics"] = (
        "synthetic explicit non-certificate"
    )
    huan_result["run"]["accepted_steps"] = 3
    for sample in huan_result["samples"]:
        sample["accepted_steps"] = 3
    matrix["cells"]["acc-safe-distance"]["huan"]["result_record"][
        "sha256"
    ] = write_json(path, huan_result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("accepted steps disagree" in error for error in errors)


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
    result["widths"]["validated_prefix"]["value"] = 0
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("completed prefix must be positive" in error for error in errors)

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
    result["property"]["status"] = "not_applicable"
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "completed run lacks explicit property/certificate" in error
        for error in errors
    )

    result = copy.deepcopy(original)
    result["property"]["status"] = "failed"
    result["property"]["certificate_status"] = "passed"
    result["eligibility"]["certificate_semantics_passed"] = True
    result["eligibility"]["numerical_soundness_class"] = (
        "independently outward replayed for exact benchmark workload"
    )
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    result = copy.deepcopy(original)
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
    for name in (
        "command", "source_identity", "binary_identity", "arithmetic",
        "numerics", "controller_execution", "property_checker", "runtime",
    ):
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
        "controller_updates": 0,
        "controller_update_trace": {
            "kind": "time_s", "points": [],
            "boundary_update_status": "not_executed", "artifact": None,
        },
        "nn_calls": 0,
        "partition_coverage": {
            "requested_partitions": 1,
            "completed_partitions": 0,
            "failed_partitions": 0,
            "unattempted_partitions": 1,
            "boxes_sha256": canonical_sha256([{"x": [0.0, 1.0]}]),
            "ledger_artifact": None,
        },
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
    }
    completed_sample = copy.deepcopy(result["samples"][0])
    result["samples"] = []
    unavailable = {
        "status": "unavailable",
        "domain": {"kind": "time_s", "start": 0, "end": 0},
        "per_coordinate": [],
        "artifact": None,
    }
    result["widths"] = {
        "status": "unavailable",
        "validated_prefix": {"kind": "time_s", "value": 0},
        "coordinate_order": ["x"],
        "coordinate_units": ["arbitrary"],
        "aggregation_semantics": "union_and_per_partition",
        "endpoint": copy.deepcopy(unavailable),
        "last_segment_tube": copy.deepcopy(unavailable),
        "full_horizon_tube": copy.deepcopy(unavailable),
        "series": [],
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
                "validated_prefix", {"kind": "steps", "value": 9999}
            ),
            "validated prefix disagrees with run",
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
        (
            lambda value: value.__setitem__("samples", [completed_sample]),
            "skipped run must have no samples",
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
        lambda result: result["eligibility"].__setitem__("soundness_scope", []),
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


def test_running_state_cannot_bypass_pause_contract_or_plan(tmp_path):
    manifest, matrix = inputs()
    matrix["cells"]["acc-safe-distance"]["pytorch_gpu"]["run"] = {
        "status": "running", "failure_category": None, "failure_detail": None,
    }
    matrix["status"] = "running"
    errors = validate_matrix(manifest, matrix)
    assert any("forbidden while experiments are paused" in error for error in errors)
    assert any("running cell has unresolved instance contract" in error for error in errors)
    assert any("active/terminal cell lacks executable plan" in error for error in errors)

    manifest, matrix, _, cell, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    manifest["execution_policy"]["experiments_paused"] = False
    cell["run"] = {
        "status": "running", "failure_category": None, "failure_detail": None,
    }
    matrix["cells"]["acc-safe-distance"]["pytorch_gpu"] = cell
    matrix["status"] = "running"
    errors = validate_matrix(
        manifest, matrix, root=tmp_path, now_utc=SYNTHETIC_LAUNCH_NOW
    )
    assert any("running cell lacks an active lock receipt" in error for error in errors)
    bind_active_run_receipt(tmp_path, matrix, "acc-safe-distance")
    assert validate_matrix(
        manifest, matrix, root=tmp_path, now_utc=SYNTHETIC_LAUNCH_NOW
    ) == []
    cell["runtime"]["timeout_s"] = 172800.0
    matrix["comparison_campaign"]["timeout_s"] = 172800.0
    after_audit_expiry = datetime(2026, 1, 2, 0, 0, 1, tzinfo=timezone.utc)
    assert validate_matrix(
        manifest, matrix, root=tmp_path, now_utc=after_audit_expiry
    ) == []
    cell["command"]["argv"] = None
    errors = validate_matrix(
        manifest, matrix, root=tmp_path, now_utc=SYNTHETIC_LAUNCH_NOW
    )
    assert any(
        "active/terminal cell lacks executable plan" in error
        and "command_argv_missing" in error
        for error in errors
    )

    cell["command"]["argv"] = ["synthetic-run"]
    matrix["cells"]["acc-safe-distance"]["huan"] = copy.deepcopy(cell)
    errors = validate_matrix(
        manifest, matrix, root=tmp_path, now_utc=SYNTHETIC_LAUNCH_NOW
    )
    assert any("permits at most one running cell" in error for error in errors)


def test_code_fixed_shared_profile_cannot_be_weakened(tmp_path):
    manifest, matrix, record, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    row = next(row for row in manifest["instances"] if row["id"] == "acc-safe-distance")
    manifest["unresolved_field_profiles"]["full_execution_contract_v1"].remove(
        "dynamics.equations"
    )
    record["fields"]["dynamics"].pop("equations")
    record["evidence"][0]["supports"].remove("dynamics.equations")
    record_path = tmp_path / row["contract"]["record"]["path"]
    row["contract"]["record"]["sha256"] = write_json(record_path, record)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("code-fixed shared contract" in error for error in errors)
    assert any("dynamics.equations is missing" in error for error in errors)


def test_partition_boxes_have_fixed_variable_interval_shape(tmp_path):
    manifest, matrix, record, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    row = next(row for row in manifest["instances"] if row["id"] == "acc-safe-distance")
    record["fields"]["initial_set"]["partitions"] = [True]
    record["fields"]["initial_set"]["boxes_sha256"] = canonical_sha256([True])
    record_path = tmp_path / row["contract"]["record"]["path"]
    row["contract"]["record"]["sha256"] = write_json(record_path, record)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("must contain variable_order exactly" in error for error in errors)


def test_cell_arithmetic_and_runtime_types_are_fail_closed(tmp_path):
    manifest, matrix, _, _, _ = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    for group, name, value in (
        ("arithmetic", "mode", True),
        ("arithmetic", "controller_domain", ["bad"]),
        ("arithmetic", "relaxation", {"bad": 1}),
        ("arithmetic", "dtype", 64),
        ("arithmetic", "transport", False),
        ("runtime", "hardware", ["bad"]),
        ("runtime", "gpu", {"bad": 1}),
        ("runtime", "resource_limits", "unstructured"),
    ):
        broken = copy.deepcopy(matrix)
        broken["cells"]["acc-safe-distance"]["pytorch_gpu"][group][name] = value
        errors = validate_matrix(manifest, broken, root=tmp_path)
        assert any(f"{group}.{name}" in error for error in errors), (group, name)


def test_completed_result_binds_partition_coverage_and_controller_updates(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    original = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", original
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    result = copy.deepcopy(original)
    coverages = [result["run"]["partition_coverage"]] + [
        sample["partition_coverage"] for sample in result["samples"]
    ]
    for coverage in coverages:
        coverage["completed_partitions"] = 0
        coverage["unattempted_partitions"] = 1
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("does not include every partition" in error for error in errors)

    result = copy.deepcopy(original)
    result["run"]["controller_updates"] = 0
    for sample in result["samples"]:
        sample["controller_updates"] = 0
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("controller updates do not match contract" in error for error in errors)


def test_partition_ledger_recomputes_counts_and_box_identity(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    coverage = result["run"]["partition_coverage"]
    ledger_path = tmp_path / coverage["ledger_artifact"]["path"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["entries"][0]["outcome"] = "failed"
    coverage["ledger_artifact"]["sha256"] = write_json(ledger_path, ledger)
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("counts disagree with partition ledger" in error for error in errors)

    ledger["entries"][0]["outcome"] = "completed"
    ledger["entries"][0]["box_sha256"] = "0" * 64
    coverage["ledger_artifact"]["sha256"] = write_json(ledger_path, ledger)
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("box hash does not match contract" in error for error in errors)


def test_partial_controller_trace_covers_updates_before_validated_extent(tmp_path):
    artifact_path = tmp_path / "trace.json"
    artifact_path.write_text("{}\n", encoding="utf-8")
    artifact = {
        "path": "trace.json",
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }
    contract = {
        "profile": "full_execution_contract_v1",
        "fields": {"controller_update": {"schedule_points": [0.0, 0.5]}},
    }
    errors = _validate_controller_update_trace(
        {
            "kind": "time_s", "points": [],
            "boundary_update_status": "not_scheduled", "artifact": artifact,
        },
        tmp_path, "trace", contract, observed_count=0,
        validated_extent={"kind": "time_s", "value": 0.75},
        outcome="timeout",
    )
    assert any("scheduled updates before/at" in error for error in errors)
    assert _validate_controller_update_trace(
        {
            "kind": "time_s", "points": [0.0, 0.5],
            "boundary_update_status": "executed", "artifact": artifact,
        },
        tmp_path, "trace", contract, observed_count=2,
        validated_extent={"kind": "time_s", "value": 0.5},
        outcome="timeout",
    ) == []


def test_partial_widths_require_measured_evidence(tmp_path):
    unavailable = {
        "status": "unavailable",
        "domain": {"kind": "time_s", "start": 0, "end": 0},
        "per_coordinate": [],
        "artifact": None,
    }
    widths = {
        "status": "partial",
        "validated_prefix": {"kind": "time_s", "value": 0},
        "coordinate_order": ["x"],
        "coordinate_units": ["m"],
        "aggregation_semantics": "union_and_per_partition",
        "endpoint": copy.deepcopy(unavailable),
        "last_segment_tube": copy.deepcopy(unavailable),
        "full_horizon_tube": copy.deepcopy(unavailable),
        "series": [],
        "trajectory_artifact": None,
    }
    errors = _validate_widths(widths, tmp_path, "widths")
    assert "widths: partial widths contain no measured evidence" in errors


def test_actual_receipts_must_follow_frozen_rotation(tmp_path):
    manifest, matrix, _, template_cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    results = {}
    for method in manifest["methods"]:
        cell = copy.deepcopy(template_cell)
        cell["run"] = {
            "status": "completed", "failure_category": None,
            "failure_detail": None,
        }
        result = completed_result_record(
            manifest, cell, "acc-safe-distance", evidence_sha, method=method
        )
        result_path = bind_completed_result(
            tmp_path, matrix, cell, "acc-safe-distance", result, method=method
        )
        results[method] = (cell, result, result_path)
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    for method, (cell, result, result_path) in results.items():
        for sample in result["samples"]:
            role = sample["role"]
            index = sample["index"]
            position = sample["campaign"]["sequence_position"]
            started_second = (
                (0 if role == "cold" else index + 1) * 10
                + 2 * (3 - position)
            )
            sample["process_identity"]["started_at_utc"] = (
                f"2026-01-01T00:00:{started_second:02d}Z"
            )
            sample["process_identity"]["finished_at_utc"] = (
                f"2026-01-01T00:00:{started_second + 1:02d}Z"
            )
        cell["result_record"]["sha256"] = write_json(
            tmp_path / result_path, result
        )
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "actual process receipts overlap or violate sequence" in error
        for error in errors
    )

    for method, (cell, result, result_path) in results.items():
        for sample in result["samples"]:
            role = sample["role"]
            index = sample["index"]
            position = sample["campaign"]["sequence_position"]
            round_base = 10 if role == "cold" else (0 if index == 0 else (index + 1) * 10)
            started_second = round_base + 2 * position
            sample["process_identity"]["started_at_utc"] = (
                f"2026-01-01T00:00:{started_second:02d}Z"
            )
            sample["process_identity"]["finished_at_utc"] = (
                f"2026-01-01T00:00:{started_second + 1:02d}Z"
            )
        cell["result_record"]["sha256"] = write_json(
            tmp_path / result_path, result
        )
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("cold/steady rounds overlap or run out of order" in error for error in errors)


def test_typed_plot_artifact_requires_parseable_complete_media(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    fake_plot = tmp_path / "fake.png"
    fake_plot.write_bytes(b"\x89PNG\r\n\x1a\n")
    result["artifacts"].append({
        "role": "plot_png", "path": "fake.png",
        "sha256": hashlib.sha256(fake_plot.read_bytes()).hexdigest(),
    })
    def png_chunk(name, data):
        return (
            struct.pack(">I", len(data)) + name + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    undecodable_png = tmp_path / "undecodable.png"
    undecodable_png.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
        + png_chunk(b"IDAT", b"not-a-zlib-stream")
        + png_chunk(b"IEND", b"")
    )
    result["artifacts"].append({
        "role": "plot_png", "path": "undecodable.png",
        "sha256": hashlib.sha256(undecodable_png.read_bytes()).hexdigest(),
    })
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(
        b"%PDF-1.7\n1 0 obj << /Type /Page >> endobj\n"
        b"startxref\n0\n%%EOF\n"
    )
    result["artifacts"].append({
        "role": "plot_pdf", "path": "fake.pdf",
        "sha256": hashlib.sha256(fake_pdf.read_bytes()).hexdigest(),
    })
    fake_svg = tmp_path / "fake.svg"
    fake_svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    result["artifacts"].append({
        "role": "plot_svg", "path": "fake.svg",
        "sha256": hashlib.sha256(fake_svg.read_bytes()).hexdigest(),
    })
    bind_completed_result(tmp_path, matrix, cell, "acc-safe-distance", result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("plot_png is not a complete PNG image" in error for error in errors)
    assert any("plot_png has an undecodable PNG IDAT stream" in error for error in errors)
    assert any("plot_pdf is not a complete page-bearing PDF" in error for error in errors)
    assert any("plot_svg is not a parseable SVG with graphics" in error for error in errors)


def test_typed_plot_artifact_accepts_decoded_png_and_pdf(tmp_path):
    from pypdf import PdfWriter

    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )

    def png_chunk(name, data):
        return (
            struct.pack(">I", len(data)) + name + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    png_path = tmp_path / "valid.png"
    png_path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(b"\x00\x00"))
        + png_chunk(b"IEND", b"")
    )
    pdf_path = tmp_path / "valid.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)
    for role, path in (("plot_png", png_path), ("plot_pdf", pdf_path)):
        result["artifacts"].append({
            "role": role,
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    bind_completed_result(tmp_path, matrix, cell, "acc-safe-distance", result)
    assert validate_matrix(manifest, matrix, root=tmp_path) == []


def test_discrete_update_trace_rejects_float_encoded_steps(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "airplane-discrete", "discrete_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    result = completed_result_record(
        manifest, cell, "airplane-discrete", evidence_sha
    )
    result["run"]["controller_update_trace"]["points"][0] = 0.0
    for sample in result["samples"]:
        sample["controller_update_trace"]["points"][0] = 0.0
    bind_completed_result(tmp_path, matrix, cell, "airplane-discrete", result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "discrete update points must be integers" in error for error in errors
    )


def test_formal_attempts_require_unique_process_receipts_and_schedule(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    original = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", original
    )
    result = copy.deepcopy(original)
    result["samples"][1]["process_identity"]["invocation_id"] = (
        result["samples"][0]["process_identity"]["invocation_id"]
    )
    result["samples"][1]["artifact"] = copy.deepcopy(
        result["samples"][0]["artifact"]
    )
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("duplicate invocation_id" in error for error in errors)
    assert any("formal samples reuse an artifact path" in error for error in errors)
    assert any("formal samples reuse artifact bytes" in error for error in errors)

    schedule_link = matrix["comparison_campaign"]["rotation"]["schedule_artifact"]
    schedule_path = tmp_path / schedule_link["path"]
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["entries"][0]["sequence_position"] = 3
    schedule_link["sha256"] = write_json(schedule_path, schedule)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("violates balanced rotation" in error for error in errors)


def test_terminal_failure_requires_matching_raw_attempt(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    detail = "synthetic deadline"
    cell["run"] = {
        "status": "timeout", "failure_category": "timeout",
        "failure_detail": detail,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    event = {
        "stage": "solver", "reason_code": "timeout", "step": 0,
        "time_s": 0.0, "detail": detail, "exit_code": None, "signal": None,
    }
    coverage = {
        "requested_partitions": 1, "completed_partitions": 0,
        "failed_partitions": 1, "unattempted_partitions": 0,
        "boxes_sha256": canonical_sha256([{"x": [0.0, 1.0]}]),
        "ledger_artifact": {"path": "evidence.txt", "sha256": evidence_sha},
    }
    result["run"].update({
        "status": "timeout", "outcome": "timeout",
        "validated_extent": {"kind": "time_s", "value": 0},
        "requested_horizon_completed": False, "accepted_steps": 0,
        "rejected_steps": 0, "controller_updates": 0, "nn_calls": 0,
        "controller_update_trace": {
            "kind": "time_s", "points": [],
            "boundary_update_status": "not_executed",
            "artifact": {"path": "evidence.txt", "sha256": evidence_sha},
        },
        "partition_coverage": coverage, "first_failure": event,
    })
    result["property"] = {
        "status": "not_checked", "checker": cell["property_checker"]["identity"],
        "certificate_status": "not_checked",
        "certificate_semantics": cell["property_checker"]["certificate_semantics"],
        "artifact": None,
    }
    result["eligibility"].update({
        "requested_horizon_completed": False,
        "certificate_semantics_passed": False,
        "formal_claim_eligible": False,
        "performance_measurement_eligible": False,
    })
    sample = copy.deepcopy(result["samples"][0])
    sample.update({
        "role": "diagnostic", "index": 0,
        "attempt_index": 0, "included_in_timing": False,
        "outcome": "timeout",
        "validated_extent": {"kind": "time_s", "value": 0},
        "accepted_steps": 0, "rejected_steps": 0,
        "controller_updates": 0, "nn_calls": 0,
        "controller_update_trace": {
            "kind": "time_s", "points": [],
            "boundary_update_status": "not_executed",
            "artifact": copy.deepcopy(sample["artifact"]),
        },
        "partition_coverage": copy.deepcopy(coverage),
        "failure": copy.deepcopy(event),
    })
    result["samples"] = [sample]
    unavailable = {
        "status": "unavailable",
        "domain": {"kind": "time_s", "start": 0, "end": 0},
        "per_coordinate": [], "artifact": None,
    }
    result["widths"] = {
        "status": "unavailable",
        "validated_prefix": {"kind": "time_s", "value": 0},
        "coordinate_order": ["x"], "coordinate_units": ["arbitrary"],
        "aggregation_semantics": "union_and_per_partition",
        "endpoint": copy.deepcopy(unavailable),
        "last_segment_tube": copy.deepcopy(unavailable),
        "full_horizon_tube": copy.deepcopy(unavailable),
        "series": [], "trajectory_artifact": None,
    }
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    result["samples"] = []
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("lacks a raw sample" in error for error in errors)


def test_property_early_stop_binds_policy_prefix_and_certificate(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    detail = "synthetic property became decisive"
    cell["property_checker"]["early_stop_policy"] = "on_pass"
    cell["run"] = {
        "status": "early_stopped", "failure_category": "property_early_stop",
        "failure_detail": detail,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    event = {
        "stage": "property", "reason_code": "property_early_stop_pass",
        "step": 0, "time_s": 0.0, "detail": detail,
        "exit_code": None, "signal": None,
    }
    extent = {"kind": "time_s", "value": 0}
    trace = {
        "kind": "time_s", "points": [],
        "boundary_update_status": "not_executed",
        "artifact": {"path": "evidence.txt", "sha256": evidence_sha},
    }
    result["run"].update({
        "status": "early_stopped", "outcome": "property_early_stop_pass",
        "validated_extent": copy.deepcopy(extent),
        "requested_horizon_completed": False,
        "accepted_steps": 0, "rejected_steps": 0,
        "controller_updates": 0, "controller_update_trace": copy.deepcopy(trace),
        "nn_calls": 0, "first_failure": copy.deepcopy(event),
    })
    result["property"].update({
        "status": "passed", "certificate_status": "passed",
        "artifact": {"path": "evidence.txt", "sha256": evidence_sha},
    })
    result["eligibility"].update({
        "requested_horizon_completed": False,
        "certificate_semantics_passed": True,
        "formal_claim_eligible": False,
        "performance_measurement_eligible": False,
    })
    sample = copy.deepcopy(result["samples"][0])
    sample.update({
        "role": "diagnostic", "index": 0, "attempt_index": 0,
        "included_in_timing": False, "outcome": "property_early_stop_pass",
        "validated_extent": copy.deepcopy(extent),
        "accepted_steps": 0, "rejected_steps": 0,
        "controller_updates": 0, "controller_update_trace": copy.deepcopy(trace),
        "nn_calls": 0, "failure": copy.deepcopy(event),
    })
    result["samples"] = [sample]
    unavailable = {
        "status": "unavailable",
        "domain": {"kind": "time_s", "start": 0, "end": 0},
        "per_coordinate": [], "artifact": None,
    }
    coordinate = {
        "coordinate": "x",
        "union": {"lo": 0.0, "hi": 1.0, "width": 1.0},
        "per_partition_width": {"mean": 1.0, "max": 1.0},
    }
    result["widths"] = {
        "status": "partial", "validated_prefix": copy.deepcopy(extent),
        "coordinate_order": ["x"], "coordinate_units": ["arbitrary"],
        "aggregation_semantics": "union_and_per_partition",
        "endpoint": copy.deepcopy(unavailable),
        "last_segment_tube": copy.deepcopy(unavailable),
        "full_horizon_tube": copy.deepcopy(unavailable),
        "series": [{
            "extent": copy.deepcopy(extent), "per_coordinate": [coordinate],
        }],
        "trajectory_artifact": {"path": "evidence.txt", "sha256": evidence_sha},
    }
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []
    baseline = copy.deepcopy(result)

    cell["property_checker"]["early_stop_policy"] = "never"
    result["contract_identity"]["cell_plan_sha256"] = canonical_sha256({
        name: cell[name] for name in (
            "support", "command", "source_identity", "binary_identity",
            "arithmetic", "numerics", "controller_execution", "property_checker",
            "runtime", "measurement_plan",
        )
    })
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("early-stop outcome is forbidden" in error for error in errors)

    cell["property_checker"]["early_stop_policy"] = "on_pass"
    result = copy.deepcopy(baseline)
    result["run"]["validated_extent"] = copy.deepcopy(
        result["run"]["requested_extent"]
    )
    result["samples"][0]["validated_extent"] = copy.deepcopy(
        result["run"]["requested_extent"]
    )
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("early stop must end before requested extent" in error for error in errors)

    result = copy.deepcopy(baseline)
    result["property"].update({
        "status": "not_checked", "certificate_status": "not_checked",
        "artifact": None,
    })
    result["eligibility"]["certificate_semantics_passed"] = False
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("early stop lacks matching property" in error for error in errors)


def test_attempt_ledger_retains_prior_failed_retry_and_detects_deletion(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    result = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )
    diagnostic_path = tmp_path / "attempts/prior-failure.txt"
    diagnostic_path.write_text("prior failed attempt\n", encoding="utf-8")
    diagnostic_artifact = {
        "path": "attempts/prior-failure.txt",
        "sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
    }
    event = {
        "stage": "environment", "reason_code": "environment_error",
        "step": 0, "time_s": 0.0, "detail": "prior environment failure",
        "exit_code": 1, "signal": None,
    }
    failed_retry = copy.deepcopy(result["samples"][0])
    result["samples"][0]["attempt_index"] = 1
    failed_retry.update({
        "role": "cold", "index": 0,
        "attempt_index": 0, "included_in_timing": False,
        "process_identity": {
            "invocation_id": "prior-failed-attempt", "pid": 999,
            "started_at_utc": "2025-12-31T23:59:59Z",
            "finished_at_utc": "2026-01-01T00:00:00Z",
        },
        "outcome": "environment_error",
        "validated_extent": {"kind": "time_s", "value": 0},
        "accepted_steps": 0, "rejected_steps": 0,
        "controller_updates": 0,
        "controller_update_trace": {
            "kind": "time_s", "points": [],
            "boundary_update_status": "not_executed",
            "artifact": copy.deepcopy(diagnostic_artifact),
        },
        "nn_calls": 0,
        "partition_coverage": {
            "requested_partitions": 1, "completed_partitions": 0,
            "failed_partitions": 1, "unattempted_partitions": 0,
            "boxes_sha256": canonical_sha256([{"x": [0.0, 1.0]}]),
            "ledger_artifact": {"path": "evidence.txt", "sha256": evidence_sha},
        },
        "failure": event,
        "artifact": diagnostic_artifact,
    })
    result["samples"].insert(0, failed_retry)
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    assert validate_matrix(manifest, matrix, root=tmp_path) == []

    result["samples"].pop(0)
    cell["result_record"]["sha256"] = write_json(tmp_path / result_path, result)
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "attempt_ledger: entries do not exactly match result samples" in error
        for error in errors
    )


def test_attempt_ledger_rejects_gaps_and_completed_sample_selection(tmp_path):
    manifest, matrix, _, cell, evidence_sha = synthetic_inputs(
        tmp_path, "acc-safe-distance", "full_execution_contract_v1"
    )
    cell["run"] = {
        "status": "completed", "failure_category": None, "failure_detail": None,
    }
    baseline = completed_result_record(
        manifest, cell, "acc-safe-distance", evidence_sha
    )

    result = copy.deepcopy(baseline)
    result["samples"][0]["attempt_index"] = 99
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any("attempt indices are not contiguous from zero" in error for error in errors)

    result = copy.deepcopy(baseline)
    first = result["samples"][0]
    first["included_in_timing"] = True
    slower = copy.deepcopy(first)
    slower_artifact_path = tmp_path / "attempts/pytorch_gpu-cold-0-retry.txt"
    slower_artifact_path.write_text("slower completed retry\n", encoding="utf-8")
    slower_artifact = {
        "path": "attempts/pytorch_gpu-cold-0-retry.txt",
        "sha256": hashlib.sha256(slower_artifact_path.read_bytes()).hexdigest(),
    }
    slower.update({
        "attempt_index": 1,
        "included_in_timing": False,
        "process_identity": {
            "invocation_id": "completed-retry-selection",
            "pid": 1999,
            "started_at_utc": "2026-01-01T00:00:02Z",
            "finished_at_utc": "2026-01-01T00:00:04Z",
        },
        "timing_s": {**slower["timing_s"], "process_total": 2.0},
        "artifact": slower_artifact,
    })
    slower["controller_update_trace"]["artifact"] = slower_artifact
    result["samples"].insert(1, slower)
    result_path = bind_completed_result(
        tmp_path, matrix, cell, "acc-safe-distance", result
    )
    errors = validate_matrix(manifest, matrix, root=tmp_path)
    assert any(
        "more than one completed formal attempt" in error
        or "attempts after its first completion" in error
        for error in errors
    )
