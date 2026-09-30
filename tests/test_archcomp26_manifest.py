import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = (
    "acc-safe-distance",
    "airplane-continuous",
    "airplane-discrete",
    "attitude-control-avoid",
    "balancing-reach",
    "docking-constraint",
    "double-pendulum-less-robust",
    "double-pendulum-more-robust",
    "nav-standard",
    "nav-robust",
    "quad-reach",
    "single-pendulum-reach",
    "tora-remain",
    "tora-reach-sigmoid",
    "tora-reach-tanh",
    "unicycle-reach",
)
METHODS = ("pytorch_gpu", "huan", "xiangru", "flowstar_native")
RESUME_GATE = (
    "Explicit user authorization is required. Before any launch, read-only "
    "recheck the original native QUAD job's actual terminal state and do not "
    "duplicate it."
)


def load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_archcomp26_manifest_has_exact_scope_and_pause_gate():
    manifest = load("benchmarks/archcomp26/manifest.json")
    assert manifest["schema_version"] == "archcomp26-four-way-manifest-v1"
    assert manifest["execution_policy"] == {
        "experiments_paused": True,
        "run_status": "not_started",
        "resume_gate": RESUME_GATE,
    }
    assert tuple(manifest["methods"]) == METHODS
    assert manifest["execution_matrix"] == {
        "path": "benchmarks/archcomp26/execution_matrix.json",
        "schema_version": "archcomp26-execution-matrix-v1",
        "status": "all_cells_not_started",
    }

    instances = manifest["instances"]
    assert tuple(row["id"] for row in instances) == EXPECTED_IDS
    assert len(instances) == manifest["scope"]["expected_instance_count"] == 16
    assert all(row["benchmark"].casefold() != "vcas" for row in instances)
    assert all("vcas" not in row["id"].casefold() for row in instances)

    by_id = {row["id"]: row for row in instances}
    discrete = by_id["airplane-discrete"]["contract"]
    assert discrete["status"] == "unresolved"
    assert discrete["model_kind"] == "discrete_time"
    assert discrete["unresolved_field_profile"] == "discrete_execution_contract_v1"
    assert set(discrete["not_applicable"]) == {
        "integration.step_size", "integration.solution_order",
        "integration.validation_order", "integration.horizon",
        "remainder.cutoff", "remainder.cap", "remainder.symbolic_queue",
    }
    for instance_id, row in by_id.items():
        if instance_id != "airplane-discrete":
            assert row["contract"] == {
                "status": "unresolved",
                "unresolved_field_profile": "full_execution_contract_v1",
            }
    profiles = manifest["unresolved_field_profiles"]
    assert len(profiles["full_execution_contract_v1"]) == len(
        set(profiles["full_execution_contract_v1"])
    )
    assert len(profiles["discrete_execution_contract_v1"]) == len(
        set(profiles["discrete_execution_contract_v1"])
    )
    assert {"transition.state_update", "property.step_semantics",
            "discrete.transition_count"} <= set(
                profiles["discrete_execution_contract_v1"]
            )


def test_execution_matrix_explicitly_has_all_64_not_started_cells():
    manifest = load("benchmarks/archcomp26/manifest.json")
    matrix = load(manifest["execution_matrix"]["path"])
    assert matrix["schema_version"] == manifest["execution_matrix"]["schema_version"]
    assert matrix["source_manifest"] == "benchmarks/archcomp26/manifest.json"
    assert matrix["merge_semantics"] == "shallow_override_of_complete_cell_defaults"
    assert tuple(matrix["methods"]) == METHODS
    assert tuple(matrix["cells"]) == EXPECTED_IDS
    required = set(matrix["required_cell_fields"])
    assert set(matrix["cell_defaults"]) == required
    assert matrix["cell_defaults"]["run"]["status"] == "not_started"
    assert matrix["cell_defaults"]["widths"]["status"] == "not_measured"
    count = 0
    for methods in matrix["cells"].values():
        assert tuple(methods) == METHODS
        for override in methods.values():
            assert isinstance(override, dict)
            resolved = {**matrix["cell_defaults"], **override}
            assert set(resolved) == required
            assert resolved["run"]["status"] == "not_started"
            count += 1
    assert count == 64


def test_legacy_14_mapping_does_not_hide_new_instances():
    manifest = load("benchmarks/archcomp26/manifest.json")
    mapping = {row["id"]: row["legacy_candidate"]["config_id"]
               for row in manifest["instances"]}
    assert mapping["airplane-discrete"] is None
    assert mapping["docking-constraint"] is None
    assert sum(value is not None for value in mapping.values()) == 14
    quad = next(row for row in manifest["instances"] if row["id"] == "quad-reach")
    assert quad["legacy_candidate"]["status"] == "dynamics_mismatch_unresolved"
    assert any("report equations" in issue for issue in quad["known_issues"])


def test_official_asset_inventory_and_detached_audit_receipt_are_bound():
    manifest = load("benchmarks/archcomp26/manifest.json")
    inventory_path = ROOT / "benchmarks/archcomp26/official_assets.json"
    assets = json.loads(inventory_path.read_text(encoding="utf-8"))
    receipt = load("benchmarks/archcomp26/evidence/official_assets_audit_20261001.json")
    commit = manifest["official_sources"]["benchmark_repository"]["commit"]
    assert assets["schema_version"] == "archcomp26-official-assets-v1"
    assert assets["source_commit"] == commit
    assert receipt["remote_ref_observed_commit"] == commit
    assert receipt["detached_checkout_commit"] == commit
    assert receipt["inventory"]["sha256"] == hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()
    assert receipt["inventory"]["schema_version"] == assets["schema_version"]

    rows = assets["instances"]
    assert tuple(row["id"] for row in rows) == EXPECTED_IDS
    references = []
    for row in rows:
        model_asset = row.get("dynamics") or row.get("continuous_dynamics_candidate")
        assert model_asset is not None
        references.extend([row["specification"], model_asset,
                           *row["controller_candidates"]])
        for asset in [row["specification"], model_asset,
                      *row["controller_candidates"]]:
            assert asset["path"].startswith("benchmarks/")
            assert len(asset["sha256"]) == 64
            int(asset["sha256"], 16)
        assert row["controller_candidates"]
        assert isinstance(row["selection_status"], str)

    airplane_discrete = next(row for row in rows if row["id"] == "airplane-discrete")
    assert "dynamics" not in airplane_discrete
    assert "continuous_dynamics_candidate" in airplane_discrete
    assert airplane_discrete["model_selection_status"].startswith(
        "discrete_transition_source_unresolved"
    )
    verification = receipt["verification"]
    assert len(references) == verification["reference_count"] == 50
    assert len({asset["path"] for asset in references}) == verification[
        "unique_file_count"
    ] == 41
    assert verification["missing_count"] == verification["mismatch_count"] == 0
    assert verification["result"] == "passed"
    assert "do not fetch or re-hash" in receipt["scope_limit"]
