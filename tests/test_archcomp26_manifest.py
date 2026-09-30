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
        "schema_version": "archcomp26-execution-matrix-v2",
        "status": "all_cells_not_started",
    }
    contract_schema = manifest["instance_contract_record"]
    contract_schema_path = ROOT / contract_schema["path"]
    assert contract_schema["schema_version"] == "archcomp26-instance-contract-v1"
    assert hashlib.sha256(contract_schema_path.read_bytes()).hexdigest() == (
        contract_schema["sha256"]
    )

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
    assert matrix["cell_defaults"]["measurement_plan"] == {
        "cold_runs": 1,
        "steady_runs": 10,
        "fresh_process_per_run": True,
        "timing_boundary_version": "total_configuration_v1",
    }
    assert set(matrix["cell_defaults"]["result_record"].values()) == {None}
    result_schema = matrix["result_record_contract"]
    result_schema_path = ROOT / result_schema["path"]
    assert result_schema["schema_version"] == "archcomp26-cell-result-v1"
    assert hashlib.sha256(result_schema_path.read_bytes()).hexdigest() == (
        result_schema["sha256"]
    )
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


def test_plot_contract_status_accounts_for_every_instance_without_promotion():
    manifest = load("benchmarks/archcomp26/manifest.json")
    status = load("benchmarks/plot_specs/archcomp26_status.json")
    assert status["schema_version"] == "archcomp26-plot-contract-status-v1"
    assert status["experiments_started"] is False
    assert status["execution_contracts_resolved"] is False
    assert status["source_manifest"] == "benchmarks/archcomp26/manifest.json"
    assert status["source_contract_audits"] == manifest["contract_audits"]["path"]

    expected = {
        row["id"]: (row["benchmark"], row["contract_audit"])
        for row in manifest["instances"]
    }
    rows = status["instances"]
    assert len(rows) == len(expected) == 16
    assert {row["instance_id"] for row in rows} == set(expected)
    assert len({row["instance_id"] for row in rows}) == len(rows)
    for row in rows:
        assert (row["benchmark"], row["contract_audit"]) == expected[row["instance_id"]]
        assert row["status"] in {
            "materialized_v2_content_contract",
            "materialized_v3_content_contract",
            "materialized_v4_content_contract",
            "axis_aligned_content_ready_not_materialized",
            "blocked_fail_closed",
        }
        assert isinstance(row["content_blockers"], list)
        if row["status"] == "blocked_fail_closed":
            assert row["content_blockers"]
        else:
            assert row["content_blockers"] == []

    counts = {
        label: sum(row["status"] == label for row in rows)
        for label in (
            "materialized_v2_content_contract",
            "materialized_v3_content_contract",
            "materialized_v4_content_contract",
            "axis_aligned_content_ready_not_materialized",
            "blocked_fail_closed",
        )
    }
    assert status["counts"] == {"instances": 16, **counts}
    materialized = [row for row in rows if row["plot_spec"] is not None]
    assert len(materialized) == 10
    for row in materialized:
        spec = load(row["plot_spec"])
        assert spec["instance_id"] == row["instance_id"]
        assert spec["benchmark"] == row["benchmark"]
        assert "execution contract" in spec["contract_status"]

    specs = {row["instance_id"]: load(row["plot_spec"]) for row in materialized}
    acc = specs["acc-safe-distance"]
    assert acc["schema"] == "torch-tm-flowpipe-plot-spec-v3"
    assert acc["horizon"]["end"] == 5.0
    assert acc["derived_coordinates"] == {
        "safe_distance_margin": {
            "kind": "affine",
            "offset": -10.0,
            "coefficients": {"x1": 1.0, "x4": -1.0, "x5": -1.4},
        }
    }
    assert acc["regions"][0]["constraint"] == {
        "kind": "threshold",
        "coordinate": "safe_distance_margin",
        "operator": ">=",
        "value": 0.0,
    }
    assert acc["regions"][0]["time"] == {"kind": "all"}

    docking = specs["docking-constraint"]
    assert docking["schema"] == "torch-tm-flowpipe-plot-spec-v4"
    assert docking["horizon"]["end"] == 40.0
    assert docking["units"] == {"t": "s"}
    assert docking["derived_coordinates"] == {
        "docking_safety_margin": {
            "kind": "radial_speed_margin",
            "offset": 0.2,
            "radial_gain": 0.002054,
            "position_coordinates": ["sx", "sy"],
            "velocity_coordinates": ["sx_dot", "sy_dot"],
        }
    }
    assert docking["regions"][0]["constraint"] == {
        "kind": "threshold",
        "coordinate": "docking_safety_margin",
        "operator": ">=",
        "value": 0.0,
    }
    assert docking["regions"][0]["time"] == {"kind": "all"}

    airplane = specs["airplane-continuous"]
    assert airplane["horizon"]["end"] == 2.0
    assert airplane["initial_set"]["bounds"]["x4"] == [0.0, 1.0]
    assert airplane["regions"][0]["bounds"] == {
        "x2": [-1.0, 1.0], "x7": [-1.0, 1.0],
        "x8": [-1.0, 1.0], "x9": [-1.0, 1.0],
    }

    less = specs["double-pendulum-less-robust"]
    more = specs["double-pendulum-more-robust"]
    assert less["horizon"]["end"] == 1.0
    assert more["horizon"]["end"] == 0.4
    assert set(tuple(value) for value in less["initial_set"]["bounds"].values()) == {
        (1.0, 1.3)
    }
    assert set(tuple(value) for value in less["regions"][0]["bounds"].values()) == {
        (-1.7, 2.0)
    }
    assert set(tuple(value) for value in more["regions"][0]["bounds"].values()) == {
        (-1.5, 1.5)
    }

    standard = specs["nav-standard"]
    robust = specs["nav-robust"]
    assert standard["horizon"]["end"] == robust["horizon"]["end"] == 6.0
    assert standard["property_quantifier"] == robust["property_quantifier"] == (
        "conjunction"
    )
    assert standard["initial_set"] == robust["initial_set"]
    assert standard["regions"] == robust["regions"]
    assert standard["source_refs"][2]["path"].endswith("nn-nav-point.onnx")
    assert robust["source_refs"][2]["path"].endswith("nn-nav-set.onnx")

    single = specs["single-pendulum-reach"]
    assert single["coordinate_names"] == ["x1", "x2"]
    assert single["initial_set"]["bounds"] == {
        "x1": [1.0, 1.175], "x2": [0.0, 0.2],
    }
    assert single["regions"][0]["time"] == {
        "kind": "interval", "lo": 0.5, "hi": 1.0,
    }

    tora = specs["tora-remain"]
    assert tora["horizon"]["end"] == 20.0
    assert tora["initial_set"]["bounds"] == {
        "x1": [0.6, 0.7], "x2": [-0.7, -0.6],
        "x3": [-0.4, -0.3], "x4": [0.5, 0.6],
    }
    assert set(tuple(value) for value in tora["regions"][0]["bounds"].values()) == {
        (-2.0, 2.0)
    }

    quad = specs["quad-reach"]
    assert quad["horizon"]["end"] == 5.0
    assert quad["regions"][0]["bounds"] == {"x3": [0.94, 1.06]}


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


def test_contract_audits_fail_closed():
    manifest = load("benchmarks/archcomp26/manifest.json")
    audit_link = manifest["contract_audits"]
    audits = load(audit_link["path"])
    assert audits["schema_version"] == audit_link["schema_version"]
    assert audits["experiments_started"] is False
    assert audits["sources"]["report"]["sha256"] == manifest[
        "official_sources"
    ]["report"]["pdf_sha256"]
    assert audits["sources"]["benchmark_repository"]["commit"] == manifest[
        "official_sources"
    ]["benchmark_repository"]["commit"]
    historical = audits["sources"]["historical_quad_contract"]
    assert historical["sha256"] == hashlib.sha256(
        (ROOT / historical["path"]).read_bytes()
    ).hexdigest()
    assets = load("benchmarks/archcomp26/official_assets.json")
    assets_by_id = {row["id"]: row for row in assets["instances"]}

    by_id = {row["id"]: row for row in manifest["instances"]}
    assert all("contract_audit" in row for row in manifest["instances"])
    assert {row["contract_audit"] for row in manifest["instances"]} == set(
        audits["audits"]
    )
    airplane = audits["audits"][by_id["airplane-continuous"]["contract_audit"]]
    airplane_assets = assets_by_id["airplane-continuous"]
    assert airplane["source_files"] == {
        "specification": airplane_assets["specification"],
        "continuous_dynamics": airplane_assets["dynamics"],
        "controller": airplane_assets["controller_candidates"][0],
    }
    assert airplane["continuous"]["horizon_s"] == 2.0
    assert airplane["continuous"]["property_time_semantics"].startswith(
        "for_all_t"
    )
    assert airplane["discrete"]["index_set"] == {
        "start": 0, "end": 20, "inclusive": True,
    }
    assert airplane["discrete"]["transition_count"] == 20
    assert airplane["discrete"]["report_transition_rule"] == {
        "source": "report printed page 89",
        "method": "forward_euler",
        "state_update": "x[k+1] = x[k] + f(x[k]) * delta_t",
        "delta_t_s": 0.1,
        "evidence_scope": (
            "mathematical rule only; not a selected executable transition "
            "implementation"
        ),
    }
    assert airplane["discrete"]["transition_source"] is None
    assert airplane["discrete"]["control_application_order"] is None
    assert airplane["discrete"]["status"] == "unresolved"
    assert by_id["airplane-discrete"]["contract"]["status"] == "unresolved"

    linked = {
        "acc-safe-distance": "acc",
        "attitude-control-avoid": "attitude_control",
        "balancing-reach": "balancing",
        "docking-constraint": "docking",
        "double-pendulum-less-robust": "double_pendulum",
        "double-pendulum-more-robust": "double_pendulum",
        "nav-standard": "navigation",
        "nav-robust": "navigation",
        "single-pendulum-reach": "single_pendulum",
        "tora-remain": "tora",
        "tora-reach-sigmoid": "tora",
        "tora-reach-tanh": "tora",
        "unicycle-reach": "unicycle",
    }
    for instance_id, audit_name in linked.items():
        assert by_id[instance_id]["contract_audit"] == audit_name
        assert by_id[instance_id]["contract"]["status"] == "unresolved"

    acc = audits["audits"]["acc"]
    acc_assets = assets_by_id["acc-safe-distance"]
    assert acc["source_files"] == {
        "specification": acc_assets["specification"],
        "dynamics": acc_assets["dynamics"],
        "controller": acc_assets["controller_candidates"][0],
    }
    assert acc["shared_contract"]["controller_updates"] == 50
    assert acc["shared_contract"]["property"]["kind"] == "linear_halfspace"
    assert acc["unresolved_fields"]["v_rel_sign_definition"] is None

    attitude = audits["audits"]["attitude_control"]
    attitude_assets = assets_by_id["attitude-control-avoid"]
    assert attitude["source_files"] == {
        "specification": attitude_assets["specification"],
        "dynamics": attitude_assets["dynamics"],
        "controller_candidates": attitude_assets["controller_candidates"],
    }
    assert len(attitude["source_files"]["controller_candidates"]) == 2
    assert "does not hold" in attitude["source_conflicts"]["property_polarity"]

    balancing = audits["audits"]["balancing"]
    balancing_assets = assets_by_id["balancing-reach"]
    assert balancing["source_files"] == {
        "specification": balancing_assets["specification"],
        "dynamics": balancing_assets["dynamics"],
        "controller": balancing_assets["controller_candidates"][0],
    }
    assert balancing["shared_contract"]["controller_updates"] == 500
    assert "five-feature" in balancing["source_conflicts"]["controller_input"]
    assert "[8,10]" in balancing["source_conflicts"]["property_time"]

    docking = audits["audits"]["docking"]
    docking_assets = assets_by_id["docking-constraint"]
    assert docking["source_files"] == {
        "specification": docking_assets["specification"],
        "dynamics": docking_assets["dynamics"],
        "controller": docking_assets["controller_candidates"][0],
    }
    assert docking["shared_contract"]["horizon_s"] == 40.0
    assert docking["shared_contract"]["property"]["kind"] == (
        "nonlinear_coupled_inequality"
    )

    double_pendulum = audits["audits"]["double_pendulum"]
    less_assets = assets_by_id["double-pendulum-less-robust"]
    more_assets = assets_by_id["double-pendulum-more-robust"]
    assert double_pendulum["source_files"] == {
        "specification": less_assets["specification"],
        "dynamics": less_assets["dynamics"],
        "controllers": {
            "less_robust": less_assets["controller_candidates"][0],
            "more_robust": more_assets["controller_candidates"][0],
        },
    }
    assert double_pendulum["source_files"]["controllers"][
        "less_robust"
    ]["sha256"] != double_pendulum["source_files"]["controllers"][
        "more_robust"
    ]["sha256"]
    legacy = double_pendulum["legacy_more_robust_conflict"]
    assert legacy["controller_sha256_used"] == less_assets[
        "controller_candidates"
    ][0]["sha256"]
    assert legacy["initial_set_used"] == [[1.3, 1.3]] * 4
    for identity in (legacy["config"], legacy["boxes"]):
        assert identity["sha256"] == hashlib.sha256(
            (ROOT / identity["path"]).read_bytes()
        ).hexdigest()

    navigation = audits["audits"]["navigation"]
    nav_standard_assets = assets_by_id["nav-standard"]
    nav_robust_assets = assets_by_id["nav-robust"]
    assert navigation["source_files"]["specification"] == nav_standard_assets[
        "specification"
    ]
    assert navigation["source_files"]["repository_dynamics"] == (
        nav_standard_assets["dynamics"]
    )
    assert navigation["source_files"]["controllers"]["standard"] == (
        nav_standard_assets["controller_candidates"][0]
    )
    assert navigation["source_files"]["controllers"]["robust"] == (
        nav_robust_assets["controller_candidates"][0]
    )
    assert navigation["state_order_conflict"]["report"] == [
        "x", "y", "theta", "nu",
    ]
    assert navigation["state_order_conflict"][
        "repository_dynamics_effective_order"
    ] == ["x", "y", "nu", "theta"]
    assert navigation["state_order_conflict"]["report_control_derivatives"] != (
        navigation["state_order_conflict"][
            "repository_control_derivatives_under_effective_order"
        ]
    )
    assert navigation["state_order_conflict"]["resolution"].startswith(
        "unresolved"
    )

    pendulum = audits["audits"]["single_pendulum"]
    pendulum_assets = assets_by_id["single-pendulum-reach"]
    assert pendulum["source_files"] == {
        "specification": pendulum_assets["specification"],
        "repository_dynamics": pendulum_assets["dynamics"],
        "controller": pendulum_assets["controller_candidates"][0],
    }
    assert pendulum["repository_conflict"]["specification_state_count"] == 2
    assert pendulum["repository_conflict"]["dynamics_return_count"] == 3
    assert pendulum["repository_conflict"]["extra_derivative"] == "dx(3) = 1"
    assert pendulum["repository_conflict"]["resolution"].startswith(
        "unresolved"
    )

    tora = audits["audits"]["tora"]
    remain_assets = assets_by_id["tora-remain"]
    sigmoid_assets = assets_by_id["tora-reach-sigmoid"]
    tanh_assets = assets_by_id["tora-reach-tanh"]
    assert tora["source_files"]["remain"] == {
        "specification": remain_assets["specification"],
        "dynamics": remain_assets["dynamics"],
        "controller": remain_assets["controller_candidates"][0],
    }
    assert tora["source_files"]["reach"] == {
        "specification": sigmoid_assets["specification"],
        "dynamics": sigmoid_assets["dynamics"],
        "sigmoid_controller": sigmoid_assets["controller_candidates"][0],
        "relu_tanh_controller": tanh_assets["controller_candidates"][0],
    }
    assert tora["remain_controller_boundary"]["resolution"].startswith(
        "unresolved"
    )
    assert tora["reach_controllers"][
        "repository_sigmoid_mat_activations"
    ] == ["sigmoid"] * 4
    assert tora["reach_controllers"][
        "repository_relu_tanh_mat_activations"
    ] == ["relu", "relu", "relu", "tanh"]
    for source in tora["reach_controllers"][
        "activation_metadata_sources"
    ].values():
        assert source["path"].endswith(".mat")
        assert len(source["sha256"]) == 64
        int(source["sha256"], 16)
    assert tora["reach_property"]["checker_semantics_status"] == "unresolved"

    unicycle = audits["audits"]["unicycle"]
    unicycle_assets = assets_by_id["unicycle-reach"]
    assert unicycle["source_files"] == {
        "specification": unicycle_assets["specification"],
        "repository_dynamics": unicycle_assets["dynamics"],
        "controller": unicycle_assets["controller_candidates"][0],
    }
    assert unicycle["report_contract"]["disturbance_interval"] == [
        -0.0001, 0.0001,
    ]
    assert unicycle["repository_conflict"]["disturbance_implemented"] is False
    assert unicycle["property_semantics"]["shared_checker_status"] == "unresolved"

    quad = audits["audits"][by_id["quad-reach"]["contract_audit"]]
    quad_assets = assets_by_id["quad-reach"]
    assert quad["source_files"] == {
        "specification": quad_assets["specification"],
        "repository_dynamics": quad_assets["dynamics"],
        "controller_candidates": quad_assets["controller_candidates"],
    }
    assert quad["status"].endswith("selection_unresolved")
    assert {row["state"] for row in quad["differences"]} == {"x2", "x4", "x5"}
    report = quad["dynamics_variants"]["archcomp26_report"]
    author = quad["dynamics_variants"][
        "pinned_repository_and_historical_author"
    ]
    assert all(report[state] != author[state] for state in ("x2", "x4", "x5"))
    assert len(quad["source_files"]["controller_candidates"]) == 2
    assert "Do not label" in quad["execution_rule"]
    assert by_id["quad-reach"]["contract"]["status"] == "unresolved"
