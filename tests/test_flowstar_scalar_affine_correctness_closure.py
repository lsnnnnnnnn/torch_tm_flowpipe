from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "flowstar_scalar_affine_closure"
RUN = (
    ROOT
    / "outputs"
    / "flowstar_scalar_affine_correctness_closure"
    / "20260804T131445Z"
)
SPEC = importlib.util.spec_from_file_location(
    "flowstar_scalar_affine_analysis", EXPERIMENT / "analysis.py"
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "flowstar_scalar_affine_runner", EXPERIMENT / "run_closure.py"
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
sys.path.insert(0, str(EXPERIMENT))
try:
    RUNNER_SPEC.loader.exec_module(RUNNER)
finally:
    sys.path.pop(0)


def _json(name: str) -> dict:
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def test_optional_remote_revision_is_absent_in_single_branch_style_clone(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repository, check=True
    )
    (repository / "tracked").write_text("value\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repository, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    assert RUNNER.optional_git_revision("HEAD", cwd=repository) == head
    assert RUNNER.optional_git_revision("origin/main", cwd=repository) is None


@pytest.mark.unit
def test_scalar_affine_oracle_is_outward_and_monotone() -> None:
    oracle = ANALYSIS.high_precision_outward_oracle("0", "0.1", "0.01")
    assert oracle["precision_decimal_digits"] == 100
    assert oracle["monotonicity"]["verified"] is True
    assert oracle["endpoint"] == [
        0.010100670013377904,
        0.1121208040160535,
    ]
    assert oracle["tube"] == [-5e-324, 0.1121208040160535]

    frozen = _json("analytic_oracle.json")
    assert frozen["classification"] == "formal_mpfr_directed_oracle"
    assert frozen["mpfr"]["meta"]["precision_bits"] == 256
    assert frozen["mpfr"]["meta"]["rounding"] == "explicit_directed"
    assert frozen["endpoint_binary64_outward"] == oracle["endpoint"]
    assert frozen["mpfr"]["bounds"]["endpoint_lower"]["binary64_hex"] == (
        "0x1.4afa8fb004c89p-7"
    )
    assert frozen["mpfr"]["bounds"]["endpoint_upper"]["binary64_hex"] == (
        "0x1.cb3f2f2733eafp-4"
    )


@pytest.mark.unit
def test_containment_defect_is_strict_and_exposes_primary_failure() -> None:
    exported = [0.010100670333333329, 0.1121208036666667]
    oracle = [0.010100670013377904, 0.1121208040160535]
    defect = ANALYSIS.containment_defect(exported, oracle)
    assert defect == {
        "lower_defect": 3.199554250016279e-10,
        "upper_defect": 3.4938679727147814e-10,
        "max_defect": 3.4938679727147814e-10,
        "contained": False,
        "tolerance": None,
    }


@pytest.mark.unit
def test_range_fields_remain_distinct_and_no_repaired_hull_exists() -> None:
    trace = _json("primary_repeat_1/parsed_trace.json")
    fields = ANALYSIS.validate_field_separation(trace)
    assert fields["endpoint_raw"] == [
        0.010100670333333329,
        0.1121208036666667,
    ]
    assert fields["endpoint_collapsed"] == fields["endpoint_raw"]
    assert fields["full_tube"] != fields["endpoint_raw"]
    assert fields["endpoint_tightened"] == {"availability": "unavailable"}
    assert fields["repaired_hull"] == {
        "availability": "unavailable",
        "computed": False,
    }


@pytest.mark.unit
def test_actual_first_loss_is_second_accepted_remainder_refinement() -> None:
    trace = _json("primary_repeat_1/parsed_trace.json")
    oracle = _json("analytic_oracle.json")["endpoint_binary64_outward"]
    rows = ANALYSIS.first_loss_rows(trace, oracle)
    first = next(row for row in rows if row["first_loss"])
    frozen = _json("first_containment_loss.json")

    assert [row["contained"] for row in rows[:5]] == [True] * 5
    assert first["stage"] == "refinement_2_accepted_tmv"
    assert first["flowstar_source"] == "Continuous.cpp:1013-1029"
    assert first == frozen["first_loss"]
    assert frozen["selected_outcome"] == "F_clean_stock_flowstar_core_behavior"
    assert frozen["prevalidation_polynomial_diagnostic"]["term_missing"] is True
    assert frozen["correctness_gate"] == "OPEN"


@pytest.mark.unit
@pytest.mark.protocol
def test_clean_stock_identity_fails_closed_against_patched_checkout() -> None:
    identity = _json("backend_identity.json")
    assert identity["backend_identity"] == "clean-stock"
    assert identity["source_sha"] == "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
    assert identity["tracked_source_clean"] is True
    assert identity["library"]["sha256"] == (
        "b5ff500af66354b0518cf12e7d951f4525f435e8e2d695cf84b91821992c9d9a"
    )
    excluded = identity["patched_checkout_explicitly_excluded"]
    assert excluded["used"] is False
    assert excluded["library_sha256"] != identity["library"]["sha256"]
    assert identity["audit_behavior_environment_variables_enabled"] == []


@pytest.mark.unit
def test_primary_repeat_and_compact_artifact_contract() -> None:
    comparison = _json("primary_repeat_comparison.json")
    assert comparison["stdout_byte_identical"] is True
    assert comparison["parsed_trace_identical"] is True
    assert comparison["passed"] is True

    binary_manifest = _json("ephemeral_binary_manifest.json")
    assert all(not row["retained"] for row in binary_manifest["binaries"])
    assert not (RUN / "build").exists()
    committed_files = {
        row["path"] for row in _json("artifact_manifest.json")["files"]
    }
    assert not any(path.startswith("build/") for path in committed_files)


@pytest.mark.unit
def test_official_route_confirms_under_enclosure_without_field_conflation() -> None:
    parity = _json("official_generated_parity.json")
    assert parity["configuration_parity"] == {
        "candidate_remainder": True,
        "initial_set_representation": True,
        "model_text_and_constants": True,
        "order_and_cutoff": True,
        "preconditioning": True,
        "symbolic_remainder": True,
    }
    assert parity["schedule_parity"]["passed"] is False
    assert parity["official_stock_result"]["completed"] is True
    assert (
        parity["official_stock_result"]["endpoint_defect_at_accepted_right"][
            "contained"
        ]
        is False
    )
    assert parity["diagnostic_replay_matches_accepted_object"]["passed"] is True
