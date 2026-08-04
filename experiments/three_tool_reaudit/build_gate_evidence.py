#!/usr/bin/env python3
"""Build the eight current-run gate evidence records from raw artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torch_tm_flowpipe.protocol.backend_identity import (
    BackendIdentityError,
    inspect_primary_flowstar_backend,
)
from torch_tm_flowpipe.protocol.reaudit import sha256_file, validate_primary_row


REPORTS = {
    "stock_backend_identity": "docs/FLOWSTAR_STOCK_VDP_REPRODUCTION.md",
    "official_parser_generated_stock_field_parity": "docs/TORCH_FLOWSTAR_ONE_STEP_TRACE_PARITY.md",
    "endpoint_segment_tube_exporter_semantics": "docs/FLOWSTAR_ENDPOINT_EXPORTER_AUDIT.md",
    "raw_tightened_separation": "docs/FLOWSTAR_ENDPOINT_EXPORTER_AUDIT.md",
    "order_basis_contract": "docs/TORCH_FLOWSTAR_ONE_STEP_TRACE_PARITY.md",
    "runtime_boundary_parity": "docs/THREE_TOOL_FINAL_CORRECTNESS_REPORT.md",
    "completion_validation_fail_closed": "docs/THREE_TOOL_FINAL_CORRECTNESS_REPORT.md",
    "patched_rows_excluded_from_primary": "docs/THREE_TOOL_FINAL_CORRECTNESS_REPORT.md",
}

TESTS = {
    "stock_backend_identity": "tests/test_backend_identity.py",
    "official_parser_generated_stock_field_parity": "tests/test_current_run_gate_evidence.py::test_every_gate_has_current_machine_evidence",
    "endpoint_segment_tube_exporter_semantics": "tests/test_one_step_source_coordinates.py::test_current_run_trajectory_sanity_is_fail_closed",
    "raw_tightened_separation": "tests/test_protocol_contracts.py::test_diagnostic_endpoint_semantics_are_distinct_and_primary_ineligible",
    "order_basis_contract": "tests/test_current_run_gate_evidence.py::test_current_support_contract_records_match_and_mismatch",
    "runtime_boundary_parity": "tests/test_protocol_contracts.py::test_total_timer_includes_completion_boundary",
    "completion_validation_fail_closed": "tests/test_diffreach_native_reproduction.py::test_completion_requires_every_picard_contraction",
    "patched_rows_excluded_from_primary": "tests/test_backend_identity.py",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(run_dir: Path, path: Path) -> str:
    return str(path.resolve().relative_to(run_dir.resolve()))


def inputs(run_dir: Path, paths: Iterable[Path]) -> list[dict[str, str]]:
    return [
        {"path": relative(run_dir, path), "sha256": sha256_file(path)}
        for path in paths
    ]


def write_gate(
    run_dir: Path,
    name: str,
    *,
    passed: bool,
    applies_to: list[str],
    blocker: str | None,
    facts: Mapping[str, Any],
    source_paths: Iterable[Path],
) -> Path:
    record = {
        "schema_version": "cross-tool-gate-evidence-1.0.0",
        "run_id": run_dir.name,
        "gate": name,
        "passed": bool(passed),
        "blocker": blocker,
        "applies_to": applies_to,
        "automated_test": TESTS[name],
        "report": REPORTS[name],
        "inputs": inputs(run_dir, source_paths),
        "facts": facts,
    }
    output = run_dir / "gate_evidence" / f"{name}.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _ldd(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        ["ldd", str(path)], text=True, capture_output=True, check=False
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "exit_code": process.returncode,
        "dependencies": [
            re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", line)
            for line in process.stdout.splitlines()
        ],
        "stderr": process.stderr,
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build(run_dir: Path, flowstar_root: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    gate_dir = run_dir / "gate_evidence"
    gate_dir.mkdir(parents=True, exist_ok=True)
    official_path = run_dir / "raw" / "flowstar_official_vdp" / "evidence.json"
    official = load(official_path)
    vdp_flow_path = run_dir / "one_step_trace" / "flowstar_van_der_pol_official_o4.json"
    vdp_flow = load(vdp_flow_path)
    generated_binary = run_dir / "one_step_trace" / "flowstar_van_der_pol_official_o4_work" / "export_segment"
    generated_source = generated_binary.with_suffix(".cpp")
    identity = inspect_primary_flowstar_backend(flowstar_root, environment={})
    same_sha = vdp_flow["execution"]["repository_commit"] == identity.repository_sha
    same_library = (
        official["library"]["sha256"]
        == identity.library_sha256
    )
    compatibility_scope_ok = bool(
        identity.backend_class == "unmodified-stock"
        or (
            identity.backend_class == "stock-plus-gcc15-compat"
            and identity.gcc15_derivative_compatibility_change
        )
    )
    stock_pass = bool(
        identity.primary_eligible
        and compatibility_scope_ok
        and same_sha
        and same_library
        and generated_binary.is_file()
        and generated_source.is_file()
        and not identity.audit_behavior_variables_enabled
    )
    stock_path = write_gate(
        run_dir,
        "stock_backend_identity",
        passed=stock_pass,
        applies_to=["official-stock", "generated-stock", "van_der_pol_order4"],
        blocker=None if stock_pass else "stock repository/library/binary identity check failed",
        facts={
            "identity": identity.to_record(),
            "official_binary": _ldd(Path(official["binary"]["path"])),
            "generated_binary": _ldd(generated_binary),
            "generated_compile_link_contract": {
                "library_search_path": str(flowstar_root / "flowstar-toolbox"),
                "link_argument": "-lflowstar",
                "only_matching_library": str(flowstar_root / "flowstar-toolbox" / "libflowstar.a"),
                "library_sha256_at_compile_evidence": identity.library_sha256,
            },
            "same_repository_sha": same_sha,
            "same_library_sha256": same_library,
            "gcc15_patch_scope": "TaylorModel::derivative_assign assignment target only",
        },
        # Record the generated executable hash/linkage above, but do not make
        # a disposable build product part of the committed evidence closure.
        source_paths=[official_path, vdp_flow_path, generated_source],
    )

    parity_dir = run_dir / "raw" / "flowstar_official_generated_parity"
    comparison_path = parity_dir / "generated_flowstar_vs_original_comparison.csv"
    comparison = {
        row["metric"]: row["value"] for row in _rows(comparison_path)
    }
    segment_plot_parity = (
        comparison["segment_count_match"] == "true"
        and float(comparison["max_abs_segment_field_diff"]) == 0.0
    )
    required_unavailable = [
        "source_polynomial_and_raw_remainder",
        "picard_iteration",
        "discarded_monomials",
        "candidate_self_map_defect",
        "native_fixed_time_endpoint",
        "accepted_and_rejected_attempt_reason",
    ]
    parity_path = write_gate(
        run_dir,
        "official_parser_generated_stock_field_parity",
        passed=False,
        applies_to=["official-stock", "generated-stock", "van_der_pol_order4_t10"],
        blocker="official program exposes plot segment boxes but not the required internal field trace",
        facts={
            "plot_segment_parity_passed": segment_plot_parity,
            "segments": int(comparison["original_num_segments"]),
            "max_abs_plot_segment_field_difference": float(comparison["max_abs_segment_field_diff"]),
            "required_fields_unavailable_on_official_route": required_unavailable,
            "partial_plot_parity_does_not_upgrade_gate": True,
        },
        source_paths=[comparison_path, parity_dir / "parity_report.md", official_path],
    )

    one_step_path = gate_dir / "one_step_parity.json"
    one_step = load(one_step_path)
    trajectory_failures = {
        name: {
            backend: value["trajectory_sanity"][backend]
            for backend in ("flowstar", "torch")
            if not value["trajectory_sanity"][backend]["passed"]
        }
        for name, value in one_step["cases"].items()
        if any(
            not value["trajectory_sanity"][backend]["passed"]
            for backend in ("flowstar", "torch")
        )
    }
    exporter_pass = not trajectory_failures and all(
        value["enclosures"]["flowstar_raw_endpoint_inside_flowstar_last_segment"]
        and value["enclosures"]["torch_raw_endpoint_inside_torch_last_segment"]
        for value in one_step["cases"].values()
    )
    exporter_path = write_gate(
        run_dir,
        "endpoint_segment_tube_exporter_semantics",
        passed=exporter_pass,
        applies_to=["generated-stock", "torch-sparse", "one_step_matched_suite"],
        blocker=(
            None
            if exporter_pass
            else "independent trajectory sanity found an exported enclosure violation"
        ),
        facts={
            "raw_endpoint_inside_segment_all_cases": all(
                value["enclosures"]["flowstar_raw_endpoint_inside_flowstar_last_segment"]
                and value["enclosures"]["torch_raw_endpoint_inside_torch_last_segment"]
                for value in one_step["cases"].values()
            ),
            "trajectory_failures": trajectory_failures,
            "zero_violations_would_be_sanity_not_proof": True,
            "official_t10_fixed_endpoint_available": False,
        },
        source_paths=[one_step_path, vdp_flow_path],
    )

    vdp_audit = vdp_flow["native_metadata"]["endpoint_path_audit"]
    under_enclosure = [
        row
        for row in vdp_audit
        if row["padding_lower"] < 0.0 or row["padding_upper"] > 0.0
    ]
    raw_sep_pass = bool(
        under_enclosure
        and vdp_flow["enclosures"]["endpoint_raw"]["box"]
        != vdp_flow["enclosures"]["endpoint_collapsed"]["box"]
        and vdp_flow["enclosures"]["repaired_hull"]["box"]
        != vdp_flow["enclosures"]["endpoint_collapsed"]["box"]
    )
    raw_path = write_gate(
        run_dir,
        "raw_tightened_separation",
        passed=raw_sep_pass,
        applies_to=["all_exported_endpoint_fields", "primary_raw_endpoint"],
        blocker=None if raw_sep_pass else "raw/collapsed/repaired fields were not independently exported",
        facts={
            "flowstar_vdp_endpoint_path_audit": vdp_audit,
            "collapsed_under_enclosure_diagnostic_states": under_enclosure,
            "primary_field": "endpoint_raw",
            "fallback_forbidden": True,
        },
        source_paths=[vdp_flow_path, one_step_path],
    )

    diffreach_t10_path = run_dir / "raw" / "diffreach_adapter_vdp_t10.json"
    diffreach_t10 = load(diffreach_t10_path)
    matched_support = {
        name: value["effective_support"] for name, value in one_step["cases"].items()
    }
    observed_equal = all(value["all_equal"] for value in matched_support.values())
    flow_hashes = {value["flowstar_sha256"] for value in matched_support.values()}
    diffreach_hash = diffreach_t10["basis"]["effective_support_sha256"]
    support_pass = observed_equal and bool(diffreach_hash) and diffreach_hash not in flow_hashes
    basis_path = write_gate(
        run_dir,
        "order_basis_contract",
        passed=support_pass,
        applies_to=["one_step_matched_suite", "diffreach-canonical-adapter", "all_order_grouping"],
        blocker=None if support_pass else "effective support hashes are incomplete or ambiguously grouped",
        facts={
            "flowstar_torch_observed_support": matched_support,
            "diffreach_restricted_support": diffreach_t10["basis"],
            "diffreach_support_intentionally_not_grouped_as_complete_order": True,
            "nominal_order_only_grouping_forbidden": True,
        },
        source_paths=[one_step_path, diffreach_t10_path],
    )

    h10_summary_path = run_dir / "vdp_t10" / "h10_right_map_centering" / "h10_right_map_centering_summary.csv"
    h10_rows = _rows(h10_summary_path)
    xiangru_path = run_dir / "raw" / "xiangru_source_inventory.json"
    xiangru = load(xiangru_path)
    runtime_path = write_gate(
        run_dir,
        "runtime_boundary_parity",
        passed=False,
        applies_to=["all_performance_rows", "time_to_certificate", "headline_speedup"],
        blocker="current run lacks matched 1-cold plus 10-steady total-configuration timing for every eligible backend",
        facts={
            "official_correctness_repetitions": len(official["runs"]),
            "official_steady_repetitions": len(official["runs"]) - 1,
            "required_steady_repetitions_for_timing": 10,
            "diffreach_steady_repetitions": len(diffreach_t10["execution"]["steady_execute_s"]),
            "torch_long_horizon_rows": [
                {
                    "mode": row["mode"],
                    "status": row["status"],
                    "runtime_s": row["runtime_s"],
                    "reached_t": row["reached_t"],
                }
                for row in h10_rows
                if row["source"] == "torch"
            ],
            "time_to_certificate_ranking_allowed": False,
            "xiangru_private_source_status": xiangru["private_source_status"],
            "xiangru_historical_timing_recomputed": xiangru["historical_timing_recomputed"],
        },
        source_paths=[official_path, diffreach_t10_path, h10_summary_path, xiangru_path],
    )

    order2_path = run_dir / "one_step_trace" / "flowstar_vdp_o2_rejection" / "order2_failure_manifest.json"
    order2 = load(order2_path)
    native_diff_path = run_dir / "raw" / "diffreach_native_vdp_t1.json"
    native_diff = load(native_diff_path)
    incomplete_torch = [
        row
        for row in h10_rows
        if row["source"] == "torch" and row["reached_h10"] != "true"
    ]
    completion_pass = bool(
        order2["flowstar_diagnostic"]["failure_category"] == "validation_rejected"
        and native_diff["completion"]["requested_horizon_reached"] is False
        and incomplete_torch
        and diffreach_t10["completion"]["requested_horizon_reached"] is True
    )
    completion_path = write_gate(
        run_dir,
        "completion_validation_fail_closed",
        passed=completion_pass,
        applies_to=["all_primary_rows", "pareto", "time_to_certificate"],
        blocker=None if completion_pass else "external incomplete/warning paths were not rejected",
        facts={
            "flowstar_order2": order2["flowstar_diagnostic"],
            "diffreach_upstream_native": native_diff["completion"],
            "diffreach_canonical_adapter": diffreach_t10["completion"],
            "torch_incomplete_rows": [
                {
                    "mode": row["mode"],
                    "status": row["status"],
                    "reached_t": row["reached_t"],
                    "reached_h10": row["reached_h10"],
                }
                for row in incomplete_torch
            ],
            "failed_rows_enter_headline": False,
        },
        source_paths=[order2_path, native_diff_path, diffreach_t10_path, h10_summary_path],
    )

    environment_rejections: dict[str, str] = {}
    for variable in (
        "FLOWSTAR_AUDIT_CACHE_LEAF_TRUNCATION",
        "FLOWSTAR_AUDIT_REVALIDATE_REFINEMENT",
    ):
        try:
            inspect_primary_flowstar_backend(flowstar_root, environment={variable: "1"})
        except BackendIdentityError as error:
            environment_rejections[variable] = str(error)
    fixture = {
        "backend": "torch-dense-prototype",
        "lane": "matched_plant_backend",
        "completed_horizon": 10.0,
        "requested_horizon": 10.0,
        "validation_status": "completed",
        "soundness_level": "empirical_enclosure_only",
        "primary_eligible": True,
        "endpoint_semantics": "raw_endpoint",
        "effective_support_sha256": "f" * 64,
        "runtime_boundary": "total_configuration_v2",
        "backend_sha": "1" * 40,
        "run_authority": "authoritative",
    }
    dense_errors = validate_primary_row(fixture)
    patched_errors = validate_primary_row({**fixture, "backend": "patched-audit"})
    exclusion_pass = (
        len(environment_rejections) == 2
        and bool(dense_errors)
        and bool(patched_errors)
    )
    exclusion_path = write_gate(
        run_dir,
        "patched_rows_excluded_from_primary",
        passed=exclusion_pass,
        applies_to=["patched-audit", "torch-dense-prototype", "all_primary_rows"],
        blocker=None if exclusion_pass else "negative backend/row exclusions did not all trigger",
        facts={
            "audit_environment_rejections": environment_rejections,
            "dense_row_errors": dense_errors,
            "patched_row_errors": patched_errors,
            "dense_kernel_semantics": "Euler feasibility prototype, not a validated TM flowpipe",
        },
        source_paths=[official_path, vdp_flow_path],
    )

    paths = [
        stock_path,
        parity_path,
        exporter_path,
        raw_path,
        basis_path,
        runtime_path,
        completion_path,
        exclusion_path,
    ]
    index = {
        "schema_version": "cross-tool-gate-index-1.0.0",
        "run_id": run_dir.name,
        "gates": {
            path.stem: {
                "path": relative(run_dir, path),
                "sha256": sha256_file(path),
                "passed": load(path)["passed"],
                "blocker": load(path)["blocker"],
            }
            for path in paths
        },
        "all_passed": all(load(path)["passed"] for path in paths),
        "headline_comparison_allowed": False,
    }
    index_path = gate_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    args = parser.parse_args()
    index = build(args.run_dir, args.flowstar_root.resolve())
    print(json.dumps(index, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
