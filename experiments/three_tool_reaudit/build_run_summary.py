#!/usr/bin/env python3
"""Build fail-closed summary rows and collate command provenance."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BOUNDARY = "total_configuration_v2"
FLOWSTAR_SUPPORT = "61a0e19f805fd253ce21e64e2c85c5b4e9c8c86d9d2fbfdf155c1d6a32b98994"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row(
    *,
    backend: str,
    lane: str,
    completed_horizon: float,
    requested_horizon: float,
    validation_status: str,
    soundness_level: str,
    endpoint_semantics: str,
    effective_support_sha256: str,
    runtime_boundary: str,
    backend_sha: str,
    blocker: str,
    **facts: Any,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "lane": lane,
        "completed_horizon": completed_horizon,
        "requested_horizon": requested_horizon,
        "validation_status": validation_status,
        "soundness_level": soundness_level,
        "primary_eligible": False,
        "endpoint_semantics": endpoint_semantics,
        "effective_support_sha256": effective_support_sha256,
        "runtime_boundary": runtime_boundary,
        "backend_sha": backend_sha,
        "run_authority": "authoritative",
        "blocker": blocker,
        **facts,
    }


def _captured_commands(log_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.command.json")):
        record = load(path)
        record["record_path"] = str(path.relative_to(REPO_ROOT))
        record["capture_kind"] = "direct_stream_capture"
        records.append(record)
    return records


def _official_commands(run_dir: Path, official: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = run_dir / "raw" / "flowstar_official_vdp"
    return [
        {
            "schema_version": "reaudit-command-1.0.0",
            "command": run["command"],
            "cwd": run["cwd"],
            "exit_code": run["exit_code"],
            "elapsed_s": run["wall_time_s"],
            "stdout_path": str((root / run["stdout"]).relative_to(REPO_ROOT)),
            "stderr_path": str((root / run["stderr"]).relative_to(REPO_ROOT)),
            "phase": run["phase"],
            "capture_kind": "direct_reproduction_runner_capture",
        }
        for run in official["runs"]
    ]


def _diffreach_commands(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name in (
        "diffreach_native_vdp_t1.json",
        "diffreach_adapter_vdp_t1.json",
        "diffreach_adapter_vdp_t10.json",
    ):
        path = run_dir / "raw" / name
        artifact = load(path)
        records.append(
            {
                "schema_version": "reaudit-command-1.0.0",
                "command": artifact["command"],
                "cwd": str(REPO_ROOT),
                "exit_code": 0,
                "stdout_path": str(path.relative_to(REPO_ROOT)),
                "stderr_path": str(path.relative_to(REPO_ROOT)),
                "capture_kind": "structured_runner_output",
                "capture_note": (
                    "The runner stores completion and any internal exception in this "
                    "single structured artifact; separate process streams were not retained."
                ),
            }
        )
    return records


def _recovered_commands(run_dir: Path) -> list[dict[str, Any]]:
    order_dir = run_dir / "one_step_trace" / "flowstar_vdp_o2_rejection"
    diagnostic = load(order_dir / "diagnostic_manifest.json")
    compile_record = load(order_dir / "flowstar_probe_compile.json")
    run_record = load(order_dir / "flowstar_probe_run.json")
    records: list[dict[str, Any]] = [
        {
            "schema_version": "reaudit-command-1.0.0",
            "command": diagnostic["command"],
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "stdout_path": str((order_dir / "diagnostic_manifest.json").relative_to(REPO_ROOT)),
            "stderr_path": str((order_dir / "diagnostic_manifest.json").relative_to(REPO_ROOT)),
            "capture_kind": "structured_runner_output",
        }
    ]
    for phase, record in (("compile", compile_record), ("run", run_record)):
        records.append(
            {
                "schema_version": "reaudit-command-1.0.0",
                "command": record["command"],
                "cwd": str(REPO_ROOT),
                "exit_code": record.get("return_code", record.get("exit_code")),
                "stdout_path": str((order_dir / record["stdout"]).relative_to(REPO_ROOT)),
                "stderr_path": str((order_dir / record["stderr"]).relative_to(REPO_ROOT)),
                "phase": f"flowstar_order2_probe_{phase}",
                "capture_kind": "direct_diagnostic_runner_capture",
            }
        )
    h10 = run_dir / "vdp_t10" / "h10_right_map_centering"
    records.append(
        {
            "schema_version": "reaudit-command-1.0.0",
            "command": [
                "conda", "run", "-n", "py11", "python",
                "experiments/flowstar_raw_remainder_compat_h10_right_map_centering.py",
                "--horizon", "10", "--wall-cap-s", "300",
                "--flowstar-segments",
                str(
                    run_dir.relative_to(REPO_ROOT)
                    / "raw/flowstar_official_generated_parity/original_flowstar/original_flowstar_segments.csv"
                ),
                "--out-dir", str(h10.relative_to(REPO_ROOT)),
            ],
            "environment": {"OMP_NUM_THREADS": "1"},
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "stdout_path": {
                "availability": "unavailable",
                "reason": "the long-horizon runner's separate process stdout was not retained",
            },
            "stderr_path": {
                "availability": "unavailable",
                "reason": "the long-horizon runner's separate process stderr was not retained",
            },
            "structured_output_path": str(
                (h10 / "h10_right_map_centering_summary.csv").relative_to(REPO_ROOT)
            ),
            "capture_kind": "recovered_from_exact_documented_command_and_raw_artifacts",
        }
    )
    return records


def build(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    official = load(run_dir / "raw" / "flowstar_official_vdp" / "evidence.json")
    diff_native = load(run_dir / "raw" / "diffreach_native_vdp_t1.json")
    diff_adapter = load(run_dir / "raw" / "diffreach_adapter_vdp_t10.json")
    xiangru = load(run_dir / "raw" / "xiangru_source_inventory.json")
    h10_rows = csv_rows(
        run_dir / "vdp_t10" / "h10_right_map_centering" / "h10_right_map_centering_summary.csv"
    )
    manifest = load(run_dir / "manifest.json")
    flowstar_sha = manifest["repositories"]["flowstar"]["sha"]
    diffreach_sha = manifest["repositories"]["diffreach"]["sha"]
    torch_sha = git_sha(REPO_ROOT)

    official_last = official["runs"][-1]
    rows = [
        row(
            backend="official-stock",
            lane="native_reproduction",
            completed_horizon=10.0,
            requested_horizon=10.0,
            validation_status="completed",
            soundness_level="formal_outward_rounding",
            endpoint_semantics="unavailable",
            effective_support_sha256="unavailable_on_official_program_route",
            runtime_boundary="correctness_reproduction_process_wall_v1",
            backend_sha=flowstar_sha,
            blocker="native raw fixed-time endpoint and matched 1+10 timing are unavailable",
            accepted_segments=official_last["accepted_segments"],
            cold_runtime_s=official["runs"][0]["wall_time_s"],
            steady_runtime_s=[run["wall_time_s"] for run in official["runs"][1:]],
        ),
        row(
            backend="generated-stock",
            lane="native_reproduction",
            completed_horizon=10.0,
            requested_horizon=10.0,
            validation_status="schema_invalid",
            soundness_level="formal_outward_rounding",
            endpoint_semantics="segment_box",
            effective_support_sha256="unavailable_on_official_program_route",
            runtime_boundary="unmatched_generated_process_wall_v1",
            backend_sha=flowstar_sha,
            blocker="plot boxes match, but official internal fields and native endpoint are unavailable",
            accepted_segments=290,
            plot_segment_max_abs_difference=0.0,
        ),
    ]
    for source in ("constant_adaptive_h10", "range_midpoint_adaptive_h10"):
        result = next(value for value in h10_rows if value["mode"] == source)
        rows.append(
            row(
                backend=f"torch-sparse-{source}",
                lane="native_reproduction",
                completed_horizon=float(result["reached_t"]),
                requested_horizon=10.0,
                validation_status="validation_rejected",
                soundness_level="safeguarded_float64_not_fully_proved",
                endpoint_semantics="raw_endpoint",
                effective_support_sha256=FLOWSTAR_SUPPORT,
                runtime_boundary="diagnostic_mode_elapsed_v1",
                backend_sha=torch_sha,
                blocker=result["first_failure_reason"],
                accepted_steps=int(result["accepted_steps"]),
                rejected_attempts=int(result["rejected_attempts"]),
                final_segment_width_sum=float(result["final_segment_width_sum"]),
                runtime_s=float(result["runtime_s"]),
            )
        )
    rows.extend(
        [
            row(
                backend="diffreach-native",
                lane="native_reproduction",
                completed_horizon=float(diff_native["completion"]["completed_horizon"]),
                requested_horizon=1.0,
                validation_status=diff_native["completion"]["validation_status"],
                soundness_level="unknown",
                endpoint_semantics="unavailable",
                effective_support_sha256=diff_adapter["basis"]["effective_support_sha256"],
                runtime_boundary="setup_failed_before_runtime_boundary",
                backend_sha=diffreach_sha,
                blocker=diff_native["failure"]["message"],
            ),
            row(
                backend="diffreach-canonical-adapter",
                lane="matched_plant_backend",
                completed_horizon=10.0,
                requested_horizon=10.0,
                validation_status="completed",
                soundness_level="unknown",
                endpoint_semantics="unavailable",
                effective_support_sha256=diff_adapter["basis"]["effective_support_sha256"],
                runtime_boundary="compile_and_execute_plus_execute_v1",
                backend_sha=diffreach_sha,
                blocker="adapter is non-native, lacks directed rounding/raw endpoint, and has only one steady run",
                partitions=diff_adapter["config"]["partitions"],
                cold_runtime_s=diff_adapter["execution"]["cold_compile_and_execute_s"],
                steady_runtime_s=diff_adapter["execution"]["steady_execute_s"],
                endpoint_box=diff_adapter["output"]["endpoint_box"],
            ),
            row(
                backend="xiangru-private-2026",
                lane="native_end_to_end_certificate",
                completed_horizon=0.0,
                requested_horizon=10.0,
                validation_status=xiangru["private_source_status"],
                soundness_level="unknown",
                endpoint_semantics="unavailable",
                effective_support_sha256="unavailable_missing_private_source",
                runtime_boundary="unavailable",
                backend_sha="unavailable_missing_private_source",
                blocker=xiangru["historical_timing_blocker"],
                historical_timing_recomputed=False,
            ),
        ]
    )

    gate_index = load(run_dir / "gate_evidence" / "index.json")
    passed = sorted(name for name, value in gate_index["gates"].items() if value["passed"])
    blocked = sorted(name for name, value in gate_index["gates"].items() if not value["passed"])
    summary = {
        "schema_version": "three-tool-summary-1.0.0",
        "run_id": run_dir.name,
        "status": "fail_closed_gate_blocked",
        "headline_comparison_generated": False,
        "headline_pareto_generated": False,
        "headline_speedup_generated": False,
        "gate_counts": {"passed": len(passed), "blocked": len(blocked)},
        "passed_gates": passed,
        "remaining_blockers": blocked,
        "rows": rows,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commands = [
        *_official_commands(run_dir, official),
        *_diffreach_commands(run_dir),
        *_recovered_commands(run_dir),
        *_captured_commands(run_dir / "logs"),
    ]
    (run_dir / "logs" / "command_records.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    summary = build(arguments.run_dir)
    print(json.dumps({key: summary[key] for key in ("run_id", "status", "gate_counts")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
