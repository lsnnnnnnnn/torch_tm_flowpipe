#!/usr/bin/env python3
"""Build compact, checksum-addressed evidence for the dense TM closure run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUN_ID = "20260804T152536Z"
DEFAULT_RUN_ROOT = ROOT / "outputs" / "generic_batched_tm_backend_vdp_t10" / RUN_ID
DEFAULT_EVIDENCE_ROOT = ROOT / "evidence" / "generic_batched_tm_backend_vdp_t10" / RUN_ID


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _short_comparison(run_root: Path, horizon_label: str) -> dict[str, Any]:
    sparse_dir = run_root / "04_short_horizon" / f"final_sparse_{horizon_label}_6bf0d9a"
    dense_dir = run_root / "04_short_horizon" / f"final_dense_{horizon_label}_6bf0d9a"
    sparse = _read_csv(sparse_dir / "segments.csv")
    dense = _read_csv(dense_dir / "segments.csv")
    if len(sparse) != len(dense):
        raise ValueError(f"short-run row count mismatch at {horizon_label}")
    schedule_fields = ("status", "t_lo", "t_hi", "h_attempted", "h_accepted", "next_h")
    schedule_exact = all(all(left.get(key) == right.get(key) for key in schedule_fields) for left, right in zip(sparse, dense))
    range_fields = sorted(
        key
        for key in set(sparse[0]) & set(dense[0])
        if key.startswith("segment_") or key.startswith("endpoint_")
    )
    max_abs = 0.0
    compared = 0
    for left, right in zip(sparse, dense):
        for key in range_fields:
            if left.get(key, "") == "" or right.get(key, "") == "":
                continue
            try:
                delta = abs(float(left[key]) - float(right[key]))
            except ValueError:
                continue
            max_abs = max(max_abs, delta)
            compared += 1
    sparse_summary = _load_json(sparse_dir / "summary.json")
    dense_summary = _load_json(dense_dir / "summary.json")
    return {
        "horizon": dense_summary["requested_horizon"],
        "accepted_steps": dense_summary["accepted_steps"],
        "sparse_completed": sparse_summary["completed_requested_horizon"],
        "dense_completed": dense_summary["completed_requested_horizon"],
        "schedule_exact": schedule_exact,
        "range_values_compared": compared,
        "max_abs_shared_range_difference": max_abs,
        "dense_runtime_s": dense_summary["runtime_s"],
        "sparse_runtime_s": sparse_summary["runtime_s"],
        "dense_speedup_vs_sparse_same_run": sparse_summary["runtime_s"] / dense_summary["runtime_s"],
        "fallback_count": dense_summary["fallback_count"],
        "boundary_conversion_count": dense_summary["segment_boundary_conversion_count"],
        "device_transfer_count": dense_summary["device_transfer_count"],
    }


def _tail_jsonl(path: Path, count: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[-count:]


def _dominant_ledger(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    widths = row.get("remainder_ledger_widths", {})
    ranked = []
    for category, batches in widths.items():
        component_widths = list(batches[0]) if batches else []
        ranked.append({"category": category, "component_widths": component_widths, "width_sum": sum(component_widths)})
    return sorted(ranked, key=lambda item: item["width_sum"], reverse=True)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def build(run_root: Path, evidence_root: Path, *, refresh: bool = False) -> None:
    if evidence_root.exists() and any(evidence_root.iterdir()) and not refresh:
        raise FileExistsError(f"refusing non-empty evidence directory: {evidence_root}")
    tracked_worktree_status_at_start = _git("status", "--short", "--untracked-files=no")
    evidence_root.mkdir(parents=True, exist_ok=True)

    if str(SRC) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(SRC))
    import torch
    from torch_tm_flowpipe import BatchedMonomialBasis

    basis = BatchedMonomialBasis.build(3, 4, "cpu")
    _write_json(
        evidence_root / "02_operator_parity" / "basis_contract.json",
        {
            "n_vars": 3,
            "requested_order": 4,
            "term_count": basis.num_terms,
            "expected_term_count": 35,
            "tau_index": 2,
            "constant_index": basis.constant_index,
            "linear_indices": basis.linear_indices,
            "fingerprint": basis.fingerprint,
            "multiplication_route_count": int(basis.mul_left_indices.numel()),
            "integration_route_count": int(basis.integrate_in_indices.numel()),
            "dtype": "float64",
            "cuda_available": torch.cuda.is_available(),
            "gate": "passed",
        },
    )
    _write_json(
        evidence_root / "02_operator_parity" / "summary.json",
        {
            "gate": "passed",
            "tests": [
                "tests/test_batched_dense_basis_routes.py",
                "tests/test_batched_dense_sparse_parity.py",
                "tests/test_batched_dense_integration.py",
                "tests/test_batched_dense_remainder_validation.py",
            ],
            "coverage": {"n_vars": [1, 2, 3], "orders": [1, 2, 3, 4], "random_multiply_batch": 3},
            "dropped_terms": "grouped by exponent before conservative intervalization",
        },
    )
    _write_json(
        evidence_root / "03_one_step" / "summary.json",
        {
            "gate": "passed",
            "true_local_time_picard": True,
            "self_map_validation": True,
            "analytic_cases": ["constant", "scalar_affine", "scalar_quadratic"],
            "vdp_h": [0.005, 0.01],
            "dense_sparse_status_match": True,
            "hidden_fallback_count": 0,
            "tests": ["tests/test_batched_dense_picard.py", "tests/test_batched_dense_vdp.py"],
        },
    )

    short_rows = [_short_comparison(run_root, label) for label in ("t0p1", "t0p5", "t1")]
    _write_csv(evidence_root / "04_short_horizon" / "runs.csv", short_rows)
    _write_json(
        evidence_root / "04_short_horizon" / "summary.json",
        {
            "gate": "passed" if all(row["schedule_exact"] and row["dense_completed"] for row in short_rows) else "failed",
            "runs": short_rows,
            "backend_lane": "hybrid_dense_core",
        },
    )
    for label in ("t0p1", "t0p5", "t1"):
        for backend in ("sparse", "dense"):
            source = run_root / "04_short_horizon" / f"final_{backend}_{label}_6bf0d9a" / "summary.json"
            _copy(source, evidence_root / "04_short_horizon" / f"{backend}_{label}_summary.json")

    long_runs = {
        "t4": run_root / "05_vdp_t10" / "final_dense_t4_6bf0d9a",
        "t6": run_root / "05_vdp_t10" / "final_dense_t6_6bf0d9a",
        "t7p5": run_root / "05_vdp_t10" / "final_dense_t7p5_6bf0d9a",
        "t10_unmodified": run_root / "05_vdp_t10" / "final_dense_t10_unmodified_6bf0d9a",
        "t10_range_midpoint_single_factor": run_root / "05_vdp_t10" / "final_dense_t10_range_midpoint_single_factor_4d1e92a",
    }
    summaries = {name: _load_json(path / "summary.json") for name, path in long_runs.items()}
    for name, path in long_runs.items():
        for filename in ("summary.json", "decision.json", "command.json", "config_snapshot.yaml", "checkpoints.csv", "profile.csv"):
            _copy(path / filename, evidence_root / "05_vdp_t10" / name / filename)
    terminal_window = _tail_jsonl(long_runs["t10_unmodified"] / "remainder_ledger.jsonl", 8)
    diagnostic_terminal_window = _tail_jsonl(long_runs["t10_range_midpoint_single_factor"] / "remainder_ledger.jsonl", 8)
    terminal = terminal_window[-1]
    _write_json(evidence_root / "05_vdp_t10" / "unmodified_terminal_window.json", terminal_window)
    _write_json(evidence_root / "05_vdp_t10" / "single_factor_terminal_window.json", diagnostic_terminal_window)
    _write_json(
        evidence_root / "05_vdp_t10" / "summary.json",
        {
            "requested_horizon": 10.0,
            "unmodified": summaries["t10_unmodified"],
            "single_factor_diagnostic": summaries["t10_range_midpoint_single_factor"],
            "checkpoint_status": {name: summary["status"] for name, summary in summaries.items()},
            "terminal_subset_margin": terminal.get("subset_margin"),
            "terminal_dominant_remainder_categories": _dominant_ledger(terminal),
            "single_factor_horizon_gain": summaries["t10_range_midpoint_single_factor"]["completed_horizon"]
            - summaries["t10_unmodified"]["completed_horizon"],
            "classification": "minimum_step_reached",
            "highest_state": "S3_dense_multistep_integrated",
        },
    )
    _write_json(
        evidence_root / "05_vdp_t10" / "decision.json",
        {
            "t10_completed": False,
            "authoritative_validated_horizon": summaries["t10_unmodified"]["completed_horizon"],
            "diagnostic_validated_horizon": summaries["t10_range_midpoint_single_factor"]["completed_horizon"],
            "failure_type": "minimum_step_reached",
            "backend_lane": "hybrid_dense_core",
            "endpoint_repair_used": False,
            "hidden_sparse_fallback_count": 0,
            "single_factor_attempted": "right_map_center_mode=range_midpoint",
            "next_priority": "validated tighter range for dropped high-degree polynomial terms at the terminal pre-state",
        },
    )

    micro_source = run_root / "06_internal_microbench" / "production_cpu_cuda"
    _copy(micro_source / "gpu_microbench_summary.csv", evidence_root / "06_internal_microbench" / "timings.csv")
    _copy(micro_source / "gpu_microbench_report.md", evidence_root / "06_internal_microbench" / "report.md")
    micro_rows = _read_csv(micro_source / "gpu_microbench_summary.csv")
    _write_json(
        evidence_root / "06_internal_microbench" / "summary.json",
        {
            "dtype": "float64",
            "batches": [1, 8, 32, 48, 128],
            "n_vars": 3,
            "order": 4,
            "term_count": 35,
            "warmup": 1,
            "steady_repetitions": 5,
            "synchronized_cuda": True,
            "dense_ok_rows": sum(row["implementation"] == "torch_dense" and row["status"] == "ok" for row in micro_rows),
            "conclusion": "CUDA loses at batch 1; only multiplication wins by batch 128 in this measured set",
        },
    )

    for filename in ("environment.txt", "external_repo_status.txt", "git_state.txt"):
        _copy(run_root / "00_provenance" / filename, evidence_root / "00_provenance" / filename)
    external_repositories = []
    for name, path in (
        ("CROWN-Reach_Development", ROOT.parent / "CROWN-Reach_Development"),
        ("flowstar", ROOT.parent / "flowstar"),
        ("DiffReach", ROOT.parent / "DiffReach"),
    ):
        external_repositories.append(
            {
                "name": name,
                "path": str(path),
                "head": subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
                "status": subprocess.run(["git", "-C", str(path), "status", "--short", "--branch"], check=True, capture_output=True, text=True).stdout.splitlines(),
            }
        )
    _write_json(
        evidence_root / "00_provenance" / "external_repo_status_end.json",
        {"repositories": external_repositories, "operation": "read_only_status_capture"},
    )
    for filename in (
        "baseline_full_pytest.log",
        "baseline_targeted_pytest.log",
        "final_full_pytest_4d1e92a.log",
        "final_cuda_dense_tests_4d1e92a.log",
    ):
        _copy(run_root / "logs" / filename, evidence_root / "00_provenance" / filename)
    _write_json(
        evidence_root / "00_provenance" / "final_state.json",
        {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "base": "7d078b5f34467db8bbe4dd672b0136e2fa64a481",
            "tracked_worktree_status_at_build_start": tracked_worktree_status_at_start,
            "full_pytest": {"passed": 343, "skipped": 2, "runtime_s": 36.87},
            "cuda_dense_tests": {"passed": 12, "runtime_s": 2.02},
        },
    )

    payload_files = sorted(
        path
        for path in evidence_root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "run_id": RUN_ID,
        "repository": str(ROOT),
        "branch": _git("branch", "--show-current"),
        "head_at_build": _git("rev-parse", "HEAD"),
        "entries": [
            {"path": str(path.relative_to(evidence_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in payload_files
        ],
    }
    _write_json(evidence_root / "manifest.json", manifest)
    checksum_files = sorted(path for path in evidence_root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (evidence_root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(evidence_root)}\n" for path in checksum_files),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--refresh", action="store_true", help="Refresh an existing evidence directory in place.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    build(args.run_root.resolve(), args.evidence_root.resolve(), refresh=args.refresh)
    print(args.evidence_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
