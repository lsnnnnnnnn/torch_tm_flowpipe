from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_huan_proof_closure_package.py"
SPEC = importlib.util.spec_from_file_location("verify_huan_proof_closure_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def _d2_row(backend: str, *, invoked: bool = True) -> dict[str, object]:
    row: dict[str, object] = {
        "schedule_name": "engine_interval_dot",
        "execution_backend": backend,
        "actual_device": "cuda:0" if "cuda" in backend else "cpu",
        "kernel_path": "flowstar_gpu.interval.dot_point_iv",
        "kernel_invocation_observed": invoked,
        "m": 2,
        "finite_hypotheses_satisfied": True,
        "m_u_gate": True,
        "exact_error": None,
        "computed_inflation": 1e-15,
        "containment": True,
        "status": "PASS",
    }
    if backend == "custom_cuda":
        row["custom_cuda_invocation_count"] = int(invoked)
    return row


def test_cuda_availability_cannot_replace_actual_invocation() -> None:
    rows = [_d2_row("torch_cuda"), _d2_row("custom_cuda", invoked=False)]
    payload = {
        "schema": "torch_tm_flowpipe.huan_proof_kernel_audit/2",
        "cuda_kernel_available": True,
        "d2": {"rows": rows, "checked": 2, "passed": 2},
    }
    errors = VERIFIER.verify_d2_route_evidence(payload, "cuda")
    assert "D2 cuda row 2 lacks actual invocation" in errors
    assert "D2 CUDA custom route lacks nonzero invocation evidence" in errors


def test_altered_phase_d_gate_is_rejected() -> None:
    names = {
        "D1_ELEMENTWISE_NO_FTZ_NONFINITE",
        "D2_ANY_ORDER_ACTUAL_ROUTES",
        "D3_DENSE_SPARSE_SUPPORT",
        "D4_CHUNK_LANE",
        "D5_REFINEMENT_LEDGER_CACHE",
        "D6_STRICT_ROUNDOFF",
    }
    gate = {
        "schema": "torch_tm_flowpipe.huan_scientific_gate/2",
        "engine_head": VERIFIER.HUAN_HEAD,
        "overall_gate_passed": True,
        "phase_e_authorized": True,
        "gates": {name: {"passed": True} for name in names},
    }
    assert VERIFIER.verify_phase_d(gate) == []
    gate["gates"]["D6_STRICT_ROUNDOFF"]["passed"] = False
    assert "Phase-D contains an altered or failed D gate" in VERIFIER.verify_phase_d(gate)


def test_stale_cache_evidence_tamper_is_rejected() -> None:
    payload = {
        "behavioral_passed": True,
        "contract_gate_passed": True,
        "cache_freshness": {
            "stale_generation_rejected": True,
            "stale_owner_rejected": True,
        },
    }
    assert VERIFIER.verify_refinement_evidence(payload, "cpu") == []
    payload["cache_freshness"]["stale_owner_rejected"] = False
    assert VERIFIER.verify_refinement_evidence(payload, "cpu") == [
        "D5 cpu stale cache evidence missing"
    ]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _phase_e_fixture(root: Path) -> None:
    vdp = root / "vdp"
    vdp.mkdir()
    (vdp / "huan_final").mkdir()
    (vdp / "huan_final/run_index.json").write_text(
        json.dumps({"engine_head": VERIFIER.HUAN_HEAD, "primary_status": VERIFIER.PRIMARY})
    )
    (vdp / "superseded_runs.json").write_text(
        json.dumps(
            {
                "authoritative_huan_sha": VERIFIER.HUAN_HEAD,
                "superseded": [{"eligible_for_final_claims": False}],
            }
        )
    )
    (vdp / "phase_e_decision.json").write_text(
        json.dumps(
            {
                "primary_status": VERIFIER.PRIMARY,
                "huan_fixed_runs_executed": 8,
                "huan_fixed_runs_completed": 8,
                "contract_was_changed": False,
                "native_parity": "NOT_RUN_CONTRACT_NOT_PORTABLE",
                "native_strict": "NOT_RUN_CONTRACT_NOT_PORTABLE",
            }
        )
    )
    blank = {
        "completed_horizon": "",
        "accepted_steps": "",
        "rejected_attempts": "",
        "refinement_iterations": "",
        "status_code": "",
        "endpoint_x": "",
        "endpoint_y": "",
        "endpoint_x_width": "",
        "endpoint_y_width": "",
        "segment_tube_x": "",
        "segment_tube_y": "",
        "segment_tube_x_width": "",
        "segment_tube_y_width": "",
        "runtime_s": "",
        "peak_gpu_memory_bytes": "",
    }
    rows = []
    for lane in ("huan_parity", "huan_strict"):
        for scenario in ("step1", "fixed_T1", "fixed_T3", "fixed_T6p32"):
            rows.append(
                {
                    "lane": lane,
                    "scenario": scenario,
                    "execution_status": "EXECUTED",
                    "completed_requested_horizon": "True",
                    **{key: "1" for key in blank},
                }
            )
    for lane in ("stock_flowstar", "torch_c2"):
        for scenario in ("step1", "fixed_T1", "fixed_T3", "fixed_T6p32"):
            rows.append(
                {
                    "lane": lane,
                    "scenario": scenario,
                    "execution_status": VERIFIER.STOP,
                    "completed_requested_horizon": "False",
                    **blank,
                }
            )
    _write_csv(vdp / "fixed_horizon_matrix.csv", rows)
    (vdp / "native_terminal.json").write_text(
        json.dumps(
            {
                "huan_parity": {"status": "NOT_RUN_CONTRACT_NOT_PORTABLE"},
                "huan_strict": {"status": "NOT_RUN_CONTRACT_NOT_PORTABLE"},
            }
        )
    )
    (vdp / "first_divergence.json").write_text(
        json.dumps(
            {
                "status": "NOT_ADJUDICATED_CONTRACT_PORTABILITY_STOP",
                "huan_vs_flowstar": None,
                "huan_vs_torch_c2": None,
            }
        )
    )


def test_fabricated_phase_e_value_after_stop_is_rejected(tmp_path: Path) -> None:
    _phase_e_fixture(tmp_path)
    assert VERIFIER.verify_phase_e(tmp_path) == []
    rows = VERIFIER._csv(tmp_path / "vdp/fixed_horizon_matrix.csv")
    stopped = next(row for row in rows if row["lane"] == "stock_flowstar")
    stopped["endpoint_x_width"] = "0.1"
    _write_csv(tmp_path / "vdp/fixed_horizon_matrix.csv", rows)
    assert "stopped Flow*/Torch row contains fabricated scientific data" in VERIFIER.verify_phase_e(tmp_path)


def test_checksum_tamper_and_uncovered_file_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n")
    digest = VERIFIER._sha256(tmp_path / "a.txt")
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  a.txt\n")
    assert VERIFIER.verify_checksums(tmp_path) == []
    (tmp_path / "a.txt").write_text("tampered\n")
    (tmp_path / "extra.txt").write_text("extra\n")
    assert VERIFIER.verify_checksums(tmp_path) == [
        "checksum mismatch: a.txt",
        "uncovered file: extra.txt",
    ]
