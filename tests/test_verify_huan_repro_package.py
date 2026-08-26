from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_huan_repro_package.py"
SPEC = importlib.util.spec_from_file_location("verify_huan_repro_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def test_committed_huan_package_verifies() -> None:
    errors = VERIFIER.verify(ROOT, ROOT / "outputs" / "huan_repro_audit")
    assert errors == [
        "D2 cpu evidence lacks route-tagged schema v2",
        "D2 cuda evidence lacks route-tagged schema v2",
    ]


def test_cuda_availability_cannot_substitute_for_invocation() -> None:
    row = {
        "schedule_name": "engine_interval_dot",
        "execution_backend": "custom_cuda",
        "actual_device": "cuda:0",
        "kernel_path": "flowstar_gpu.interval.dot_point_iv",
        "kernel_invocation_observed": False,
        "custom_cuda_invocation_count": 0,
        "m": 2,
        "finite_hypotheses_satisfied": True,
        "m_u_gate": True,
        "exact_error": None,
        "computed_inflation": 1e-15,
        "containment": True,
        "status": "PASS",
    }
    payload = {
        "schema": "torch_tm_flowpipe.huan_proof_kernel_audit/2",
        "cuda_kernel_available": True,
        "d2": {"rows": [row], "checked": 1, "passed": 1},
    }
    errors = VERIFIER.verify_d2_route_evidence(payload, "cuda")
    assert "D2 CUDA custom route lacks nonzero invocation evidence" in errors


def test_capture_header_requires_delimiter_and_parses_json(tmp_path: Path) -> None:
    capture = tmp_path / "capture.log"
    capture.write_text('{"returncode": 0}\n--- combined stdout/stderr ---\nok\n')
    header, body = VERIFIER.capture_header(capture)
    assert header == {"returncode": 0}
    assert body == "ok\n"

    capture.write_text(json.dumps({"returncode": 0}))
    try:
        VERIFIER.capture_header(capture)
    except ValueError as error:
        assert "delimiter missing" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("missing delimiter was accepted")


def test_checksum_verifier_rejects_tamper_and_uncovered_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a\n")
    digest = VERIFIER._sha256(tmp_path / "a.txt")
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  a.txt\n")
    assert VERIFIER.verify_checksums(tmp_path) == []
    (tmp_path / "a.txt").write_text("changed\n")
    (tmp_path / "b.txt").write_text("b\n")
    assert VERIFIER.verify_checksums(tmp_path) == [
        "checksum mismatch: a.txt",
        "uncovered file: b.txt",
    ]
