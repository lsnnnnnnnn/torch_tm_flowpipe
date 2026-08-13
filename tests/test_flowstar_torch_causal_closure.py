from __future__ import annotations

import math
from pathlib import Path
import hashlib

import pytest

from experiments.run_vdp_dense_backend import parse_args as parse_vdp_args
from experiments.verify_flowstar_torch_causal_closure_package import (
    verify_checksums,
    verify_claims,
    verify_exact_claim,
)
from torch_tm_flowpipe.lossless_state_queue_schema import (
    SCHEMA,
    decode_binary64_exact,
    encode_binary64,
    iter_canonical_dyadics,
    parse_file,
    parse_records,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "outputs/flowstar_torch_causal_mechanism_closure_20260813"
    / "20260813T060020Z"
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.1,
        -123456789.25,
        math.ulp(0.0),
        float.fromhex("0x1p-1022"),
        float.fromhex("0x1.fffffffffffffp+1023"),
    ],
)
def test_canonical_binary64_roundtrip_is_bit_exact(value: float) -> None:
    encoded = encode_binary64(value)
    decoded = decode_binary64_exact(encoded)
    assert decoded.hex() == value.hex()
    assert encode_binary64(decoded) == encoded


@pytest.mark.unit
@pytest.mark.parametrize(
    "encoded",
    [
        "nan",
        "53:0:1:0",
        "52:1:10000000000000:-52",
        "53:1:-1:0",
        "53:1:10000000000000:999999999999999999999",
    ],
)
def test_canonical_binary64_decoder_rejects_noncanonical_values(encoded: str) -> None:
    with pytest.raises(ValueError):
        decode_binary64_exact(encoded)


@pytest.mark.protocol
def test_schema_record_parser_rejects_missing_duplicate_and_unterminated() -> None:
    valid = f"schema={SCHEMA}\nvalue=1\n"
    assert parse_records(valid)["value"] == "1"
    with pytest.raises(ValueError, match="duplicate"):
        parse_records(f"schema={SCHEMA}\nschema={SCHEMA}\n")
    with pytest.raises(ValueError, match="newline-terminated"):
        parse_records(f"schema={SCHEMA}")
    with pytest.raises(ValueError, match="wrong or missing schema"):
        parse_records("schema=wrong\n")


@pytest.mark.regression
def test_committed_lossless_flowstar_fixture_is_fully_binary64_exact() -> None:
    fixture = (
        PACKAGE
        / "09_lossless_schema_roundtrip/flowstar_fixtures_retry/artifacts/fixtures"
        / "step_100_pre_reset.state"
    )
    records = parse_file(fixture)
    dyadics = list(iter_canonical_dyadics(records))
    assert int(records["state_dimension"]) == 3
    assert int(records["variable_dimension"]) == 4
    assert int(records["queue.J_count"]) == 100
    assert int(records["queue.Phi_L_count"]) == 100
    assert len(dyadics) > 600
    assert all(encode_binary64(decode_binary64_exact(value)) == value for _, value in dyadics)


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode",
    [
        "normalized_insertion",
        "normalized_insertion_horner",
        "normalized_insertion_symqueue_v2",
        "normalized_insertion_horner_symqueue_v2",
    ],
)
def test_vdp_runner_accepts_all_preregistered_factorial_modes(
    tmp_path: Path, mode: str
) -> None:
    args = parse_vdp_args(
        [
            "--output-dir",
            str(tmp_path / mode),
            "--tm-backend",
            "dense",
            "--reset-mode",
            mode,
        ]
    )
    assert args.reset_mode == mode


@pytest.mark.regression
def test_committed_gate_d_and_e_results_do_not_conflate_bridge_with_operator_matrix() -> None:
    bridge = parse_file(
        PACKAGE
        / "09_lossless_schema_roundtrip/cross_language_retry2/artifacts/audit"
        / "torch_initial.state"
    )
    assert bridge["producer"] == "torch_binary64"
    import json

    gate_d = json.loads(
        (
            PACKAGE
            / "09_lossless_schema_roundtrip/flowstar_fixtures_retry/artifacts/fixtures"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    gate_e = json.loads(
        (
            PACKAGE
            / "10_same_prestate_2x2/operator_matrix/artifacts/matrix/summary.json"
        ).read_text(encoding="utf-8")
    )
    assert gate_d["status"] == "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE"
    assert gate_e["gate_d_lossless_serialization_roundtrip_closed"] is True
    assert gate_e["full_two_by_two_same_prestate_executed"] is False
    assert gate_e["operator_attribution_closed"] is False
    assert gate_e["queue_dropped"] is False
    assert gate_e["downstream_authorization"] == "NO_FIX_AUTHORIZED"


@pytest.mark.protocol
def test_causal_verifier_rejects_tampered_number_and_status() -> None:
    verify_exact_claim("accepted steps", 632, 632)
    with pytest.raises(ValueError, match="not raw-derived"):
        verify_exact_claim("accepted steps", 633, 632)
    statuses = ["BASELINE_CONCLUSIONS_REPRODUCED", "NO_FIX_AUTHORIZED"]
    verification = {
        "scientific_statuses": list(statuses),
        "scientific_outcome_uses_process_exit_code": False,
    }
    manifest = {"scientific_statuses": list(statuses)}
    verify_claims(statuses, verification, manifest)
    verification["scientific_statuses"] = ["CAUSAL_SOURCE_DELTA_CLOSED"]
    with pytest.raises(ValueError, match="status"):
        verify_claims(statuses, verification, manifest)


@pytest.mark.protocol
def test_causal_verifier_rejects_deleted_file_and_wrong_sha(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("scientific raw\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{digest}  payload.txt\n", encoding="utf-8")
    assert verify_checksums(tmp_path) == 1
    sums.write_text(f"{'0' * 64}  payload.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksums(tmp_path)
    sums.write_text(f"{digest}  payload.txt\n", encoding="utf-8")
    payload.unlink()
    with pytest.raises(ValueError, match="missing checksummed file"):
        verify_checksums(tmp_path)
