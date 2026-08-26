#!/usr/bin/env python3
"""Rerun Huan plant-engine D1--D6; this is the scientific verifier.

Unlike the package verifier, this program imports the selected engine source,
executes exact oracles and production routes, and launches the targeted pytest
batteries.  A passing JSON result is the sole authorization for frozen Phase E.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import huan_chunk_boundary_audit as chunk_audit
import huan_proof_kernel_audit as kernel_audit
import huan_refinement_boundary_audit as refinement_audit
import huan_strict_roundoff_audit as strict_audit


TESTS: dict[str, tuple[str, ...]] = {
    "D3_DENSE_SPARSE_SUPPORT": (
        "tests/unit/test_sparse_support.py",
        "tests/unit/test_sparse_exec.py",
        "tests/soundness/test_sparse_containment.py",
    ),
    "D4_CHUNK_LANE": (
        "tests/unit/test_cuda_kernels.py",
        "tests/adversarial/test_batch_invariance.py",
        "tests/unit/test_safety.py::test_spatial_images_member_chunking_is_bitwise_neutral",
        "tests/unit/test_flowpipe.py::test_reach_mixed_fate_batch",
        "tests/unit/test_sparse_exec.py::test_advance_sparse_div_rhs_bad_channel_plumbed",
    ),
    "D5_REFINEMENT": (
        "tests/unit/test_flowpipe.py",
        "tests/unit/test_tape_kernels.py",
        "tests/unit/test_coverage_edges.py::test_refine_graph_cap_exit_and_diagnostics_ring",
        "tests/soundness/test_strict_proof_contract.py",
    ),
    "D6_STRICT_ROUNDOFF": (
        "tests/unit/test_composition.py",
        "tests/unit/test_sparse_exec.py::test_advance_sparse_vs_dense",
        "tests/unit/test_sparse_exec.py::test_advance_sparse_crosses_sr_queue_resets_like_dense",
        "tests/soundness/test_strict_proof_contract.py",
    ),
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def _pytest(
    engine_root: Path,
    python: Path,
    selectors: tuple[str, ...],
) -> dict[str, Any]:
    command = [str(python), "-m", "pytest", "-q", *selectors]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(engine_root / "src")
    result = subprocess.run(
        command,
        cwd=engine_root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    encoded = result.stdout.encode()
    return {
        "command": command,
        "cwd": str(engine_root),
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(encoded).hexdigest(),
        "stdout": result.stdout,
    }


def run(engine_root: Path, python: Path, expected_head: str) -> dict[str, Any]:
    head = _git(engine_root, "rev-parse", "HEAD")
    dirty = bool(_git(engine_root, "status", "--porcelain"))
    if head != expected_head or dirty:
        raise RuntimeError(
            f"engine provenance mismatch: head={head}, expected={expected_head}, dirty={dirty}"
        )

    d1_d2 = {device: kernel_audit.run(engine_root, device) for device in ("cpu", "cuda")}
    d1_pass = all(
        row["d1"]["passed"] == row["d1"]["case_count"]
        and row["d1"]["no_ftz_observed"]
        and row["production_no_ftz_assertion_passed"]
        for row in d1_d2.values()
    )
    cuda_custom = [
        row for row in d1_d2["cuda"]["d2"]["rows"]
        if row["execution_backend"] == "custom_cuda"
    ]
    d2_pass = all(not row["d2"]["failures"] for row in d1_d2.values()) and bool(
        cuda_custom
    ) and all(
        row["kernel_invocation_observed"]
        and row.get("custom_cuda_invocation_count", 0) > 0
        for row in cuda_custom
    )

    pytest_results = {
        name: _pytest(engine_root, python, selectors)
        for name, selectors in TESTS.items()
    }
    chunk = {
        device: chunk_audit.run(engine_root, device) for device in ("cpu", "cuda")
    }
    refinement = {
        device: refinement_audit.run(engine_root, device)
        for device in ("cpu", "cuda")
    }
    witness = strict_audit.run(engine_root)

    gates = {
        "D1_ELEMENTWISE_NO_FTZ_NONFINITE": {
            "passed": d1_pass,
            "evidence": {
                device: {
                    "d1": data["d1"],
                    "production_no_ftz_assertion_passed": data[
                        "production_no_ftz_assertion_passed"
                    ],
                }
                for device, data in d1_d2.items()
            },
        },
        "D2_ANY_ORDER_ACTUAL_ROUTES": {
            "passed": d2_pass,
            "evidence": {
                device: {
                    "checked": data["d2"]["checked"],
                    "passed": data["d2"]["passed"],
                    "route_counts": data["d2"]["route_counts"],
                    "kernel_invocation_counts": data["d2"][
                        "kernel_invocation_counts"
                    ],
                }
                for device, data in d1_d2.items()
            },
        },
        "D3_DENSE_SPARSE_SUPPORT": {
            "passed": pytest_results["D3_DENSE_SPARSE_SUPPORT"]["returncode"] == 0,
            "evidence": pytest_results["D3_DENSE_SPARSE_SUPPORT"],
        },
        "D4_CHUNK_LANE": {
            "passed": (
                pytest_results["D4_CHUNK_LANE"]["returncode"] == 0
                and all(row["gate_passed"] for row in chunk.values())
            ),
            "evidence": {
                "pytest": pytest_results["D4_CHUNK_LANE"],
                "chunk_audits": chunk,
            },
        },
        "D5_REFINEMENT_LEDGER_CACHE": {
            "passed": (
                pytest_results["D5_REFINEMENT"]["returncode"] == 0
                and all(row["contract_gate_passed"] for row in refinement.values())
            ),
            "evidence": {
                "pytest": pytest_results["D5_REFINEMENT"],
                "refinement_audits": refinement,
            },
        },
        "D6_STRICT_ROUNDOFF": {
            "passed": (
                pytest_results["D6_STRICT_ROUNDOFF"]["returncode"] == 0
                and witness["classification"]
                == "D6_REPAIRED_STRICT_CONTAINS_MINIMAL_WITNESS"
            ),
            "evidence": {
                "pytest": pytest_results["D6_STRICT_ROUNDOFF"],
                "minimal_witness": witness,
                "parity_trust_model": (
                    "Flow* point-coefficient arithmetic retained for reproduction; "
                    "not promoted to the strict soundness claim"
                ),
            },
        },
    }
    passed = all(gate["passed"] for gate in gates.values())
    return {
        "schema": "torch_tm_flowpipe.huan_scientific_gate/2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "scientific rerun: imports engine and executes D1-D6",
        "engine_root": str(engine_root),
        "engine_head": head,
        "engine_clean": not dirty,
        "python": str(python),
        "gates": gates,
        "overall_gate_passed": passed,
        "phase_e_authorized": passed,
        "primary_status": (
            "HUAN_STRICT_PROOF_CONTRACT_CLOSED__PHASE_E_AUTHORIZED"
            if passed
            else "HUAN_STRICT_PROOF_CONTRACT_REMAINS_OPEN"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.engine_root.resolve(), args.python.resolve(), args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "overall_gate_passed": payload["overall_gate_passed"],
                "phase_e_authorized": payload["phase_e_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["overall_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
