"""The reference stop and direct tampering, even after recomputing outer hashes."""
import csv
import json
from pathlib import Path
import shutil

import pytest

from experiments.our_solver_performance.verify_evidence import STOP, checksum, verify

ART = Path(__file__).resolve().parents[1] / "artifacts/runs/our_solver_replay_performance_20260907T074937Z"


def test_reference_stop_evidence():
    assert verify(ART, check_hashes=False, check_tests=False)["decision"] == STOP


@pytest.mark.parametrize("kind", ["interval_endpoint", "replay_decision", "source", "runtime", "final_state"])
def test_reference_stop_rehashed_tampering_rejected(tmp_path, kind):
    root = tmp_path / "evidence"
    shutil.copytree(ART, root)
    if kind == "runtime":
        path = root / "full_timings.csv"
        with path.open() as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]["solve_seconds"] = float(rows[0]["solve_seconds"]) / 2
        with path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        relative = {"interval_endpoint": "raw_minimal/endpoint_reference.json",
                    "replay_decision": "optimization_decision.json", "source": "SOURCE_MAP.json",
                    "final_state": "RESULT.json"}[kind]
        path = root / relative
        data = json.loads(path.read_text())
        if kind == "interval_endpoint":
            data["rows"][0]["full_range_hex"][0][1] = (1e-16).hex()
        elif kind == "replay_decision":
            data["replay_decision"] = "ACCEPTED_WITH_FEWER_REPLAYS"
        elif kind == "source":
            data["base_sha"] = "4939fb288c941a67f55cc191f4d75f8594692f47"
        else:
            data["status"] = "OUR_SOLVER_EXECUTION_OPTIMIZED__SEMANTICS_PRESERVED"
        path.write_text(json.dumps(data))
    (root / "SHA256SUMS").write_text(checksum(root))
    with pytest.raises(ValueError):
        verify(root, check_tests=False)
