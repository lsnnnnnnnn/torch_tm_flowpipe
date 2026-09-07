"""Exact stop decision and direct tampering after regenerating outer hashes."""
import csv
import json
from pathlib import Path
import shutil

import pytest

from experiments.backend_reevaluation.verify_evidence import STOP, checksum, verify

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/runs/backend_reevaluation_20260907T061907Z"


def test_exact_crosscheck_and_independent_stop():
    result = verify(ART, check_hashes=False)
    assert result["decision"] == STOP
    assert result["exact_map_rows"] == 51
    assert result["independent_endpoint_rows"] == 8


@pytest.mark.parametrize("kind", ["input", "output", "result", "mode", "candidate_sha", "timing", "endpoint_input", "decision"])
def test_rehashed_tampering_is_rejected(tmp_path, kind):
    root = tmp_path / "evidence"
    shutil.copytree(ART, root)
    if kind == "timing":
        path = root / "timings_raw.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["solve_seconds"] = str(float(rows[0]["solve_seconds"]) / 5)
        with path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    else:
        path = root / ("RESULT.json" if kind == "decision" else
                       "raw_minimal/endpoint_A.json" if kind == "endpoint_input" else "raw_minimal/A.json")
        data = json.loads(path.read_text())
        if kind == "input":
            data["rows"][0]["matrices"][1][0][0] = 1.
        elif kind == "output":
            row = next(r for r in data["rows"] if r["witness"] == "normalization")
            row["retained_coefficients"][0][1] = 1.
        elif kind == "result":
            data["rows"][0]["contains"] = True
        elif kind == "mode":
            data["rows"][0]["actual_mode"] = "strict"
        elif kind == "candidate_sha":
            data["source_sha"] = "743f6205e6408072193ad76e940e7f15030e8d3c"
        elif kind == "endpoint_input":
            data["rows"][0]["h"] = .02
        else:
            data["status"] = "ADOPT_REPAIRED_OPTIONAL_BACKEND_FOR_MATCHED_FIXED_PLANTS"
        path.write_text(json.dumps(data))
    (root / "SHA256SUMS").write_text(checksum(root))
    with pytest.raises(ValueError):
        verify(root)
