from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARE_PATH = ROOT / "scripts/native_reproduction/compare_native_json.py"
SPEC = importlib.util.spec_from_file_location("compare_native_json", COMPARE_PATH)
assert SPEC is not None and SPEC.loader is not None
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)

CSV_COMPARE_PATH = ROOT / "scripts/native_reproduction/compare_native_csv.py"
CSV_SPEC = importlib.util.spec_from_file_location("compare_native_csv", CSV_COMPARE_PATH)
assert CSV_SPEC is not None and CSV_SPEC.loader is not None
CSV_COMPARE = importlib.util.module_from_spec(CSV_SPEC)
CSV_SPEC.loader.exec_module(CSV_COMPARE)


def test_explicit_field_lookup_never_falls_back_between_result_scopes() -> None:
    payload = {
        "endpoint": {"lo": [1.0]},
        "tube": {"hi": [2.0]},
        "segments": [{"lo": [3.0]}],
    }
    assert COMPARE.field_at(payload, "endpoint.lo") == [1.0]
    assert COMPARE.field_at(payload, "endpoint.hi") is COMPARE.MISSING
    assert COMPARE.field_at(payload, "tube.lo") is COMPARE.MISSING
    assert COMPARE.field_at(payload, "segment.lo") is COMPARE.MISSING


def test_csv_loader_requires_explicit_unique_key(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("mode,status\na,failed\n", encoding="utf-8")
    try:
        CSV_COMPARE.load(missing, "run_kind")
    except KeyError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("missing key must fail closed")

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("mode,status\na,failed\na,timeout\n", encoding="utf-8")
    try:
        CSV_COMPARE.load(duplicate, "mode")
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("duplicate key must fail closed")
