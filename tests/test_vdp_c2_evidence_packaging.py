from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/package_vdp_c2_evidence_20260820.py"
SPEC = importlib.util.spec_from_file_location("vdp_c2_evidence_packaging", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PACKAGING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGING)


def test_raw_run_packaging_retains_ledgers_and_discloses_trace_exclusions(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    retained = (
        "attempts.csv",
        "segments.csv",
        "remainder_ledger.jsonl",
        "refinement_ledger.jsonl",
        "summary.json",
    )
    for name in retained + PACKAGING.RAW_RUN_EXCLUDED_FILES:
        (source / name).write_text(name + "\n", encoding="utf-8")
    destination = tmp_path / "destination"
    PACKAGING._copy_raw_runs(source, destination)
    assert {path.name for path in destination.iterdir()} == set(retained)
