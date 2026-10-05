#!/usr/bin/env python3
"""Index saved new-candidate receipts; never run a solver or old checker."""

import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def main():
    root = Path(__file__).resolve().parent
    rows = []
    for run in sorted((root / "results").glob("*/*_001")):
        if run.name not in ("run_001", "gate_001"):
            continue
        outer = read(run / "RESULT.json")
        candidate = run / ("data" if run.name == "gate_001" else "candidate")
        result = read(candidate / "RESULT.json")
        data = candidate if run.name == "gate_001" else candidate / "data"
        metrics = read(data / "metrics.json")
        comparison = read(candidate / "SAVED_COMPARISON.json")
        rows.append({
            "run_id": run.parent.name, "stage": run.name,
            "status": result.get("status", "missing_result"),
            "outer_status": outer.get("status"),
            "completed_substeps": result.get("completed_substeps"),
            "boxes": metrics.get("B"),
            "outer_wall_s": outer.get("wall_s"),
            "wrapper_wall_s": result.get("wall_s"),
            "payload_wall_s": read(data / "RESULT.json").get("wall_s"),
            "driver_elapsed_s": metrics.get("elapsed_s"),
            "saved_comparison": str((candidate / "SAVED_COMPARISON.json").relative_to(root))
                if comparison else None,
            "comparison_scope": comparison.get("scope"),
            "raw_result": str((run / "RESULT.json").relative_to(root)),
        })
    if not rows:
        raise RuntimeError("no saved new-candidate receipts")
    summary = {
        "scope": "New implementation candidates only; the old 289-attempt index is unchanged",
        "timing": "Single processes; wrapper/payload/startup boundaries differ; use each START. Concurrent GPU jobs are not isolated timings.",
        "qualification": "Saved-output equality is not hidden TM/SR equality or an independent NNCS certificate. Width comparisons remain separate.",
        "digest_operations": 0,
        "runs": rows,
    }
    (root / "RUN_INDEX.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (root / "RUN_INDEX.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Indexed {len(rows)} saved candidate stages; no experiment or checker executed.")


if __name__ == "__main__":
    main()
