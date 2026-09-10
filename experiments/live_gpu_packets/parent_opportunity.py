"""Reconstruct ready-key fragmentation from frozen parent lifecycle records.

This is a static opportunity measurement.  It never loads answers into a
solver, and its per-dispatch projections overlap; they are not counterfactual
online timings or globally summable savings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "artifacts/runs/live_range_solver_20260909T053007Z"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values)-1)*fraction)] if values else None


def analyze_lifecycle(path, plant, batch):
    ready = {}
    snapshots = {}
    selected = Counter()
    with gzip.open(path, "rt") as stream:
        for line in stream:
            event = json.loads(line)
            kind = event["event"]
            if kind == "submit":
                ready[event["request_id"]] = event
            elif kind == "dispatch":
                group = event["group"]
                if group not in snapshots:
                    ordered = sorted(ready.values(), key=lambda row: row["submitted_ns"])
                    projected = ordered[:32]
                    selected_key = digest(ready[event["request_id"]]["structure"])
                    key_sizes = Counter(digest(row["structure"]) for row in ordered)
                    snapshots[group] = dict(
                        plant=plant, batch=batch, parent_group=group,
                        reason=event["reason"], ready_requests=len(ordered),
                        ready_keys=len(key_sizes), selected_key=selected_key,
                        selected_key_ready_requests=key_sizes[selected_key],
                        unavailable_tasks=batch-len(ordered),
                        projected_packet_requests=len(projected),
                        projected_packet_keys=len({digest(row["structure"]) for row in projected}),
                    )
                assert digest(ready[event["request_id"]]["structure"]) == snapshots[group]["selected_key"]
                selected[group] += 1
                assert ready.pop(event["request_id"])["request_id"] == event["request_id"]
    assert not ready
    rows = []
    for group in sorted(snapshots):
        row = snapshots[group]
        row["parent_selected_requests"] = selected[group]
        row["other_key_ready_requests"] = (
            row["ready_requests"] - row["selected_key_ready_requests"])
        keys = row["projected_packet_keys"]
        row["projected_device_submissions_saved"] = max(0, keys-1)
        row["projected_kernel_launches_saved"] = 4*max(0, keys-1)
        row["projected_h2d_copy_operations_saved"] = max(0, 8*keys-2)
        row["projected_d2h_copy_operations_saved"] = max(0, 4*keys-2)
        row["projected_hot_device_allocations_saved"] = 16*keys
        row["classification"] = ("KEY_SPLIT" if row["other_key_ready_requests"] > 0
            else "ARRIVAL_LIMITED" if row["unavailable_tasks"] > 0 else "ONE_READY_KEY")
        rows.append(row)
    return rows


def summarize(rows):
    output = []
    for plant in ("van_der_pol", "brusselator"):
        for batch in (8, 32):
            chosen = [row for row in rows if row["plant"] == plant and row["batch"] == batch]
            classifications = Counter(row["classification"] for row in chosen)
            ready = [row["ready_requests"] for row in chosen]
            keys = [row["ready_keys"] for row in chosen]
            selected = [row["parent_selected_requests"] for row in chosen]
            projected_keys = [row["projected_packet_keys"] for row in chosen]
            output.append(dict(plant=plant, batch=batch, dispatches=len(chosen),
                requests=sum(selected), parent_selected_mean=sum(selected)/len(selected),
                parent_selected_max=max(selected), ready_p50=statistics.median(ready),
                ready_p95=percentile(ready, .95), ready_max=max(ready),
                ready_keys_p50=statistics.median(keys), ready_keys_p95=percentile(keys, .95),
                ready_keys_max=max(keys), projected_packet_keys_p50=statistics.median(projected_keys),
                key_split_dispatches=classifications["KEY_SPLIT"],
                key_split_fraction=classifications["KEY_SPLIT"]/len(chosen),
                arrival_limited_dispatches=classifications["ARRIVAL_LIMITED"],
                one_ready_key_dispatches=classifications["ONE_READY_KEY"]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    sources = {}
    for plant in ("van_der_pol", "brusselator"):
        for batch in (8, 32):
            path = PARENT / "diagnostic" / f"prefix-b{batch}-{plant}-G" / "lifecycle.jsonl.gz"
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.extend(analyze_lifecycle(path, plant, batch))
    with (args.output / "ready_packet_opportunity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = dict(schema="live-gpu-packet-parent-opportunity-v1",
        evidence="REUSED_PARENT_LIFECYCLE_STATIC_ANALYSIS",
        projected_not_measured_online=True, snapshots_overlap_and_are_not_summable=True,
        parent_max_group=32, parent_h2d_operations_per_semantic_group=8,
        parent_d2h_operations_per_semantic_group_nondiagnostic=4,
        parent_device_allocations_per_semantic_group=16,
        packet_projection_h2d_operations=2, packet_projection_d2h_operations=2,
        packet_projection_device_allocations_hot=0, sources=sources, summary=summarize(rows))
    (args.output / "PARENT_OPPORTUNITY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
