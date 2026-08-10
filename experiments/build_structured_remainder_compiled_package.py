#!/usr/bin/env python3
"""Build the derived package for the compiled/outward/S1 closure round."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import pstats
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PREFIXES = ("raw_public/", "figures/")
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty table: {path.name}")
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(row.get(key), sort_keys=True, allow_nan=False)
                if isinstance(row.get(key), (dict, list, tuple))
                else row.get(key, "")
                for key in fields
            })


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _stable_warm(values: Sequence[float]) -> list[float]:
    values = [float(value) for value in values]
    if len(values) > 1 and values[0] > 5.0 * min(values[1:]):
        return values[1:]
    return values


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    return ordered[len(ordered) // 2]


def _eligibility(
    *, completed: bool, certificate: bool, finite: bool, soundness: str, scope: str,
    formal: bool, performance: bool, ranking: bool = False,
) -> dict[str, Any]:
    return {
        "mathematical_contract_known": True,
        "requested_horizon_completed": completed,
        "certificate_semantics_passed": certificate,
        "finite_outputs": finite,
        "numerical_soundness_class": soundness,
        "numerical_soundness_scope": scope,
        "formal_claim_eligible": formal,
        "performance_measurement_eligible": performance,
        "cross_tool_ranking_eligible": ranking,
    }


def _copy_public_raw(run_root: Path) -> None:
    source = run_root / "raw"
    destination = run_root / "raw_public"
    if destination.exists():
        shutil.rmtree(destination)
    selections = (
        "fixed_object_baseline/trace_committed_4bb",
        "fixed_object_baseline/cpu_b64_t0p1.prof",
        "fixed_object_baseline/cpu_b64_t0p1_profile.txt",
        "fixed_object_current/cpu_b1_t10",
        "fixed_object_current/cpu_b1_t10_stdout.jsonl",
        "fixed_object_current/cpu_b1_t10_stderr.txt",
        "fixed_compiled",
        "fixed_functional",
        "fixed_outward",
        "structured_terminal",
        "structured_terminal_baseline_stdout.txt",
        "structured_terminal_baseline_stderr.txt",
        "structured_terminal_s1_stdout.jsonl",
        "structured_terminal_s1_stderr.txt",
        "second_system",
        "compiled_boundaries",
    )
    for relative in selections:
        item = source / relative
        if not item.exists():
            if relative == "compiled_boundaries":
                continue
            raise FileNotFoundError(item)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    replacements = {
        "/srv/local/shengenli/torch_tm_flowpipe_structured_remainder": "<server-workspace>/torch_tm_flowpipe_structured_remainder",
        "/srv/local/shengenli/torch_tm_flowpipe_fixed_object_baseline_4bb10d5": "<server-workspace>/torch_tm_flowpipe_fixed_object_baseline_4bb10d5",
        "/srv/local/shengenli/miniforge3/envs/py11": "<py11-environment>",
        "/srv/local/shengenli": "<server-workspace>",
        "primitive / reference multi-step lane": "multi-step lane",
    }
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    for manifest_path in destination.rglob("artifact_manifest.json"):
        files = []
        for path in sorted(manifest_path.parent.iterdir()):
            if path.is_file() and path != manifest_path:
                files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)})
        _write_json(manifest_path, {"schema": "torch_tm_flowpipe_artifact_manifest_v1", "files": files})


def _profile(run_root: Path) -> dict[str, Any]:
    path = run_root / "raw/fixed_object_baseline/cpu_b64_t0p1.prof"
    stats = pstats.Stats(str(path)).stats
    wanted = (
        "fixed_support_dr_remainder_picard", "diffreach_vdp_tm_rhs", "mul_ctrunc",
        "mul_trunc", "range", "step_once", "verify",
    )
    rows = []
    for needle in wanted:
        matches = [
            (key, values) for key, values in stats.items()
            if needle in key[2]
        ]
        rows.append({
            "operator": needle,
            "primitive_calls": sum(values[0] for _, values in matches),
            "total_calls": sum(values[1] for _, values in matches),
            "self_s": sum(values[2] for _, values in matches),
            "cumulative_s": sum(values[3] for _, values in matches),
        })
    compiled_profile_path = run_root / "raw/fixed_compiled/cuda_v100_b64_t10_b1_profile_cache_replay/summary.json"
    compiled_profile = _json(compiled_profile_path).get("profiler_counts") if compiled_profile_path.is_file() else None
    return {
        "schema": "fixed_support_object_profile_v1",
        "profile_source_sha": "4bb10d54b29dad2b47c5f91ddedafe854a52fac6",
        "signature": {"batch": 64, "steps": 10, "device": "cpu", "trace": True},
        "attribution": rows,
        "audit": {
            "pre_refactor_route_plan_rebuilt_in_multiplication": True,
            "pre_refactor_python_route_loops": True,
            "pre_refactor_dataclass_construction_per_operator": True,
            "pre_refactor_host_item_gate_per_step": True,
            "functional_plan_cached": True,
            "functional_state_tensor_count": 26,
            "compiled_solver_core_host_synchronizations": 0,
        },
        "compiled_v100_one_boundary_profiler_counts": compiled_profile,
        "compiled_profile_scope": "one logical boundary after cache replay; not a timing row",
    }


def _flow_rectangles(path: Path) -> list[tuple[float, float, float, float]]:
    rectangles = []
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        try:
            pair = (float(parts[0]), float(parts[1]))
        except ValueError:
            continue
        rows.append(pair)
        if len(rows) == 5:
            xs, ys = zip(*rows)
            rectangles.append((min(xs), max(xs), min(ys), max(ys)))
            rows = []
    return rectangles


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _torch_rectangles(rows: Iterable[Mapping[str, str]], state: str) -> list[tuple[float, float, float, float]]:
    result = []
    for row in rows:
        if row.get("status") != "accepted":
            continue
        try:
            result.append(tuple(float(row[key]) for key in ("t_lo", "t_hi", f"segment_{state}_lo", f"segment_{state}_hi")))
        except (KeyError, ValueError):
            continue
    return result


def _plot(run_root: Path, prior_root: Path, object_rows, compiled_rows, outward_rows, terminal, soundness_rows) -> None:
    figures = run_root / "figures"
    figures.mkdir(exist_ok=True)
    # Runtime versus batch.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for device_group in ("cpu", "cuda_v100"):
        for horizon in (0.1, 1.0):
            rows = sorted(
                (row for row in object_rows if row["device_group"] == device_group and row["requested_horizon"] == horizon),
                key=lambda row: row["batch"],
            )
            ax.plot([row["batch"] for row in rows], [row["warm_median_s"] for row in rows], marker="o", label=f"object {device_group} T{horizon:g}")
    ax.set(xscale="log", yscale="log", xlabel="actual independent partitions", ylabel="warm median seconds", title="Frozen object-eager fixed-support matrix")
    ax.legend(fontsize=7); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures / "fixed_support_runtime_vs_batch.png", dpi=180); plt.close(fig)

    # Runtime breakdown and synchronization counts.
    b64 = [row for row in object_rows if row["batch"] == 64 and row["requested_horizon"] == 10.0]
    labels = [f"object {row['device_group']}" for row in b64] + [f"compiled {row['device_group']}" for row in compiled_rows if row["batch"] == 64]
    warm = [row["warm_median_s"] for row in b64] + [row["stable_warm_median_s"] for row in compiled_rows if row["batch"] == 64]
    compile_time = [0.0] * len(b64) + [row["compile_execute_s"] for row in compiled_rows if row["batch"] == 64]
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = range(len(labels)); ax.bar(x, warm, label="stable warm"); ax.bar(x, compile_time, bottom=warm, alpha=.45, label="compile + first execute"); ax.set_xticks(list(x), labels, rotation=20); ax.set_yscale("log"); ax.set_ylabel("seconds (log)"); ax.set_title("Compile cost is separate from steady execution"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "fixed_support_runtime_breakdown.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8)); sync = [row["host_synchronizations"] for row in b64] + [row["host_synchronizations_in_solver_core"] for row in compiled_rows if row["batch"] == 64]; ax.bar(labels, sync, color="#9467bd"); ax.set_yscale("symlog", linthresh=1); ax.set_ylabel("solver-core host synchronizations"); profile_path = run_root / "raw/fixed_compiled/cuda_v100_b64_t10_b1_profile_cache_replay/summary.json"; kernel_count = _json(profile_path).get("profiler_counts", {}).get("cuda_kernel_events") if profile_path.is_file() else None; ax.set_title("T10 host synchronization reduction"); ax.text(.02, .96, f"compiled V100 kernels per one logical step: {kernel_count if kernel_count is not None else 'not captured'}\nobject kernel launches: not captured", transform=ax.transAxes, va="top", fontsize=8); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(figures / "fixed_support_host_sync_kernel_counts.png", dpi=180); plt.close(fig)

    # Soundness scopes.
    fig, ax = plt.subplots(figsize=(9, 4.8)); labels2 = [row["lane"] for row in soundness_rows]; rank = {"unknown": 0, "empirically sampled only": 1, "safeguarded outward under declared IEEE/backend assumptions": 2, "formally outward by construction": 3, "unsound/ineligible on a demonstrated counterexample": -1}; values = [rank.get(row["numerical_soundness_class"], 0) for row in soundness_rows]; ax.barh(labels2, values); ax.set_xlabel("classification index (scope remains in machine table)"); ax.set_title("Numerical qualification is independent of completion"); fig.tight_layout(); fig.savefig(figures / "fixed_support_soundness_scope.png", dpi=180); plt.close(fig)

    # Terminal margins and decomposition.
    baseline = terminal["terminal"]["baseline_margin"][0]
    local = terminal["terminal"]["local_empty_state_s1"]["ordinary_target_margin"][0]
    fig, ax = plt.subplots(figsize=(7, 4.8)); width=.35; ax.bar([-.2,.8], baseline, width, label="closest baseline"); ax.bar([.2,1.2], local, width, label="local empty-state attribution"); ax.axhline(0, color="black", linewidth=.8); ax.set_xticks([0,1], ["x", "y"]); ax.set_ylabel("target subset margin"); ax.set_title("Terminal local split is not a prefix-qualified candidate"); ax.legend(); fig.tight_layout(); fig.savefig(figures / "complete_o4_margin_at_terminal.png", dpi=180); plt.close(fig)
    sources = terminal["terminal"]["typed_sources"]
    fig, ax = plt.subplots(figsize=(8, 4.8)); bottoms = [0.0,0.0]
    for name in sorted(sources):
        lo, hi = sources[name]["lo"][0], sources[name]["hi"][0]
        widths = [hi[i]-lo[i] for i in range(2)]
        ax.bar([0,1], widths, bottom=bottoms, label=name)
        bottoms = [bottoms[i]+widths[i] for i in range(2)]
    ax.set_xticks([0,1], ["x", "y"]); ax.set_yscale("log"); ax.set_ylabel("terminal additive source width"); ax.set_title("Structured source decomposition at t=6.39708"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(figures / "structured_width_decomposition_vs_time.png", dpi=180); plt.close(fig)

    # Validated horizon lanes.
    horizons = {"complete O4 baseline": 6.397083942944808, "fixed object": 10.0, "fixed compiled empirical": 10.0, "fixed outward": max((row["steps"] * .01 for row in outward_rows if row["completed"]), default=0.0), "S1": 0.0}
    fig, ax = plt.subplots(figsize=(8, 4.8)); ax.bar(list(horizons), list(horizons.values())); ax.set_ylabel("validated horizon"); ax.set_title("In-framework lanes; S1 horizon not run after STOP"); ax.tick_params(axis="x", rotation=20); fig.tight_layout(); fig.savefig(figures / "validated_horizon_by_in_framework_lane.png", dpi=180); plt.close(fig)

    native = prior_root / "01_native_baselines"
    baseline_rows = _csv(native / "torch_complete_o4_authoritative_t6p5/segments.csv")
    flow_x = _flow_rectangles(native / "flowstar_stock_artifacts/vanderpol_t_x.plt")
    flow_y = _flow_rectangles(native / "flowstar_stock_artifacts/vanderpol_t_y.plt")
    torch_x, torch_y = _torch_rectangles(baseline_rows, "x"), _torch_rectangles(baseline_rows, "y")
    for state, flow, torch_rows, filename in (("x", flow_x, torch_x, "flowstar_style_t_x_overlay.png"), ("y", flow_y, torch_y, "flowstar_style_t_y_overlay.png")):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for rectangles, color, label in ((flow, "#1f77b4", "stock Flow* native tube"), (torch_rows, "#ff7f0e", "Torch complete-O4 validated prefix")):
            first=True
            for x0,x1,y0,y1 in rectangles:
                ax.fill_between([x0,x1],[y0,y0],[y1,y1],color=color,alpha=.18,label=label if first else None); first=False
        ax.set(xlabel="physical time", ylabel=state, title=f"VDP t-{state}: explicit tube semantics"); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures / filename, dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for index, (xrow, yrow) in enumerate(zip(flow_x, flow_y)):
        if index % 3 == 0: ax.add_patch(Rectangle((xrow[2],yrow[2]),xrow[3]-xrow[2],yrow[3]-yrow[2],facecolor="#1f77b4",alpha=.08,edgecolor="none"))
    for index, (xrow, yrow) in enumerate(zip(torch_x, torch_y)):
        if index % 3 == 0: ax.add_patch(Rectangle((xrow[2],yrow[2]),xrow[3]-xrow[2],yrow[3]-yrow[2],facecolor="#ff7f0e",alpha=.10,edgecolor="none"))
    ax.plot([],[],color="#1f77b4",linewidth=6,alpha=.25,label="stock Flow* tube"); ax.plot([],[],color="#ff7f0e",linewidth=6,alpha=.3,label="Torch complete-O4 prefix"); ax.set(xlabel="x",ylabel="y",title="Phase tube overlay (native contracts not ranked)"); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures / "phase_tube_overlay.png",dpi=180); plt.close(fig)


def build(run_root: Path, prior_root: Path) -> None:
    object_matrix = _json(run_root / "raw/fixed_object_baseline/trace_committed_4bb/object_trace_matrix.json")
    object_rows = object_matrix["rows"]
    if len(object_rows) != 22 or {row["source_sha"] for row in object_rows} != {"4bb10d54b29dad2b47c5f91ddedafe854a52fac6"}:
        raise ValueError("frozen object matrix is incomplete or source-mixed")
    functional = _json(run_root / "raw/fixed_functional/equivalence.json")
    oracle = _json(run_root / "raw/fixed_outward/oracle.json")
    outward = _json(run_root / "raw/fixed_outward/matrix/summary.json")
    outward_rows = []
    for source_row in outward["rows"]:
        row = dict(source_row)
        failures = [int(value) for value in row["first_failure_indices"] if int(value) >= 0]
        row["longest_all_batch_validated_steps"] = min(failures) if failures else int(row["steps"])
        row["longest_all_batch_validated_horizon"] = row["longest_all_batch_validated_steps"] * 0.01
        outward_rows.append(row)
    terminal = _json(run_root / "raw/structured_terminal/s1_local/structured_terminal_ab.json")
    field_map = _json(run_root / "raw/structured_terminal/s1_local/field_map.json")
    second = _json(run_root / "raw/second_system/summary.json")
    compiled_paths = sorted(
        path for path in (run_root / "raw/fixed_compiled").glob("*/summary.json")
        if "profile_cache_replay" not in path.parent.name
    )
    if len(compiled_paths) < 4:
        raise ValueError("compiled evidence lacks CPU/GPU B64 and CUDA B1/B8 signatures")
    compiled_rows = []
    for path in compiled_paths:
        value = _json(path)
        warm = _stable_warm(value["compiled_warm_s"])
        exact = bool(value["all_probe_inputs_bit_exact"] and value["full_run_bit_exact"])
        compiled_rows.append({
            "lane": path.parent.name,
            "device_group": "cuda_v100" if str(value["device"]).startswith("cuda") else "cpu",
            "device": value["device"], "batch": value["batch"], "steps": value["steps"],
            "requested_horizon": value["requested_horizon"], "compiled_boundary_steps": value["compiled_boundary_steps"],
            "compile_execute_s": value["compile_execute_s"], "compile_inner_s": value["compile_inner_s"],
            "warm_s": value["compiled_warm_s"], "stable_warm_s": warm,
            "stable_warm_min_s": min(warm), "stable_warm_median_s": _median(warm), "stable_warm_max_s": max(warm),
            "eager_functional_full_s": value["eager_functional_full_s"], "completed": value["completed"],
            "first_failure_indices": value["first_failure_indices"],
            "finite_outputs": value.get("finite_outputs", True), "all_probe_inputs_bit_exact": value["all_probe_inputs_bit_exact"],
            "full_run_bit_exact": value["full_run_bit_exact"], "max_abs_finite_difference": max((row["max_abs_finite_difference"] for row in value["full_run_differences"]), default=0.0),
            "compiled_semantics": value.get("compiled_semantics", "ordinary_expression_order_bit_exact" if exact else "performance_only_empirical_arithmetic_changed"),
            "implemented_negative_outcome": value.get("implemented_negative_outcome", None if exact else "FIXED_SUPPORT_COMPILE_SEMANTICS_CHANGED"),
            "graph_break_count": value["graph_break_count"], "host_synchronizations_in_solver_core": value["host_synchronizations_in_solver_core"],
            "final_decision_host_synchronizations": value["final_decision_host_synchronizations"], "solver_device_transfers": value["solver_device_transfers"],
            "peak_memory_bytes": value.get("peak_cuda_memory_bytes"), "process_max_rss_kib": value["process_max_rss_kib"],
            "source_sha": value["source_sha"], "artifact": f"raw_public/fixed_compiled/{path.parent.name}/summary.json",
            **_eligibility(completed=value["completed"], certificate=True, finite=value.get("finite_outputs", True), soundness="empirically sampled only", scope="multi-step lane", formal=False, performance=value["completed"]),
        })
    object_t10 = {row["device_group"]: row for row in object_rows if row["batch"] == 64 and row["requested_horizon"] == 10.0}
    for row in compiled_rows:
        if row["batch"] == 64 and row["steps"] == 1000:
            row["raw_runtime_ratio_vs_frozen_object"] = object_t10[row["device_group"]]["warm_median_s"] / row["stable_warm_median_s"]
            row["ratio_is_identical_semantics_speedup"] = False
    b1_object = _json(run_root / "raw/fixed_object_current/cpu_b1_t10/summary.json")
    b1_object_failure = int(b1_object["first_failure_time_reason"]["step"])
    b1_compiled = next(
        row for row in compiled_rows
        if row["device_group"] == "cpu" and row["batch"] == 1 and row["steps"] == 1000
    )
    b1_compiled_failures = [int(value) for value in b1_compiled["first_failure_indices"] if int(value) >= 0]
    b1_compiled_failure = min(b1_compiled_failures) if b1_compiled_failures else 1000
    for row in outward_rows:
        if row["batch"] == 64:
            row["ordinary_eager_decision"] = "completed"
            row["compiled_ordinary_decision"] = "completed_empirical_arithmetic_changed"
            row["ordinary_first_failure_step"] = -1
            row["compiled_first_failure_step"] = -1
        else:
            row["ordinary_eager_decision"] = "completed" if row["steps"] <= b1_object_failure else "failed_closed"
            row["compiled_ordinary_decision"] = "completed_empirical_arithmetic_changed" if row["steps"] <= b1_compiled_failure else "failed_closed_empirical_arithmetic_changed"
            row["ordinary_first_failure_step"] = b1_object_failure
            row["compiled_first_failure_step"] = b1_compiled_failure
        row["ordinary_eager_artifact"] = "raw_public/fixed_object_current/cpu_b1_t10/summary.json" if row["batch"] == 1 else "raw_public/fixed_object_baseline/trace_committed_4bb/cpu_b64_t10p0/summary.json"
    _write_csv(run_root / "fixed_support_object_baseline.csv", object_rows)
    _write_csv(run_root / "fixed_support_compiled_results.csv", compiled_rows)
    _write_csv(run_root / "fixed_support_outward_results.csv", outward_rows)
    _write_json(run_root / "fixed_support_equivalence.json", {"schema": "fixed_support_equivalence_closure_v1", "functional": functional, "compiled": {"all_signatures_bit_exact": all(row["full_run_bit_exact"] and row["all_probe_inputs_bit_exact"] for row in compiled_rows), "stop_outcome": "FIXED_SUPPORT_COMPILE_SEMANTICS_CHANGED", "rows": compiled_rows}, "outward_one_step_containment": True})
    _write_json(run_root / "fixed_support_profile.json", _profile(run_root))
    _write_json(run_root / "structured_semantics.json", {"schema": "structured_semantics_closure_v1", "field_map": field_map, "capacity": 16, "eligible_sources": ["polynomial_truncation", "integration_overflow"], "source_audit": {"flowstar_revision": "b85a3211748cb77b736fe4ad42ee02d8d2b81148", "diffreach_revision": "dd628eb443b517d6415de93e7035b4baef73963e"}, "typed_terminal_decomposition": terminal["terminal"]["validated_decomposition_contains_image"]})
    _write_json(run_root / "structured_terminal_ab.json", terminal)
    horizon_rows = [{"requested_horizon": value, "status": "not_run_after_stop", "stop_outcome": terminal["stop_outcome"], "validated_horizon": 6.397083942944808, "fresh_request_started": False, "paired_baseline_started": False} for value in (.1,.5,1,4,6,6.5,7.5,10)]
    _write_csv(run_root / "structured_horizon_ladder.csv", horizon_rows)
    _write_csv(run_root / "second_system_results.csv", second["rows"])
    native = _json(prior_root / "native_baselines.json")
    _write_json(run_root / "native_baselines.json", {"schema": "native_baselines_inherited_v1", "source_package": "outputs/mainline_realignment_20260810/20260810T025910Z/native_baselines.json", "source_sha256": _sha(prior_root / "native_baselines.json"), "rows": native})
    timing = [{"lane": "object_eager", **row} for row in object_rows] + [{"lane": "compiled_empirical", **row} for row in compiled_rows] + [{"lane": "outward_reference", **row} for row in outward_rows]
    _write_csv(run_root / "timing.csv", timing)
    _write_csv(run_root / "memory.csv", [{"lane": "object_eager", "device": row["device"], "batch": row["batch"], "steps": row["steps"], "peak_memory_bytes": row["peak_memory_bytes"]} for row in object_rows] + [{"lane": "compiled_empirical", "device": row["device"], "batch": row["batch"], "steps": row["steps"], "peak_memory_bytes": row["peak_memory_bytes"], "process_max_rss_kib": row["process_max_rss_kib"]} for row in compiled_rows])
    failures = {"compiled": "FIXED_SUPPORT_COMPILE_SEMANTICS_CHANGED", "fixed_outward": outward["implemented_negative_outcome"], "structured": terminal["stop_outcome"], "structured_fresh_horizons_started": False, "bounded_next_compiled_optimization": "preserve eager reduction order or add an outward eager shadow around Inductor reductions", "bounded_next_structured_action": "wire S1 state and typed source removal through every accepted prefix boundary before another terminal replay"}
    _write_json(run_root / "failure_attribution.json", failures)
    claim_rows = [
        {"claim": "previous_evidence_package_fresh_clone_complete", "track": "N/M/F", **_eligibility(completed=True, certificate=True, finite=True, soundness="unknown", scope="native build", formal=False, performance=False)},
        {"claim": "fixed_object_functional_bit_exact", "track": "F", **_eligibility(completed=True, certificate=True, finite=True, soundness="empirically sampled only", scope="multi-step lane", formal=False, performance=True)},
        {"claim": "fixed_compiled_cpu_t10_completed_arithmetic_changed", "track": "F", **_eligibility(completed=True, certificate=True, finite=True, soundness="empirically sampled only", scope="multi-step lane", formal=False, performance=True)},
        {"claim": "fixed_compiled_v100_t10_completed_arithmetic_changed", "track": "F", **_eligibility(completed=True, certificate=True, finite=True, soundness="empirically sampled only", scope="multi-step lane", formal=False, performance=True)},
        {"claim": "fixed_outward_fraction_oracle", "track": "F", **_eligibility(completed=oracle["all_passed"], certificate=oracle["all_passed"], finite=True, soundness="safeguarded outward under declared IEEE/backend assumptions", scope="primitive", formal=True, performance=False)},
        {"claim": "structured_s1_terminal_candidate", "track": "F", **_eligibility(completed=False, certificate=False, finite=True, soundness="safeguarded outward under declared IEEE/backend assumptions", scope="primitive", formal=False, performance=False)},
        {"claim": "second_system_plant_only_fallback", "track": "F", **_eligibility(completed=second["result_label"] == "GENERALITY_GATE_PASSED", certificate=True, finite=True, soundness="empirically sampled only", scope="multi-step lane", formal=False, performance=True)},
    ]
    _write_csv(run_root / "claim_registry.csv", claim_rows)
    soundness_rows = [
        {"lane": "stock Flow* pinned build", "numerical_soundness_class": "unsound/ineligible on a demonstrated counterexample", "numerical_soundness_scope": "native build", "note": "scalar-affine counterexample; not a statement about the abstract algorithm"},
        {"lane": "Torch complete O4", "numerical_soundness_class": "formally outward by construction", "numerical_soundness_scope": "multi-step lane", "note": "partial validated prefix"},
        {"lane": "fixed object/functional", "numerical_soundness_class": "empirically sampled only", "numerical_soundness_scope": "multi-step lane", "note": "ordinary round-to-nearest"},
        {"lane": "fixed compiled", "numerical_soundness_class": "empirically sampled only", "numerical_soundness_scope": "multi-step lane", "note": "performance-only after arithmetic difference"},
        {"lane": "fixed outward", "numerical_soundness_class": "safeguarded outward under declared IEEE/backend assumptions", "numerical_soundness_scope": "multi-step lane", "note": "CPU float64 reference; primitive oracle is a separate claim row"},
        {"lane": "structured S1", "numerical_soundness_class": "safeguarded outward under declared IEEE/backend assumptions", "numerical_soundness_scope": "primitive", "note": "not integrated through prefix"},
    ]
    _write_csv(run_root / "soundness_matrix.csv", soundness_rows)
    _copy_public_raw(run_root)
    _plot(run_root, prior_root, object_rows, compiled_rows, outward_rows, terminal, soundness_rows)
    provenance = {
        "schema": "structured_remainder_compiled_provenance_v1", "branch": _git("branch", "--show-current"),
        "build_source_sha": _git("rev-parse", "HEAD"), "worktree_status_excluding_run_root": _git("status", "--short", "--", ".", f":(exclude){run_root.relative_to(ROOT).as_posix()}"),
        "python": platform.python_version(), "platform": platform.platform(), "prior_package_sha256sums": _sha(prior_root / "SHA256SUMS"),
        "object_baseline_source_sha": "4bb10d54b29dad2b47c5f91ddedafe854a52fac6", "flowstar_revision": "b85a3211748cb77b736fe4ad42ee02d8d2b81148", "diffreach_revision": "dd628eb443b517d6415de93e7035b4baef73963e",
    }
    _write_json(run_root / "provenance.json", provenance)
    package_files = sorted(
        path for path in run_root.rglob("*") if path.is_file()
        and "raw/" not in path.relative_to(run_root).as_posix()
        and path.name not in {"manifest.json", "SHA256SUMS"}
    )
    manifest = {"schema": "structured_remainder_compiled_manifest_v1", "run_id": run_root.name, "outcomes": failures, "files": [{"path": path.relative_to(run_root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha(path)} for path in package_files]}
    _write_json(run_root / "manifest.json", manifest)
    checksum_files = package_files + [run_root / "manifest.json"]
    prefix = run_root.relative_to(ROOT)
    (run_root / "SHA256SUMS").write_text("".join(f"{_sha(path)}  {(prefix / path.relative_to(run_root)).as_posix()}\n" for path in checksum_files), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.run_root.resolve(), args.prior_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
