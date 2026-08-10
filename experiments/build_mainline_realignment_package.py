#!/usr/bin/env python3
"""Build the machine tables, required figures, manifest, and checksums."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.artifact_package import (
    REQUIRED_FIGURES,
    REQUIRED_MACHINE_FILES,
    sha256_file,
    validate_required_package,
    verify_sha256sums,
    write_sha256sums,
)

RUN_ID = "20260810T025910Z"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key, "")
                    for key in fields
                }
            )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _flowstar_rectangles(path: Path) -> list[tuple[float, float, float, float]]:
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    started = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("plot "):
            started = True
            continue
        if not started:
            continue
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        fields = line.split()
        if len(fields) == 2:
            try:
                current.append((float(fields[0]), float(fields[1])))
            except ValueError:
                pass
    if current:
        blocks.append(current)
    return [
        (min(x for x, _ in block), max(x for x, _ in block), min(y for _, y in block), max(y for _, y in block))
        for block in blocks
        if len(block) >= 4
    ]


def _fill_rectangles(
    ax: Any,
    rectangles: Iterable[tuple[float, float, float, float]],
    *,
    label: str,
    color: str,
    alpha: float,
) -> None:
    first = True
    for x0, x1, y0, y1 in rectangles:
        ax.fill_between(
            [x0, x1], [y0, y0], [y1, y1], color=color, alpha=alpha,
            linewidth=0, label=label if first else None,
        )
        first = False


def _torch_rectangles(rows: Sequence[Mapping[str, Any]], state: str, kind: str) -> list[tuple[float, float, float, float]]:
    out = []
    for row in rows:
        if row.get("status") != "accepted":
            continue
        values = (
            _float(row, "t_lo"), _float(row, "t_hi"),
            _float(row, f"{kind}_{state}_lo"), _float(row, f"{kind}_{state}_hi"),
        )
        if all(value is not None for value in values):
            out.append(tuple(values))
    return out


def _plot_required_figures(run_root: Path) -> None:
    figures = run_root / "figures"
    figures.mkdir(exist_ok=True)
    native = run_root / "01_native_baselines"
    baseline_dir = native / "torch_complete_o4_authoritative_t6p5"
    baseline_rows = _csv_rows(baseline_dir / "segments.csv")
    candidate_rows = _csv_rows(
        run_root / "04_generic_carry_candidate/final_da21a9e_t10_fresh/segments.csv"
    )
    flow_x = _flowstar_rectangles(native / "flowstar_stock_artifacts/vanderpol_t_x.plt")
    flow_y = _flowstar_rectangles(native / "flowstar_stock_artifacts/vanderpol_t_y.plt")

    for state, flow, filename in (
        ("x", flow_x, "flowstar_style_t_x_overlay.png"),
        ("y", flow_y, "flowstar_style_t_y_overlay.png"),
    ):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        _fill_rectangles(ax, flow, label="stock Flow* full-segment tube", color="#1f77b4", alpha=0.15)
        _fill_rectangles(
            ax, _torch_rectangles(baseline_rows, state, "segment"),
            label="Torch complete-O4 segment tube (partial)", color="#ff7f0e", alpha=0.2,
        )
        _fill_rectangles(
            ax, _torch_rectangles(candidate_rows, state, "segment"),
            label="complete-carry candidate (partial)", color="#d62728", alpha=0.35,
        )
        ax.set(xlabel="physical time", ylabel=state, title=f"Van der Pol t-{state}: explicit tube semantics")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for index, ((_, _, x0, x1), (_, _, y0, y1)) in enumerate(zip(flow_x, flow_y)):
        if index % 3 == 0:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#1f77b4", alpha=0.08, edgecolor="none"))
    bx = _torch_rectangles(baseline_rows, "x", "segment")
    by = _torch_rectangles(baseline_rows, "y", "segment")
    for index, (xrow, yrow) in enumerate(zip(bx, by)):
        if index % 3 == 0:
            ax.add_patch(Rectangle((xrow[2], yrow[2]), xrow[3] - xrow[2], yrow[3] - yrow[2], facecolor="#ff7f0e", alpha=0.10, edgecolor="none"))
    ax.plot([], [], color="#1f77b4", linewidth=6, alpha=0.25, label="stock Flow* tube")
    ax.plot([], [], color="#ff7f0e", linewidth=6, alpha=0.3, label="Torch complete-O4 prefix")
    ax.set(xlabel="x", ylabel="y", title="Phase tube overlay (failed prefixes remain explicit)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "phase_tube_overlay.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    times = [_float(row, "t_hi") for row in baseline_rows if row.get("status") == "accepted"]
    for state, color in (("x", "#1f77b4"), ("y", "#ff7f0e")):
        endpoints = [_float(row, f"endpoint_{state}_width") for row in baseline_rows if row.get("status") == "accepted"]
        tubes = [_float(row, f"segment_{state}_width") for row in baseline_rows if row.get("status") == "accepted"]
        axes[0].plot(times, endpoints, color=color, label=f"endpoint {state}")
        axes[1].plot(times, tubes, color=color, label=f"tube {state}")
    axes[0].set(ylabel="endpoint width", title="Torch complete-O4 validated prefix widths")
    axes[1].set(xlabel="physical time", ylabel="segment-tube width")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "endpoint_tube_width_vs_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    times2, poly, remainder = [], [], []
    for row in baseline_rows:
        if row.get("status") != "accepted":
            continue
        t = _float(row, "t_hi")
        p = _float(row, "carry_composed_poly_range_width_sum") or _float(row, "carry_composed_poly_range_width")
        r = _float(row, "carry_output_remainder_width_sum") or _float(row, "carry_output_remainder_width")
        if t is not None and p is not None and r is not None:
            times2.append(t); poly.append(p); remainder.append(r)
    ax.plot(times2, poly, label="retained polynomial range width", color="#2ca02c")
    ax.plot(times2, remainder, label="ordinary remainder width", color="#d62728")
    ax.set(xlabel="physical time", ylabel="width sum", title="Polynomial-range versus remainder contribution")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "polynomial_range_vs_remainder.png", dpi=180)
    plt.close(fig)

    horizons = {"stock Flow*": 10.0, "stock DiffReach": 10.0, "Torch fixed B64": 10.0,
                "Torch complete O4": 6.397083942944808, "complete carry": 0.04345468750000001}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#999999", "#9467bd", "#2ca02c", "#ff7f0e", "#d62728"]
    ax.bar(list(horizons), list(horizons.values()), color=colors)
    ax.axhline(10, color="black", linewidth=0.8, linestyle="--")
    ax.set(ylabel="highest validated horizon", title="Completion is not cross-contract ranking")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figures / "validated_horizon.png", dpi=180)
    plt.close(fig)

    scaling = _collect_scaling(run_root)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for lane in ("fixed_support", "complete_carry"):
        for device in ("cpu", "cuda_v100"):
            rows = [row for row in scaling if row["lane"] == lane and row["device_group"] == device]
            rows.sort(key=lambda row: row["batch"])
            ax.plot([row["batch"] for row in rows], [row["warm_min_s"] for row in rows], marker="o", label=f"{lane} {device} warm")
            ax.plot([row["batch"] for row in rows], [row["cold_s"] for row in rows], marker="x", linestyle="--", alpha=0.65, label=f"{lane} {device} cold")
    ax.set(xscale="log", yscale="log", xlabel="batch (actual independent partitions)", ylabel="seconds", title="Cold and warm runtime by batch")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "runtime_vs_batch.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    eligible = [row for row in scaling if row["lane"] == "fixed_support" and row["eligible"]]
    for device, marker in (("cpu", "o"), ("cuda_v100", "s")):
        rows = [row for row in eligible if row["device_group"] == device]
        throughput = [row["batch"] * 10 / row["warm_min_s"] for row in rows]
        widths = [row["endpoint_width_sum"] for row in rows]
        ax.scatter(throughput, widths, marker=marker, label=f"fixed-support {device}")
        for row, x, y in zip(rows, throughput, widths):
            ax.annotate(f"B{row['batch']}", (x, y), fontsize=7)
    ax.set(xlabel="eligible short-run batch-steps/s", ylabel="aggregate endpoint width", title="Eligible precision/throughput rows only")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "eligible_precision_throughput.png", dpi=180)
    plt.close(fig)


def _collect_scaling(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in ("fixed_support", "complete_carry"):
        for device_group in ("cpu", "cuda_v100"):
            for batch in (1, 8, 64, 256, 512):
                path = run_root / f"05_batch_scaling/{lane}/{device_group}_b{batch}/summary.json"
                if not path.is_file():
                    continue
                value = _json(path)
                timing = value.get("cold_warm_core_process_runtime", value.get("timing", {}))
                widths = value.get("endpoint_tube_polynomial_remainder_widths", {})
                endpoint_width = widths.get("endpoint_width", [])
                rows.append(
                    {
                        "lane": lane,
                        "device_group": device_group,
                        "device": value["dtype_device"]["device"],
                        "device_name": value["dtype_device"]["device_name"],
                        "batch": batch,
                        "actual_independent_inputs": value.get("actual_independent_inputs", True),
                        "scope": value["execution_kind"],
                        "requested_horizon": value.get("requested_horizon", value.get("h")),
                        "validated_horizon": value.get("validated_horizon", value.get("h") if value.get("accepted_all_batches") else 0),
                        "completion": value.get("completion_status", value.get("status")),
                        "eligible": bool(value.get("eligible_full_horizon", value.get("accepted_all_batches", False))),
                        "cold_s": timing.get("cold_s", timing.get("cold_first_call_s")),
                        "warm_min_s": timing.get("warm_min_s"),
                        "warm_median_s": timing.get("warm_median_s"),
                        "warm_max_s": max(timing.get("warm_s", [])) if timing.get("warm_s") else timing.get("warm_max_s"),
                        "peak_memory_bytes": value.get("peak_memory_bytes"),
                        "endpoint_width_sum": sum(endpoint_width) if endpoint_width else sum(
                            hi - lo for lo, hi in value.get("endpoint_hull", [])
                        ),
                        "soundness_classification": value.get("soundness_classification"),
                        "host_synchronizations": value.get("host_synchronizations", "aggregate inclusion gates and trace extraction"),
                        "source_sha": value.get("source_sha"),
                        "artifact": path.relative_to(run_root).as_posix(),
                    }
                )
    return rows


def _horizon_row(lane: str, summary: Mapping[str, Any], requested: float, artifact: str) -> dict[str, Any]:
    completed = bool(summary.get("completed_requested_horizon"))
    return {
        "lane": lane,
        "requested_horizon": requested,
        "validated_horizon": summary.get("completed_horizon", requested if completed else 0.0),
        "completion_status": "completed" if completed else "PARTIAL_HORIZON",
        "certificate_status": "passed" if completed else "failed_validation",
        "first_failure_reason": summary.get("failure_type", "") if not completed else "",
        "accepted_steps": summary.get("accepted_steps"),
        "rejected_attempts": summary.get("rejected_attempts"),
        "runtime_s": summary.get("runtime_s"),
        "raw_endpoint": summary.get("raw_endpoint"),
        "last_segment_tube": summary.get("last_segment"),
        "fallback_count": summary.get("fallback_count"),
        "repair_used": summary.get("endpoint_repair_used"),
        "artifact": artifact,
    }


def build(run_root: Path) -> None:
    native_source = run_root / "01_native_baselines/native_baselines.json"
    native = _json(native_source)
    native["derived_root_copy"] = {"source": native_source.relative_to(run_root).as_posix(), "source_sha256": sha256_file(native_source)}
    _write_json(run_root / "native_baselines.json", native)

    matched_yaml = ROOT / "benchmarks/vdp_three_lane_contract_20260810.yaml"
    matched = yaml.safe_load(matched_yaml.read_text(encoding="utf-8"))
    _write_json(
        run_root / "matched_contract.json",
        {
            "schema": "torch_tm_matched_contract_result_v1",
            "contract_source": str(matched_yaml.relative_to(ROOT)),
            "contract_sha256": sha256_file(matched_yaml),
            "contract": matched,
            "expressibility": {
                "flowstar_stock": "native official VDP expressible; tubes only; formal stock numerical gate fails",
                "diffreach_stock": "native official VDP expressible with B64/fixed h; stock tube and later masks unavailable",
                "torch_complete": "expressible; partial at 6.397083942944808",
                "torch_fixed": "expressible in framework; native-like fixed support, B64, h=.01",
            },
            "ranked": False,
        },
    )

    fixed_eq = _json(run_root / "02_fixed_support/fixed_support_equivalence.json")
    common_summary = _json(run_root / "03_flowstar_causal_divergence/common_basis_final/summary.json")
    one_step = _json(run_root / "04_generic_carry_candidate/one_step_grid_final/summary.json")
    _write_json(
        run_root / "operator_equivalence.json",
        {
            "schema": "torch_tm_operator_equivalence_v1",
            "fixed_support": fixed_eq,
            "common_basis": common_summary,
            "complete_carry_one_step": one_step,
            "decisions": {"fixed_support": "qualified", "common_basis": "qualified", "complete_carry": "zero one-step mismatch"},
        },
    )

    baseline_dirs = {
        0.1: "torch_complete_o4_t0p1", 0.5: "torch_complete_o4_t0p5", 1.0: "torch_complete_o4_t1p0",
        4.0: "torch_complete_o4_t4p0", 6.0: "torch_complete_o4_t6p0", 6.5: "torch_complete_o4_t6p5",
        7.5: "torch_complete_o4_t7p5", 10.0: "torch_complete_o4_t10p0",
    }
    candidate_dirs = {
        0.1: "final_da21a9e_t0p1", 0.5: "final_da21a9e_t0p5", 1.0: "final_da21a9e_t1p0",
        4.0: "final_da21a9e_t4", 6.0: "final_da21a9e_t6", 6.5: "final_da21a9e_t6p5",
        7.5: "final_da21a9e_t7p5", 10.0: "final_da21a9e_t10_fresh",
    }
    horizon_rows = []
    for horizon, directory in baseline_dirs.items():
        relative = f"01_native_baselines/{directory}/summary.json"
        horizon_rows.append(_horizon_row("torch_complete_o4_frozen_natural", _json(run_root / relative), horizon, relative))
    for horizon, directory in candidate_dirs.items():
        relative = f"04_generic_carry_candidate/{directory}/summary.json"
        horizon_rows.append(_horizon_row("torch_complete_o4_complete_carry", _json(run_root / relative), horizon, relative))
    _write_csv(run_root / "short_horizon.csv", [row for row in horizon_rows if row["requested_horizon"] <= 1.0])

    full_rows = [row for row in horizon_rows if row["requested_horizon"] >= 4.0]
    lanes = native["lanes"]
    full_rows.extend(
        [
            {"lane": "stock_flowstar_native", "requested_horizon": 10, "validated_horizon": 10, "completion_status": "completed", "certificate_status": "reported_safe_but_formal_gate_failed", "accepted_steps": 290, "runtime_s": lanes["flowstar_stock"]["core_runtime_s"], "soundness": "unsound/ineligible", "artifact": "01_native_baselines/native_baselines.json"},
            {"lane": "stock_diffreach_native", "requested_horizon": 10, "validated_horizon": 10, "completion_status": "completed", "certificate_status": "all_returned_initial_masks_pass", "accepted_steps": 1000, "runtime_s": lanes["diffreach_stock"]["timing_s"]["verification_after_jit"], "soundness": "empirically sampled only", "artifact": "01_native_baselines/native_baselines.json"},
        ]
    )
    fixed_t10_path = run_root / "02_fixed_support/cpu_b64_t10/summary.json"
    fixed_t10 = _json(fixed_t10_path)
    full_rows.append(
        {"lane": "torch_fixed_dr7_b64", "requested_horizon": 10, "validated_horizon": fixed_t10["validated_horizon"], "completion_status": fixed_t10["completion_status"], "certificate_status": fixed_t10["certificate_status"], "accepted_steps": fixed_t10["accepted_rejected_steps"]["accepted"], "runtime_s": fixed_t10["cold_warm_core_process_runtime"]["cold_s"], "soundness": fixed_t10["soundness_classification"], "artifact": fixed_t10_path.relative_to(run_root).as_posix()}
    )
    final_baseline = run_root / "06_final_baseline_ladder/torch_complete_o4_adaptive_t10_fresh/summary.json"
    if final_baseline.is_file():
        full_rows.append(_horizon_row("torch_complete_o4_adaptive_final", _json(final_baseline), 10.0, final_baseline.relative_to(run_root).as_posix()))
    _write_csv(run_root / "full_horizon.csv", full_rows)

    causal = _json(run_root / "03_flowstar_causal_divergence/common_basis_final/counterfactuals.json")
    candidate_failure = _json(run_root / "04_generic_carry_candidate/final_da21a9e_t10_fresh/summary.json")
    _write_json(
        run_root / "failure_attribution.json",
        {"schema": "torch_tm_failure_attribution_v1", "first_native_divergence": causal, "candidate_failure": candidate_failure, "candidate_decision": "CANDIDATE_REJECTED"},
    )

    scaling = _collect_scaling(run_root)
    _write_csv(run_root / "batch_scaling.csv", scaling)
    timing_rows = [
        {
            "lane": row["lane"], "device": row["device"], "batch": row["batch"],
            "boundary": row["scope"], "cold_s": row["cold_s"], "warm_min_s": row["warm_min_s"],
            "warm_median_s": row["warm_median_s"], "warm_max_s": row["warm_max_s"],
            "process_startup_import_s": "included only in fresh process, not isolated here",
            "compile_jit_s": "none for eager Torch lane", "cuda_synchronized": row["device_group"] == "cuda_v100",
            "eligible_cross_tool_deployment_claim": False,
        }
        for row in scaling
    ]
    timing_rows.extend(
        [
            {"lane": "stock_flowstar_native", "device": "cpu", "batch": 1, "boundary": "native reported certification core", "cold_s": "UNAVAILABLE", "warm_min_s": lanes["flowstar_stock"]["core_runtime_s"], "process_startup_import_s": "UNAVAILABLE", "compile_jit_s": "not applicable", "eligible_cross_tool_deployment_claim": False},
            {"lane": "stock_diffreach_native", "device": "V100", "batch": 64, "boundary": "native verification", "cold_s": lanes["diffreach_stock"]["timing_s"]["verification_warmup"], "warm_min_s": lanes["diffreach_stock"]["timing_s"]["verification_after_jit"], "compile_jit_s": "included in warmup", "eligible_cross_tool_deployment_claim": False},
        ]
    )
    _write_csv(run_root / "timing.csv", timing_rows)

    soundness = [
        {"lane": "stock_flowstar_native", "classification": "unsound/ineligible", "basis": "scalar-affine MPFR gap", "formal_completion_claim": False},
        {"lane": "stock_diffreach_native", "classification": "empirically sampled only", "basis": "ordinary mixed-builder dtype", "formal_completion_claim": False},
        {"lane": "torch_fixed_ordinary_cpu_cuda", "classification": "empirically sampled only", "basis": "binary64 not universally outward", "formal_completion_claim": False},
        {"lane": "torch_fixed_2ulp_companion_one_step", "classification": "independently outward replayed for exact benchmark workload", "basis": "exact-rational decisive replay", "formal_completion_claim": "one-step companion only"},
        {"lane": "torch_complete_o4_baseline", "classification": "formally outward by construction", "basis": "declared interval path", "formal_completion_claim": "validated prefix only"},
        {"lane": "complete_polynomial_carry_primitive", "classification": "formally outward by construction", "basis": "exact clone of validated endpoint set", "formal_completion_claim": "primitive only; surrounding dense arithmetic separate"},
    ]
    _write_csv(run_root / "soundness_matrix.csv", soundness)
    claims = [
        {"claim": "native entrypoints reproduced", "status": "valid", "scope": "own contracts"},
        {"claim": "Torch fixed explicit-f64 operators reproduce DiffReach", "status": "valid", "scope": "frozen fixture and declared full lane"},
        {"claim": "first native split is raw candidate remainder", "status": "valid", "scope": "last common state and h"},
        {"claim": "complete polynomial carry improves horizon", "status": "invalid", "scope": "rejected at 0.04345468750000001"},
        {"claim": "Torch globally faster or tighter than Flow*", "status": "invalid", "scope": "contracts and eligibility differ"},
        {"claim": "ordinary CUDA is universally outward", "status": "invalid", "scope": "no analytic universal bound"},
        {"claim": "stock Flow* is primary formal comparator", "status": "blocked", "scope": "scalar-affine MPFR gap"},
        {"claim": "multi-step complete lane scales to B512", "status": "blocked", "scope": "outer adaptive scheduler B1; kernel only"},
    ]
    _write_csv(run_root / "claim_registry.csv", claims)

    _plot_required_figures(run_root)
    manifest = {
        "schema": "torch_tm_mainline_realignment_manifest_v1",
        "run_id": RUN_ID,
        "branch": _git("branch", "--show-current"),
        "source_sha": _git("rev-parse", "HEAD"),
        "worktree_status_at_generation": _git("status", "--short"),
        "external_sources": {
            "flowstar": "b85a3211748cb77b736fe4ad42ee02d8d2b81148",
            "diffreach": "dd628eb443b517d6415de93e7035b4baef73963e",
            "xiangru": "recorded in 00_provenance/start_state.json and direction audit",
        },
        "required_machine_files": list(REQUIRED_MACHINE_FILES),
        "required_figures": list(REQUIRED_FIGURES),
        "formal_evidence_scope": "all recursive files listed by SHA256SUMS; debug directories retained but not promoted",
        "candidate_decision": "CANDIDATE_REJECTED",
    }
    _write_json(run_root / "manifest.json", manifest)
    validate_required_package(run_root)
    count = write_sha256sums(run_root)
    valid, errors = verify_sha256sums(run_root)
    if not valid:
        raise RuntimeError(errors)
    print(json.dumps({"run_root": str(run_root), "hashed_files": count, "checksums_valid": valid}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=ROOT / "outputs/mainline_realignment_20260810" / RUN_ID,
    )
    args = parser.parse_args()
    build(args.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
