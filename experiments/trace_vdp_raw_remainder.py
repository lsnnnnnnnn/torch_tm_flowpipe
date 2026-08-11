#!/usr/bin/env python3
"""Export the first Flow*/Torch VDP raw-remainder split in one DAG schema."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, getcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import analyze_vdp_causal_divergence as causal
import run_vdp_dense_backend as authoritative
from torch_tm_flowpipe import DenseRangePolicy, PolynomialODE
from torch_tm_flowpipe.raw_remainder_trace import (
    NODE_FIELDS,
    RawRemainderTraceRecorder,
    SCHEMA,
    float_record,
    interval_record,
    validate_expression_dag,
)


getcontext().prec = 100


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _policy() -> DenseRangePolicy:
    return DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )


def _torch_replay(
    checkpoint: Mapping[str, Any],
    checkpoint_sha: str,
    *,
    recorder: bool,
) -> tuple[dict[str, Any], RawRemainderTraceRecorder | None]:
    contract = authoritative.load_contract()
    trace: list[dict[str, Any]] = []
    policy = _policy()
    base = causal._checkpoint_dense_model(checkpoint["validation_base"], policy=policy, range_trace=trace)
    candidate = causal._checkpoint_dense_model(
        checkpoint["picard_iterations"][-1]["retained"],
        policy=policy,
        range_trace=trace,
        force_zero_remainder=True,
    )
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    trace_recorder = None
    if recorder:
        executable = Path(sys.executable).resolve()
        trace_recorder = RawRemainderTraceRecorder(
            run_id="vdp_first_raw_remainder_split",
            tool="torch_complete_o4",
            source_commit=_git("rev-parse", "HEAD"),
            binary_sha256=_sha(executable),
            checkpoint_sha256=checkpoint_sha,
            t_pre=float(checkpoint["t_pre"]),
            h=float(checkpoint["h_attempt"]),
            picard_iteration=int(contract["requested_order"]),
            normalization_scale=[float(value) for value in checkpoint["normalization_scale"]],
            target_intervals=[
                (-float(contract["target_remainder_radius"]), float(contract["target_remainder_radius"])),
                (-float(contract["target_remainder_radius"]), float(contract["target_remainder_radius"])),
            ],
        )
    replay = causal._torch_raw_replay(
        base,
        candidate,
        h=float(checkpoint["h_attempt"]),
        order=int(contract["requested_order"]),
        cutoff=float(contract["cutoff"]),
        target_radius=float(contract["target_remainder_radius"]),
        validation_eps=1e-12,
        ode=ode,
        raw_trace_recorder=trace_recorder,
    )
    return replay, trace_recorder


def _d(value: str | float) -> Decimal:
    return Decimal(str(value))


def _add(left: tuple[Decimal, Decimal], right: tuple[Decimal, Decimal]) -> tuple[Decimal, Decimal]:
    return left[0] + right[0], left[1] + right[1]


def _neg(value: tuple[Decimal, Decimal]) -> tuple[Decimal, Decimal]:
    return -value[1], -value[0]


def _mul(left: tuple[Decimal, Decimal], right: tuple[Decimal, Decimal]) -> tuple[Decimal, Decimal]:
    products = (left[0] * right[0], left[0] * right[1], left[1] * right[0], left[1] * right[1])
    return min(products), max(products)


def _sum(values: Sequence[tuple[Decimal, Decimal]]) -> tuple[Decimal, Decimal]:
    result = (Decimal(0), Decimal(0))
    for value in values:
        result = _add(result, value)
    return result


def _record_decimal_interval(value: tuple[Decimal, Decimal]) -> dict[str, Any]:
    return interval_record(float(value[0]), float(value[1]))


def _flowstar_row(path: Path, t_pre: float, h: float) -> dict[str, str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    matches = [
        row
        for row in rows
        if row.get("t_before")
        and row.get("h_try")
        and abs(float(row["t_before"]) - t_pre) <= 2e-12
        and float(row["h_try"]) == h
        and row.get("accepted") in {"false", "False", "0"}
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Flow* first-split row, found {len(matches)}")
    required = [
        "flowstar_node_x_squared_remainder_lo",
        "flowstar_node_x_squared_remainder_hi",
        "flowstar_node_one_minus_x_squared_remainder_lo",
        "flowstar_node_one_minus_x_squared_remainder_hi",
        "flowstar_node_nonlinear_times_y_remainder_lo",
        "flowstar_node_nonlinear_times_y_remainder_hi",
        "flowstar_node_minus_x_remainder_lo",
        "flowstar_node_minus_x_remainder_hi",
    ]
    missing = [name for name in required if not matches[0].get(name)]
    if missing:
        raise ValueError(f"Flow* probe lacks semantic node fields: {missing}")
    return matches[0]


def _flowstar_nodes(
    row: Mapping[str, str],
    *,
    source_commit: str,
    binary_sha256: str,
    checkpoint_sha256: str,
    t_pre: float,
    h: float,
    normalization_scale: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = (_d(row["flowstar_expression_input_x_remainder_lo"]), _d(row["flowstar_expression_input_x_remainder_hi"]))
    entries = [
        (_d(row[f"flowstar_y_intermediate_{index}_lo"]), _d(row[f"flowstar_y_intermediate_{index}_hi"]))
        for index in range(7)
    ]
    x_square_parts = {
        "polynomial_times_remainder": _mul(entries[1], target),
        "remainder_times_polynomial": _mul(entries[2], target),
        "remainder_times_remainder": _mul(target, target),
        "polynomial_times_polynomial_dropped": entries[3],
    }
    x_square_formula = _sum(list(x_square_parts.values()))
    x_square_production = (
        _d(row["flowstar_node_x_squared_remainder_lo"]),
        _d(row["flowstar_node_x_squared_remainder_hi"]),
    )
    x_square_coefficient_uncertainty = (
        x_square_production[0] - x_square_formula[0],
        x_square_production[1] - x_square_formula[1],
    )
    one_minus_production = (
        _d(row["flowstar_node_one_minus_x_squared_remainder_lo"]),
        _d(row["flowstar_node_one_minus_x_squared_remainder_hi"]),
    )
    outer_parts = {
        "polynomial_times_remainder": _mul(entries[4], target),
        "remainder_times_polynomial": _mul(entries[5], one_minus_production),
        "remainder_times_remainder": _mul(one_minus_production, target),
        "polynomial_times_polynomial_dropped": entries[6],
    }
    outer_formula = _sum(list(outer_parts.values()))
    outer_production = (
        _d(row["flowstar_node_nonlinear_times_y_remainder_lo"]),
        _d(row["flowstar_node_nonlinear_times_y_remainder_hi"]),
    )
    outer_coefficient_uncertainty = (
        outer_production[0] - outer_formula[0],
        outer_production[1] - outer_formula[1],
    )

    margin = min(
        _d(row["raw_ctrunc_residual_y_lo"]) - target[0],
        target[1] - _d(row["raw_ctrunc_residual_y_hi"]),
    )
    common = {
        "run_id": "vdp_first_raw_remainder_split",
        "tool": "flowstar_complete_o4",
        "source_commit": source_commit,
        "binary_sha256": binary_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "t_pre_decimal": repr(t_pre),
        "t_pre_hex": t_pre.hex(),
        "h_decimal": repr(h),
        "h_hex": h.hex(),
        "picard_iteration": 4,
        "normalization_scale": float_record(float(normalization_scale[1])),
        "target_interval": _record_decimal_interval(target),
        "subset_margin": float_record(float(margin)),
        "decision": "reject",
    }
    zero = interval_record(0.0, 0.0)
    nodes: list[dict[str, Any]] = []

    def add_node(
        node_id: str,
        operation: str,
        output: tuple[Decimal, Decimal],
        parents: Sequence[str],
        *,
        polynomial: tuple[Decimal, Decimal] | None = None,
        inputs: Sequence[tuple[Decimal, Decimal]] = (),
        components: Mapping[str, tuple[Decimal, Decimal]] | None = None,
        coefficient_uncertainty: tuple[Decimal, Decimal] = (Decimal(0), Decimal(0)),
        state_component: int = 1,
        order_before: int = 3,
        order_after: int = 3,
        roundoff: tuple[Decimal, Decimal] = (Decimal(0), Decimal(0)),
        cutoff: tuple[Decimal, Decimal] = (Decimal(0), Decimal(0)),
        integration_overflow: tuple[Decimal, Decimal] = (Decimal(0), Decimal(0)),
    ) -> None:
        component_records = {
            "polynomial_times_polynomial_dropped": zero,
            "polynomial_times_remainder": zero,
            "remainder_times_polynomial": zero,
            "remainder_times_remainder": zero,
            "coefficient_interval_uncertainty": _record_decimal_interval(coefficient_uncertainty),
            "interval_evaluation_dependency": zero,
            "outward_rounding": _record_decimal_interval(roundoff),
        }
        for name, value in (components or {}).items():
            component_records[name] = _record_decimal_interval(value)
        retained_payload = json.dumps(
            {"node": node_id, "polynomial": None if polynomial is None else [str(polynomial[0]), str(polynomial[1])]},
            sort_keys=True,
        ).encode()
        dropped_payload = json.dumps(
            {name: [str(value[0]), str(value[1])] for name, value in (components or {}).items()},
            sort_keys=True,
        ).encode()
        node = {
            **common,
            "normalization_scale": float_record(float(normalization_scale[state_component])),
            "state_component": state_component,
            "expression_node_id": node_id,
            "parent_node_ids": list(parents),
            "operation": operation,
            "polynomial_order_before": order_before,
            "polynomial_order_after": order_after,
            "retained_support_sha256": hashlib.sha256(retained_payload).hexdigest(),
            "dropped_support_sha256": hashlib.sha256(dropped_payload).hexdigest(),
            "polynomial_interval": None if polynomial is None else _record_decimal_interval(polynomial),
            "remainder_input_intervals": [_record_decimal_interval(value) for value in inputs],
            "remainder_output_interval": _record_decimal_interval(output),
            "roundoff_interval": _record_decimal_interval(roundoff),
            "cutoff_interval": _record_decimal_interval(cutoff),
            "integration_overflow_interval": _record_decimal_interval(integration_overflow),
            "multiplication_remainder_components": component_records,
        }
        if set(NODE_FIELDS) - set(node):
            raise AssertionError("Flow* common node omitted required fields")
        nodes.append(node)

    x_id = "flowstar.i4.c0.state_input"
    y_id = "flowstar.i4.c1.state_input"
    add_node(x_id, "state_input", target, (), state_component=0)
    add_node(y_id, "state_input", target, ())
    x_square_id = "flowstar.i4.c1.y_rhs.x_squared"
    add_node(
        x_square_id,
        "multiply",
        x_square_production,
        (x_id,),
        polynomial=(_d(row["flowstar_node_x_squared_polynomial_lo"]), _d(row["flowstar_node_x_squared_polynomial_hi"])),
        inputs=(target, target),
        components=x_square_parts,
        coefficient_uncertainty=x_square_coefficient_uncertainty,
    )
    nodes[-1]["multiplication_operand_polynomial_intervals"] = [
        _record_decimal_interval(entries[1]),
        _record_decimal_interval(entries[2]),
    ]
    one_minus_id = "flowstar.i4.c1.y_rhs.one_minus_x_squared"
    add_node(
        one_minus_id,
        "subtract",
        one_minus_production,
        (x_square_id,),
        polynomial=(_d(row["flowstar_node_one_minus_x_squared_polynomial_lo"]), _d(row["flowstar_node_one_minus_x_squared_polynomial_hi"])),
        inputs=((Decimal(0), Decimal(0)), x_square_production),
    )
    outer_id = "flowstar.i4.c1.y_rhs.nonlinear_times_y"
    add_node(
        outer_id,
        "multiply",
        outer_production,
        (one_minus_id, y_id),
        polynomial=(_d(row["flowstar_node_nonlinear_times_y_polynomial_lo"]), _d(row["flowstar_node_nonlinear_times_y_polynomial_hi"])),
        inputs=(one_minus_production, target),
        components=outer_parts,
        coefficient_uncertainty=outer_coefficient_uncertainty,
    )
    nodes[-1]["multiplication_operand_polynomial_intervals"] = [
        _record_decimal_interval(entries[4]),
        _record_decimal_interval(entries[5]),
    ]
    minus_id = "flowstar.i4.c1.y_rhs.minus_x"
    minus_output = (_d(row["flowstar_node_minus_x_remainder_lo"]), _d(row["flowstar_node_minus_x_remainder_hi"]))
    add_node(
        minus_id,
        "subtract",
        minus_output,
        (outer_id, x_id),
        polynomial=(_d(row["flowstar_node_minus_x_polynomial_lo"]), _d(row["flowstar_node_minus_x_polynomial_hi"])),
        inputs=(outer_production, target),
    )
    integrated = (_d(row["raw_ctrunc_residual_y_lo"]), _d(row["raw_ctrunc_residual_y_hi"]))
    integrate_id = "flowstar.i4.c1.integrate"
    add_node(integrate_id, "time_integration", integrated, (minus_id,), inputs=(minus_output,), order_after=4)
    trunc_id = "flowstar.i4.c1.truncate_o4"
    add_node(trunc_id, "truncate_to_o4_after_time_integration", integrated, (integrate_id,), inputs=(integrated,), order_before=4, order_after=4)
    cutoff_interval = (
        _d(row.get("cutoff_poly_diff_y_lo") or 0),
        _d(row.get("cutoff_poly_diff_y_hi") or 0),
    )
    cutoff_id = "flowstar.i4.c1.cutoff"
    add_node(cutoff_id, "cutoff_discard", integrated, (trunc_id,), inputs=(integrated,), order_before=4, order_after=4, cutoff=cutoff_interval)
    assembly_id = "flowstar.i4.c1.raw_assembly"
    add_node(assembly_id, "raw_candidate_assembly", integrated, (cutoff_id,), inputs=(integrated,), order_before=4, order_after=4)
    after_roundoff = (
        _d(row.get("post_cutoff_residual_y_lo") or row["raw_ctrunc_residual_y_lo"]),
        _d(row.get("post_cutoff_residual_y_hi") or row["raw_ctrunc_residual_y_hi"]),
    )
    roundoff_id = "flowstar.i4.c1.poly_roundoff"
    add_node(
        roundoff_id,
        "polynomial_roundoff_addition",
        after_roundoff,
        (assembly_id,),
        inputs=(integrated, cutoff_interval),
        order_before=4,
        order_after=4,
        roundoff=cutoff_interval,
    )
    add_node(
        "flowstar.i4.c1.subset",
        "subset_test",
        after_roundoff,
        (roundoff_id,),
        inputs=(after_roundoff,),
        order_before=4,
        order_after=4,
    )
    validate_expression_dag(nodes)
    audit = {
        "x_squared_formula_remainder": _record_decimal_interval(x_square_formula),
        "x_squared_production_remainder": _record_decimal_interval(x_square_production),
        "x_squared_coefficient_interval_uncertainty": _record_decimal_interval(x_square_coefficient_uncertainty),
        "nonlinear_formula_remainder_with_production_left_input": _record_decimal_interval(outer_formula),
        "nonlinear_production_remainder": _record_decimal_interval(outer_production),
        "nonlinear_coefficient_interval_uncertainty": _record_decimal_interval(outer_coefficient_uncertainty),
        "production_raw_y": _record_decimal_interval(integrated),
        "cached_evaluate_remainder_raw_y": interval_record(
            float(row["raw_remainder_integration_remainder_y_lo"]),
            float(row["raw_remainder_integration_remainder_y_hi"]),
        ),
    }
    return nodes, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.torch_checkpoint.resolve()
    flowstar_trace_path = args.flowstar_trace.resolve()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint_sha = _sha(checkpoint_path)

    unlogged, _ = _torch_replay(checkpoint, checkpoint_sha, recorder=False)
    logged, recorder = _torch_replay(checkpoint, checkpoint_sha, recorder=True)
    assert recorder is not None
    inert = bool(
        torch.equal(unlogged["image_lo"], logged["image_lo"])
        and torch.equal(unlogged["image_hi"], logged["image_hi"])
        and torch.equal(unlogged["subset_margin"], logged["subset_margin"])
        and unlogged["accepted"] == logged["accepted"]
    )
    if not inert:
        raise RuntimeError("TORCH_OBSERVER_NONINERT_STOP")
    expected_lo = torch.tensor(checkpoint["picard_image_remainder_lo"], dtype=torch.float64)
    expected_hi = torch.tensor(checkpoint["picard_image_remainder_hi"], dtype=torch.float64)
    if not torch.equal(logged["image_lo"], expected_lo) or not torch.equal(logged["image_hi"], expected_hi):
        raise RuntimeError("Torch trace replay moved from the frozen production checkpoint")

    torch_artifact = recorder.artifact()
    flowstar_row = _flowstar_row(
        flowstar_trace_path,
        float(checkpoint["t_pre"]),
        float(checkpoint["h_attempt"]),
    )
    flowstar_binary_sha = _sha(args.flowstar_binary.resolve()) if args.flowstar_binary else "not_provided"
    flowstar_nodes, flowstar_audit = _flowstar_nodes(
        flowstar_row,
        source_commit=args.flowstar_source_commit,
        binary_sha256=flowstar_binary_sha,
        checkpoint_sha256=_sha(flowstar_trace_path),
        t_pre=float(checkpoint["t_pre"]),
        h=float(checkpoint["h_attempt"]),
        normalization_scale=[float(value) for value in checkpoint["normalization_scale"]],
    )
    artifact = {
        "schema": SCHEMA,
        "node_fields": list(NODE_FIELDS),
        "coordinate_semantics": "same frozen physical prestate; local normalized coordinates and physical tau [0,h]",
        "torch_observer_inertness": {
            "status": "pass",
            "logged_unlogged_image_bit_exact": inert,
            "logged_matches_frozen_production_checkpoint": True,
        },
        "flowstar_probe": {
            "source_trace_sha256": _sha(flowstar_trace_path),
            "binary_sha256": flowstar_binary_sha,
            "probe_t_pre_decimal": flowstar_row["t_before"],
            "canonical_t_pre_decimal": repr(float(checkpoint["t_pre"])),
            "probe_threshold_offset": float(flowstar_row["t_before"]) - float(checkpoint["t_pre"]),
            "observation_mode": "read-only duplicate official expression evaluation on the official accepted prefix",
        },
        "flowstar_internal_audit": flowstar_audit,
        "nodes": [*flowstar_nodes, *torch_artifact["nodes"]],
    }
    output = output_dir / "raw_remainder_expression_tree.json"
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        output_label = str(output.relative_to(ROOT))
    except ValueError:
        output_label = output.name
    summary = {
        "schema": "vdp_raw_remainder_trace_run_v1",
        "outcome": "TRACE_EXPORTED_ROOT_CAUSE_ANALYSIS_PENDING",
        "torch_observer_inert": inert,
        "torch_nodes": len(torch_artifact["nodes"]),
        "flowstar_nodes": len(flowstar_nodes),
        "expression_tree": output_label,
        "expression_tree_sha256": _sha(output),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-checkpoint", type=Path, required=True)
    parser.add_argument("--flowstar-trace", type=Path, required=True)
    parser.add_argument("--flowstar-binary", type=Path)
    parser.add_argument("--flowstar-source-commit", default="b85a321")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
