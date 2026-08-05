#!/usr/bin/env python3
"""Replay one frozen Van der Pol terminal attempt with an explicit range policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe import (
    DenseRangePolicy,
    PolynomialODE,
    flowpipe_step_from_tm,
    load_terminal_checkpoint,
    tmvector_hashes,
)

import run_vdp_dense_backend as runner


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _variable_orders(value: str) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(index) for index in order.split(",") if index.strip())
        for order in value.split(";")
        if order.strip()
    )


def _candidate_hashes(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in trace if row.get("phase") == "polynomial_picard"]
    if not rows:
        return {}
    last = rows[-1]
    return {
        "coefficient_sha256": last.get("coefficient_sha256"),
        "exponent_support_sha256": last.get("exponent_support_sha256"),
        "basis_hash": last.get("basis_hash"),
        "effective_degree": last.get("effective_degree"),
        "picard_iterations": int(last.get("iteration", len(rows))),
    }


def _reference(checkpoint_path: Path) -> Mapping[str, Any] | None:
    directory = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    path = directory / "terminal_reference.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("terminal reference must be a JSON mapping")
    if "validation_rejection_reason" not in value:
        rows = value.get("validation_rows", [])
        value = dict(value)
        value["validation_rejection_reason"] = rows[-1].get("rejection_reason", "") if rows else ""
    return value


def _exact_reference_comparison(summary: Mapping[str, Any], reference: Mapping[str, Any] | None) -> dict[str, Any]:
    if reference is None:
        return {"reference_present": False, "replay_exact": False, "differences": ["missing terminal_reference.json"]}
    fields = (
        "attempted_h",
        "accepted",
        "status",
        "validation_rejection_reason",
        "candidate_hashes",
        "candidate_remainder",
        "picard_image_remainder",
        "subset_margin",
        "backend_lane",
        "backend_counters",
    )
    differences = []
    for field in fields:
        if field == "backend_counters":
            expected = dict(reference.get(field, {}))
            actual = dict(summary.get(field, {}))
            if any(actual.get(key) != value for key, value in expected.items()):
                differences.append(field)
            if any(int(actual.get(key, 0)) != 0 for key in ("range_subdivision_invocations", "range_leaf_evaluations")):
                differences.append(field)
        elif _jsonable(summary.get(field)) != _jsonable(reference.get(field)):
            differences.append(field)
    return {"reference_present": True, "replay_exact": not differences, "differences": differences}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_vars = tuple(int(item) for item in args.split_vars.split(",") if item.strip())
    named_contexts = tuple(item.strip() for item in args.named_contexts.split(",") if item.strip())
    variable_orders = _variable_orders(args.variable_orders)
    if args.subdivision_depth < 0 or args.max_leaves <= 0:
        raise ValueError("subdivision depth must be nonnegative and max leaves positive")

    contract = runner.load_contract()
    checkpoint = load_terminal_checkpoint(
        args.checkpoint.resolve(),
        expected_contract=contract,
        expected_order=contract["requested_order"],
        expected_dtype=contract["dtype"],
    )
    normalized_current = checkpoint.normal_state.normalized_initial_tm(contract["requested_order"])
    current_hashes = tmvector_hashes(checkpoint.current)
    normalized_hashes = tmvector_hashes(normalized_current)
    if current_hashes != normalized_hashes:
        raise ValueError("saved current TM does not match the normal-state normalized initial TM")

    h = float(checkpoint.scheduler["h_attempted"])
    config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_full_sha256": checkpoint.manifest["full_checkpoint_sha256"],
        "range_method": args.range_method,
        "subdivision_depth": int(args.subdivision_depth),
        "max_leaves": int(args.max_leaves),
        "split_vars": split_vars,
        "named_contexts": named_contexts,
        "variable_orders": variable_orders,
        "trigger": args.trigger,
        "device": args.device,
        "contract": contract,
    }
    (output_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(_jsonable(config), sort_keys=True), encoding="utf-8")
    _write_json(
        output_dir / "command.json",
        {
            "argv": sys.argv,
            "cwd": str(ROOT),
            "branch": _git("branch", "--show-current"),
            "commit": _git("rev-parse", "HEAD"),
            "worktree_status": _git("status", "--short"),
            "config_sha256": hashlib.sha256(json.dumps(_jsonable(config), sort_keys=True).encode("utf-8")).hexdigest(),
        },
    )
    diagnostics: list[dict[str, Any]] = []
    policy = DenseRangePolicy(
        method=args.range_method,
        max_depth=0 if args.range_method == "natural" else int(args.subdivision_depth),
        max_leaves=int(args.max_leaves),
        split_vars=split_vars,
        trigger=args.trigger,
        named_contexts=named_contexts,
        variable_orders=variable_orders,
    )
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    started = time.perf_counter()
    segment = flowpipe_step_from_tm(
        ode,
        normalized_current,
        h,
        contract["requested_order"],
        max_validation_attempts=2,
        validation_eps=1e-12,
        validation_mode=contract["validation_mode"],
        target_remainder_radius=contract["target_remainder_radius"],
        cutoff_threshold=contract["cutoff"],
        diagnostics=diagnostics,
        diagnostics_context={
            "segment_index": int(checkpoint.scheduler["accepted_segment_count"]),
            "t_before": float(checkpoint.scheduler["current_time"]),
            "mode": "dense",
        },
        tm_backend="dense",
        dense_device=args.device,
        dense_range_policy=policy,
    )
    runtime_s = time.perf_counter() - started
    accepted = segment.status == "validated" and segment.endpoint_raw_tm is not None
    summary = {
        "attempted_h": h,
        "t_before": float(checkpoint.scheduler["current_time"]),
        "accepted": accepted,
        "status": segment.status,
        "message": segment.message,
        "validation_rejection_reason": next(
            (
                row.get("rejection_reason", "")
                for row in reversed(segment.backend_trace or [])
                if row.get("phase") == "remainder_validation"
            ),
            "",
        ),
        "candidate_hashes": _candidate_hashes(segment.backend_trace or []),
        "candidate_remainder": segment.candidate_remainder,
        "picard_image_remainder": segment.picard_image_remainder,
        "subset_margin": segment.subset_margin,
        "backend_lane": segment.backend_lane,
        "backend_counters": segment.backend_counters,
        "range_method": args.range_method,
        "variable_orders": [list(order) for order in variable_orders],
        "range_leaf_count": sum(
            int(row.get("leaf_count", 0))
            for row in (segment.backend_trace or [])
            if row.get("phase") == "polynomial_range" and int(row.get("leaf_count", 1)) > 1
        ),
        "range_subdivision_invocations": sum(
            row.get("phase") == "polynomial_range" and int(row.get("leaf_count", 1)) > 1
            for row in (segment.backend_trace or [])
        ),
        "range_coverage_valid": all(
            bool(row.get("coverage_valid", False))
            for row in (segment.backend_trace or [])
            if row.get("phase") == "polynomial_range"
        ),
        "fallback_count": int((segment.backend_counters or {}).get("sparse_fallback_count", 0)),
        "endpoint_repair_used": False,
        "runtime_s": runtime_s,
        "current_hashes": current_hashes,
        "normalized_current_hashes": normalized_hashes,
        "contract_sha256": checkpoint.manifest["contract_sha256"],
        "checkpoint_full_sha256": checkpoint.manifest["full_checkpoint_sha256"],
    }
    comparison = _exact_reference_comparison(summary, _reference(args.checkpoint.resolve()))
    summary["natural_replay_exact"] = bool(comparison["replay_exact"]) if args.range_method == "natural" else None
    reference = _reference(args.checkpoint.resolve())
    summary["coefficient_contract_match"] = bool(
        reference
        and summary["candidate_hashes"] == reference.get("candidate_hashes")
        and summary["attempted_h"] == reference.get("attempted_h")
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "hash_comparison.json", {"state_hashes_equal": current_hashes == normalized_hashes, **comparison})
    trace: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    range_call_index = 0
    for source_row in segment.backend_trace or []:
        row = dict(source_row)
        stages = list(row.pop("horner_stages", []))
        if row.get("phase") == "polynomial_range":
            encoded_stages = json.dumps(_jsonable(stages), sort_keys=True, separators=(",", ":")).encode("utf-8")
            row["range_call_index"] = range_call_index
            row["horner_stage_count"] = len(stages)
            row["horner_stage_sha256"] = hashlib.sha256(encoded_stages).hexdigest()
            for stage in stages:
                stage_rows.append(
                    {
                        "range_call_index": range_call_index,
                        "context": row.get("context"),
                        **dict(stage),
                    }
                )
            range_call_index += 1
        trace.append(row)
    _write_jsonl(output_dir / "picard_trace.jsonl", trace)
    _write_jsonl(output_dir / "remainder_ledger.jsonl", [row for row in trace if row.get("phase") == "remainder_validation"])
    range_rows = [row for row in trace if row.get("phase") == "polynomial_range"]
    _write_jsonl(output_dir / "range_contexts.jsonl", range_rows)
    _write_jsonl(output_dir / "range_context_trace.jsonl", range_rows)
    _write_jsonl(output_dir / "horner_stage_trace.jsonl", stage_rows)
    _write_json(
        output_dir / "range_leaves.json",
        {
            "subdivision_invocations": summary["range_subdivision_invocations"],
            "leaf_evaluations": summary["range_leaf_count"],
            "coverage_valid": summary["range_coverage_valid"],
            "max_leaves": args.max_leaves,
            "depth": args.subdivision_depth,
            "split_vars": list(split_vars),
        },
    )
    _write_json(output_dir / "decision.json", {"replay_exact": comparison["replay_exact"] if args.range_method == "natural" else None, "coefficient_contract_match": summary["coefficient_contract_match"], "accepted": accepted, "range_method": args.range_method})
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--range-method",
        choices=(
            "natural",
            "subdivision",
            "adaptive_subdivision",
            "horner_fixed",
            "horner_registered_best",
            "subdivision_then_horner",
            "horner_per_leaf",
        ),
        default="natural",
    )
    parser.add_argument("--subdivision-depth", type=int, default=0)
    parser.add_argument("--max-leaves", type=int, default=64)
    parser.add_argument("--split-vars", default="0,1")
    parser.add_argument("--named-contexts", default="", help="optional comma-separated range contexts")
    parser.add_argument(
        "--variable-orders",
        default="0,1,2;1,0,2;2,0,1",
        help="semicolon-separated Horner variable permutations",
    )
    parser.add_argument(
        "--trigger",
        choices=("always", "on_validation_failure", "proactive_depth1_on_named_contexts"),
        default="always",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if (summary["natural_replay_exact"] if summary["range_method"] == "natural" else summary["coefficient_contract_match"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
