#!/usr/bin/env python3
"""Export a read-only stage ledger from the production dense VDP step-1 path."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_vdp_dense_backend as authoritative
from torch_tm_flowpipe import (
    DenseRangePolicy,
    FlowstarNormalFlowpipeState,
    Interval,
    PolynomialODE,
    flowpipe_step_flowstar_style_adaptive,
)
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedTaylorModel,
    DenseExecutionCounters,
    dense_picard_validate_step,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.raw_remainder_trace import RawRemainderTraceRecorder


SCHEMA = "common_step_operator_stage_ledger_v1"
BASIS = {
    "id": "canonical_tau_ux_uy_o4",
    "canonical_variables": ["tau", "ux", "uy"],
    "torch_actual_variables": ["ux", "uy", "tau"],
    "actual_to_canonical": [1, 2, 0],
    "domain": [["0", "1/100"], ["-1", "1"], ["-1", "1"]],
    "support": "complete total-degree <= 4",
}
CLASSIFICATIONS = {
    "BITWISE_EQUAL",
    "EXACT_VALUE_EQUAL_DIFFERENT_ENCODING",
    "ROUNDING_ONLY_BOTH_SOUND",
    "ENCLOSURE_DIFFERENT_BOTH_SOUND",
    "FIRST_SEMANTIC_DELTA",
    "UNDER_ENCLOSURE_WITNESS",
    "UNRESOLVED",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _number(value: float) -> dict[str, Any]:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("ledger numbers must be finite")
    numerator, denominator = number.as_integer_ratio()
    return {
        "decimal": format(number, ".17g"),
        "hex": number.hex(),
        "binary64_bits": f"0x{struct.unpack('>Q', struct.pack('>d', number))[0]:016x}",
        "exact_rational": {"numerator": str(numerator), "denominator": str(denominator)},
    }


def _interval(lo: float, hi: float, *, provenance: str) -> dict[str, Any]:
    if float(lo) > float(hi):
        raise ValueError(f"invalid interval [{lo}, {hi}]")
    return {
        "lower": _number(lo),
        "upper": _number(hi),
        "precision_bits": 53,
        "directed_rounding_provenance": provenance,
    }


def _source(function: Callable[..., Any]) -> dict[str, Any]:
    lines, first = inspect.getsourcelines(function)
    return {
        "file": str(Path(inspect.getsourcefile(function) or "").resolve().relative_to(ROOT)),
        "function": function.__qualname__,
        "line_start": first,
        "line_end": first + len(lines) - 1,
    }


def _canonical_exponent(exponent: Sequence[int]) -> list[int]:
    if len(exponent) == 3:
        return [int(exponent[2]), int(exponent[0]), int(exponent[1])]
    if len(exponent) == 2:
        return [0, int(exponent[0]), int(exponent[1])]
    raise ValueError(f"Torch step-1 exponent has wrong dimension: {exponent}")


def _tensor_intervals(lo: torch.Tensor, hi: torch.Tensor, provenance: str) -> list[dict[str, Any]]:
    left = lo.detach().cpu().reshape(-1).tolist()
    right = hi.detach().cpu().reshape(-1).tolist()
    return [_interval(a, b, provenance=provenance) for a, b in zip(left, right)]


def _dense_model(model: BatchedTaylorModel) -> dict[str, Any]:
    coeffs = model.poly.coeffs.detach().cpu()
    if coeffs.shape[0] != 1:
        raise ValueError("step-1 audit requires B1")
    actual_exponents = model.poly.basis.exponents.detach().cpu().tolist()
    components: list[dict[str, Any]] = []
    for component in range(coeffs.shape[1]):
        terms = []
        for slot, exponent in enumerate(actual_exponents):
            coefficient = float(coeffs[0, component, slot])
            if coefficient == 0.0:
                continue
            terms.append(
                {
                    "actual_exponents": [int(v) for v in exponent],
                    "canonical_exponents": _canonical_exponent(exponent),
                    "coefficient": _number(coefficient),
                }
            )
        terms.sort(key=lambda row: tuple(row["canonical_exponents"]))
        components.append(
            {
                "component": ["x", "y"][component],
                "support": [row["canonical_exponents"] for row in terms],
                "terms": terms,
                "ordinary_remainder": _interval(
                    float(model.rem_lo[0, component]),
                    float(model.rem_hi[0, component]),
                    provenance="torch.nextafter outward interval arithmetic",
                ),
                "remainder_ownership": {
                    name: _interval(
                        float(lo[0, component]),
                        float(hi[0, component]),
                        provenance="torch.nextafter outward interval arithmetic",
                    )
                    for name, (lo, hi) in model.ledger.entries.items()
                },
            }
        )
    canonical_domain_indices = (2, 0, 1) if model.domain_lo.shape[1] == 3 else (0, 1)
    domain = [
        _interval(float(model.domain_lo[0, i]), float(model.domain_hi[0, i]), provenance="stored binary64 endpoints")
        for i in canonical_domain_indices
    ]
    return {
        "basis": BASIS,
        "domain_canonical_order": domain,
        "components": components,
        "actual_basis_fingerprint": model.poly.basis.fingerprint,
        "actual_dense_coefficients_sha256": _sha_bytes(coeffs.contiguous().numpy().tobytes()),
    }


def _sparse_model(vector: Any) -> dict[str, Any]:
    components = []
    for component, model in enumerate(vector):
        terms = []
        for exponent, coefficient in sorted(model.polynomial.terms.items()):
            value = float(coefficient.detach().cpu())
            if value == 0.0:
                continue
            terms.append(
                {
                    "actual_exponents": list(exponent),
                    "canonical_exponents": (
                        _canonical_exponent(exponent)
                        if len(exponent) == 3
                        else [0, int(exponent[0]), int(exponent[1])]
                        if len(exponent) == 2
                        else list(exponent)
                    ),
                    "coefficient": _number(value),
                }
            )
        components.append(
            {
                "component": ["x", "y"][component],
                "terms": terms,
                "remainder": _interval(
                    float(model.remainder.lo),
                    float(model.remainder.hi),
                    provenance="Torch sparse TaylorModel outward interval",
                ),
            }
        )
    return {"components": components, "variables": vector.n_vars}


def _range_payload(model: BatchedTaylorModel, *, endpoint_tau: float | None = None) -> dict[str, Any]:
    selected = model.endpoint(2, endpoint_tau) if endpoint_tau is not None else model
    poly_lo, poly_hi = selected.poly.range_bound(
        selected.domain_lo,
        selected.domain_hi,
        policy=selected.range_policy,
        context="step1_actual_stage_ledger",
        trace=[],
    )
    final_lo = torch.nextafter(poly_lo + selected.rem_lo, torch.full_like(poly_lo, -torch.inf))
    final_hi = torch.nextafter(poly_hi + selected.rem_hi, torch.full_like(poly_hi, torch.inf))
    return {
        "evaluation_contract": "tau=h algebraic substitution before range" if endpoint_tau is not None else "tau in [0,h] segment range",
        "range_method": selected.range_policy.method,
        "polynomial": _tensor_intervals(poly_lo, poly_hi, "Torch natural monomial range with nextafter outward rounding"),
        "ordinary_remainder": _tensor_intervals(selected.rem_lo, selected.rem_hi, "validated binary64 interval"),
        "final": _tensor_intervals(final_lo, final_hi, "outward addition of polynomial and ordinary remainder"),
        "evaluated_model": _dense_model(selected),
    }


def _equal_dense(left: BatchedTaylorModel, right: BatchedTaylorModel) -> bool:
    return bool(
        torch.equal(left.poly.coeffs, right.poly.coeffs)
        and torch.equal(left.rem_lo, right.rem_lo)
        and torch.equal(left.rem_hi, right.rem_hi)
        and torch.equal(left.domain_lo, right.domain_lo)
        and torch.equal(left.domain_hi, right.domain_hi)
        and left.ledger.intervals() == right.ledger.intervals()
    )


class Ledger:
    def __init__(self, *, source_commit: str, observer_patch_sha256: str, working_diff_sha256: str):
        self.source_commit = source_commit
        self.observer_patch_sha256 = observer_patch_sha256
        self.working_diff_sha256 = working_diff_sha256
        self.rows: list[dict[str, Any]] = []
        self.last_hashes: list[str] = []

    def add(
        self,
        stage_id: str,
        payload: Any,
        *,
        iteration: int | None,
        source: Mapping[str, Any],
        ownership: Mapping[str, str],
        classification: str = "UNRESOLVED",
        inputs: Sequence[str] | None = None,
    ) -> str:
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"unknown classification {classification}")
        output_hash = _sha_bytes(_json_bytes(payload))
        row = {
            "schema": SCHEMA,
            "ledger_row_index": len(self.rows),
            "tool": "torch_complete_o4_legacy_production",
            "actual_source": dict(source),
            "git_sha": self.source_commit,
            "observer_patch_sha256": self.observer_patch_sha256,
            "working_diff_sha256": self.working_diff_sha256,
            "stage_id": stage_id,
            "iteration": iteration,
            "basis_id": BASIS["id"],
            "basis": BASIS,
            "support": payload.get("components", []) if isinstance(payload, dict) else [],
            "binary_exact_coefficient_encoding": "IEEE-754 binary64 bits + hexfloat",
            "exact_rational_value": "decoded per finite binary64 number with as_integer_ratio",
            "interval_precision_bits": 53,
            "directed_rounding_provenance": "stage payload; every interval names its producer",
            "remainder_ownership": dict(ownership),
            "input_artifact_hashes": list(self.last_hashes if inputs is None else inputs),
            "output_artifact_hash": output_hash,
            "classification": classification,
            "payload": payload,
        }
        self.rows.append(row)
        self.last_hashes = [output_hash]
        return output_hash


def export(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_commit = _git("rev-parse", "HEAD")
    observer_patch = ROOT / "experiments" / "flowstar_step1_stage_observer_20260813.patch"
    observer_patch_sha = _sha_path(observer_patch)
    working_diff = subprocess.run(
        ["git", "diff", "--binary"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    working_diff_sha = _sha_bytes(working_diff)
    ledger = Ledger(
        source_commit=source_commit,
        observer_patch_sha256=observer_patch_sha,
        working_diff_sha256=working_diff_sha,
    )

    contract = authoritative.load_contract()
    if contract["initial_box"] != [[1.1, 1.4], [2.35, 2.45]]:
        raise ValueError("frozen VDP initial box moved")
    if contract["validation_mode"] != "flowstar_raw_remainder_compat":
        raise ValueError("frozen production validation mode moved")
    h = 0.01
    order = 4
    state = FlowstarNormalFlowpipeState.from_initial_box(
        [Interval(1.1, 1.4), Interval(2.35, 2.45)], order
    )
    current = state.normalized_initial_tm(order)
    base = sparse_tmvector_to_dense(
        current.extend_domain(Interval(0.0, h)),
        order=order,
        device="cpu",
        dtype=torch.float64,
        counters=DenseExecutionCounters(),
        segment_boundary=True,
        range_policy=DenseRangePolicy(),
        range_trace=[],
    )

    source_validate = _source(dense_picard_validate_step)
    base_hash = ledger.add(
        "normalized_initial_tm",
        _dense_model(base),
        iteration=0,
        source=source_validate,
        ownership={"ordinary": "initial input", "truncation": "none", "cutoff": "none", "source_ledger": "none"},
    )

    iterations: list[tuple[int, BatchedTaylorModel, BatchedTaylorModel]] = []

    def observer(iteration: int, pre_cutoff: BatchedTaylorModel, retained: BatchedTaylorModel) -> None:
        iterations.append((iteration, pre_cutoff, retained))

    executable_sha = _sha_path(Path(sys.executable).resolve())
    recorder = RawRemainderTraceRecorder(
        run_id="torch-step1-stage-ledger",
        tool="torch_complete_o4_legacy_production",
        source_commit=source_commit,
        binary_sha256=executable_sha,
        checkpoint_sha256=base_hash,
        t_pre=0.0,
        h=h,
        picard_iteration=order,
        normalization_scale=state.scales,
        target_intervals=[(-1.0e-4, 1.0e-4), (-1.0e-4, 1.0e-4)],
    )
    observed = dense_picard_validate_step(
        PolynomialODE.from_system_spec(contract["canonical_system_spec"]),
        base,
        h=h,
        order=order,
        tau_index=2,
        target_remainder_radius=1.0e-4,
        cutoff_threshold=1.0e-10,
        max_validation_attempts=2,
        validation_eps=1.0e-12,
        validation_mode="flowstar_raw_remainder_compat",
        polynomial_observer=observer,
        raw_remainder_trace_recorder=recorder,
    )
    baseline = dense_picard_validate_step(
        PolynomialODE.from_system_spec(contract["canonical_system_spec"]),
        base,
        h=h,
        order=order,
        tau_index=2,
        target_remainder_radius=1.0e-4,
        cutoff_threshold=1.0e-10,
        max_validation_attempts=2,
        validation_eps=1.0e-12,
        validation_mode="flowstar_raw_remainder_compat",
    )
    hook_equal = bool(
        observed.status == baseline.status
        and observed.validation_attempts == baseline.validation_attempts
        and _equal_dense(observed.segment_tm, baseline.segment_tm)
        and (
            observed.raw_endpoint is baseline.raw_endpoint is None
            or (
                observed.raw_endpoint is not None
                and baseline.raw_endpoint is not None
                and _equal_dense(observed.raw_endpoint, baseline.raw_endpoint)
            )
        )
        and torch.equal(observed.subset_margin, baseline.subset_margin)
        and observed.trace == baseline.trace
    )
    if not hook_equal:
        raise RuntimeError("Torch observation hook changed production step-1 output")

    for iteration, pre_cutoff, retained in iterations:
        before_hash = ledger.add(
            "picard_polynomial_image_pre_cutoff",
            _dense_model(pre_cutoff),
            iteration=iteration,
            source=_source(__import__("torch_tm_flowpipe.batched_dense_tm", fromlist=["dense_polynomial_picard"]).dense_polynomial_picard),
            ownership={"ordinary": "arithmetic ledger, not fed to next polynomial iterate", "truncation": "pre-cutoff arithmetic ledger", "cutoff": "not yet applied", "source_ledger": "none"},
        )
        ledger.add(
            "picard_polynomial_image_retained",
            _dense_model(retained),
            iteration=iteration,
            source=_source(__import__("torch_tm_flowpipe.batched_dense_tm", fromlist=["dense_polynomial_picard"]).dense_polynomial_picard),
            ownership={"ordinary": "zeroed before next polynomial iterate", "truncation": "retained in audit ledger", "cutoff": "discarded terms enclosed in cutoff ledger", "source_ledger": "none"},
            inputs=[before_hash],
        )

    raw_artifact = recorder.artifact()
    _write_json(output_dir / "raw_remainder_expression_tree.json", raw_artifact)
    ledger.add(
        "ode_expression_and_raw_remainder_tree",
        raw_artifact,
        iteration=order,
        source=source_validate,
        ownership={"ordinary": "raw Picard image", "truncation": "per-node polynomial_truncation", "cutoff": "per-node cutoff", "source_ledger": "none at step 1"},
    )
    for row in observed.trace:
        if row.get("phase") == "remainder_validation":
            ledger.add(
                "remainder_refinement_candidate_image_subset",
                json.loads(json.dumps(row)),
                iteration=int(row["attempt"]),
                source=source_validate,
                ownership={"ordinary": "validated raw-compat image", "truncation": "validated ledger", "cutoff": "validated ledger", "source_ledger": "none at step 1"},
            )

    ledger.add(
        "returned_tmvPre_polynomial_and_remainder",
        _dense_model(observed.segment_tm),
        iteration=order,
        source=source_validate,
        ownership={"ordinary": "picard_residual", "truncation": "validated decomposition", "cutoff": "validated decomposition", "source_ledger": "none"},
    )
    ledger.add(
        "segment_polynomial_and_final_range",
        _range_payload(observed.segment_tm),
        iteration=order,
        source=_source(BatchedTaylorModel.range_bound),
        ownership={"ordinary": "added after polynomial range", "truncation": "inside validated remainder", "cutoff": "inside validated remainder", "source_ledger": "none"},
    )
    ledger.add(
        "endpoint_polynomial_and_final_range",
        _range_payload(observed.segment_tm, endpoint_tau=h),
        iteration=order,
        source=_source(BatchedTaylorModel.endpoint),
        ownership={"ordinary": "carried through tau=h substitution", "truncation": "inside validated remainder", "cutoff": "inside validated remainder", "source_ledger": "none"},
    )

    full = flowpipe_step_flowstar_style_adaptive(
        PolynomialODE.from_system_spec(contract["canonical_system_spec"]),
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        h=h,
        h_min=h,
        h_max=h,
        order=order,
        target_remainder_radius=1.0e-4,
        cutoff_threshold=1.0e-10,
        max_validation_attempts=2,
        validation_eps=1.0e-12,
        validation_mode="flowstar_raw_remainder_compat",
        reset_mode="normalized_insertion",
        step_policy_mode="flowstar_compat",
        tm_backend="dense",
    )
    if full.status != "validated" or full.flowstar_normal_state is None or full.reset_tm is None:
        raise RuntimeError(f"full Torch step-1 path failed: {full.status}: {full.message}")
    reset_payload = {
        "center": [_number(value) for value in full.flowstar_normal_state.center],
        "scale": [_number(value) for value in full.flowstar_normal_state.scales],
        "right_map": _sparse_model(full.flowstar_normal_state.tmv_right),
        "pre_map": _sparse_model(full.flowstar_normal_state.tmv_pre),
        "domain": [
            _interval(float(item.lo), float(item.hi), provenance="Torch normalized insertion domain")
            for item in full.flowstar_normal_state.domain
        ],
        "reset_tm": _sparse_model(full.reset_tm),
    }
    ledger.add(
        "normalized_insertion_reset",
        reset_payload,
        iteration=order,
        source=_source(flowpipe_step_flowstar_style_adaptive),
        ownership={"ordinary": "materialized in reset/pre map", "truncation": "materialized", "cutoff": "materialized", "source_ledger": "legacy path has none"},
    )
    step2_pre = full.flowstar_normal_state.normalized_initial_tm(order)
    ledger.add(
        "step2_prestate",
        _sparse_model(step2_pre),
        iteration=None,
        source=_source(FlowstarNormalFlowpipeState.normalized_initial_tm),
        ownership={"ordinary": "reset remainder", "truncation": "reset remainder", "cutoff": "reset remainder", "source_ledger": "none"},
    )

    actual_path = {
        "schema": "torch_step1_actual_path_capture_v1",
        "contract": contract,
        "source_commit": source_commit,
        "observer_patch_sha256": observer_patch_sha,
        "working_diff_sha256": working_diff_sha,
        "hook_read_only_equivalence": hook_equal,
        "status": observed.status,
        "validation_attempts": observed.validation_attempts,
        "subset_margin": observed.subset_margin.detach().cpu().tolist(),
        "iterations": len(iterations),
        "full_path_status": full.status,
        "full_path_backend_lane": full.backend_lane,
    }
    _write_json(output_dir / "actual_path_summary.json", actual_path)
    _write_json(output_dir / "stage_ledger.json", {"schema": SCHEMA, "rows": ledger.rows})
    fields = [
        "tool", "git_sha", "stage_id", "iteration", "basis_id", "classification",
        "source_file", "source_function", "source_line_start", "source_line_end",
        "input_artifact_hashes", "output_artifact_hash",
    ]
    with (output_dir / "stage_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in ledger.rows:
            source = row["actual_source"]
            writer.writerow(
                {
                    **{name: row.get(name, "") for name in fields},
                    "source_file": source["file"],
                    "source_function": source["function"],
                    "source_line_start": source["line_start"],
                    "source_line_end": source["line_end"],
                    "input_artifact_hashes": json.dumps(row["input_artifact_hashes"], separators=(",", ":")),
                }
            )
    manifest = {
        "schema": "torch_step1_stage_ledger_manifest_v1",
        "files": {
            path.name: {"sha256": _sha_path(path), "bytes": path.stat().st_size}
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
        "row_count": len(ledger.rows),
        "raw_expression_node_count": len(raw_artifact["nodes"]),
        "hook_read_only_equivalence": hook_equal,
        "status": observed.status,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = export(parse_args(argv).output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
