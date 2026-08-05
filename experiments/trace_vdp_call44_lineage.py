#!/usr/bin/env python3
"""Replay and materialize the frozen terminal call-44 multiplication lineage."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

for path in (ROOT / "src", ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import replay_vdp_terminal_range as replay
import torch_tm_flowpipe.batched_dense_tm as dense


TARGET_T_PRE = 6.397083942944808
TARGET_H = 0.003623635847674574
TARGET_RANGE_CALL = 44
TARGET_CHECKPOINT_SHA256 = "dcb8f646d45c9742e0cff23fea596c12e53d8ccd00d1544f70564a44a7463420"
TARGET_CANDIDATE_SHA256 = "bc1433d0d3c89339fca6091e41c0a6667d70c92d2dd4e35ae8b14236d131863c"
TARGET_SUPPORT_SHA256 = "d0aa354b9057267556d5bb3bc09a36ed4162b36fb44588b0b930dd9e935041e9"
HISTORICAL_SELECTED = (-0.029997247026804494, 0.02187259686437867)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def _validate_checkpoint_identity(state_path: Path) -> dict[str, Any]:
    manifest_path = state_path.with_name("terminal_state_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_full = manifest.get("full_checkpoint_sha256")
    if observed_full != TARGET_CHECKPOINT_SHA256:
        raise ValueError(
            "frozen checkpoint full SHA256 changed: "
            f"expected {TARGET_CHECKPOINT_SHA256}, observed {observed_full}"
        )
    observed_payload = hashlib.sha256(state_path.read_bytes()).hexdigest()
    expected_payload = manifest.get("payload_sha256")
    if observed_payload != expected_payload:
        raise ValueError(
            "frozen checkpoint payload SHA256 changed: "
            f"manifest {expected_payload}, observed {observed_payload}"
        )
    return manifest


def _number(value: Any) -> dict[str, str]:
    number = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if not torch.isfinite(torch.tensor(number, dtype=torch.float64)):
        raise ValueError("lineage values must be finite")
    return {"decimal": format(number, ".17g"), "hex": number.hex()}


def _tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_json_bytes(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _term_id(kind: str, *parts: Any) -> str:
    return kind + ":" + hashlib.sha256(_json_bytes([kind, *parts])).hexdigest()[:24]


def _exponents(basis: dense.BatchedMonomialBasis) -> list[tuple[int, ...]]:
    return [tuple(int(value) for value in row) for row in basis.exponents.detach().cpu().tolist()]


class _Capture:
    def __init__(self) -> None:
        self.range_index = 0
        self.mul_index = 0
        self.active_mul: int | None = None
        self.operations: dict[int, dict[str, Any]] = {}
        self.call44: dict[str, Any] | None = None
        self.original_range = dense._range_for_terms_with_policy
        self.original_mul = dense.BatchedPolynomial.mul_trunc

    def range_wrapper(self, coeffs: torch.Tensor, exponents: torch.Tensor, domain_lo: torch.Tensor, domain_hi: torch.Tensor, **kwargs: Any):
        index = self.range_index
        self.range_index += 1
        result = self.original_range(coeffs, exponents, domain_lo, domain_hi, **kwargs)
        if index == TARGET_RANGE_CALL:
            if kwargs.get("context") != "polynomial_truncation" or self.active_mul is None:
                raise RuntimeError("range call 44 is not an active polynomial truncation multiplication")
            self.call44 = {
                "range_call_index": index,
                "multiplication_operation_index": self.active_mul,
                "coeffs": coeffs.detach().cpu().clone(),
                "exponents": exponents.detach().cpu().clone(),
                "domain_lo": domain_lo.detach().cpu().clone(),
                "domain_hi": domain_hi.detach().cpu().clone(),
                "policy": kwargs["policy"],
                "context": kwargs["context"],
                "result": result,
            }
        return result

    def mul_wrapper(self, left: dense.BatchedPolynomial, right: dense.BatchedPolynomial, *args: Any, **kwargs: Any):
        index = self.mul_index
        self.mul_index += 1
        previous = self.active_mul
        self.active_mul = index
        operation = {
            "index": index,
            "left": left.coeffs.detach().cpu().clone(),
            "right": right.coeffs.detach().cpu().clone(),
            "basis": left.basis,
            "max_degree": kwargs.get("max_degree"),
            "left_hash": _tensor_hash(left.coeffs),
            "right_hash": _tensor_hash(right.coeffs),
        }
        self.operations[index] = operation
        try:
            result = self.original_mul(left, right, *args, **kwargs)
            polynomial = result[0] if isinstance(result, tuple) else result
            operation["output"] = polynomial.coeffs.detach().cpu().clone()
            operation["output_hash"] = _tensor_hash(polynomial.coeffs)
            return result
        finally:
            self.active_mul = previous

    def install(self) -> None:
        dense._range_for_terms_with_policy = self.range_wrapper

        capture = self

        def wrapper(left: dense.BatchedPolynomial, right: dense.BatchedPolynomial, *args: Any, **kwargs: Any):
            return capture.mul_wrapper(left, right, *args, **kwargs)

        dense.BatchedPolynomial.mul_trunc = wrapper

    def restore(self) -> None:
        dense._range_for_terms_with_policy = self.original_range
        dense.BatchedPolynomial.mul_trunc = self.original_mul


def _capture_replay(checkpoint: Path) -> tuple[_Capture, dict[str, Any]]:
    capture = _Capture()
    with tempfile.TemporaryDirectory(prefix="vdp-call44-lineage-") as temporary:
        args = replay.parse_args(
            [
                "--checkpoint", str(checkpoint),
                "--output-dir", str(Path(temporary) / "replay"),
                "--range-method", "subdivision_then_horner",
                "--subdivision-depth", "1",
                "--max-leaves", "4",
                "--split-vars", "0,1",
                "--named-contexts", "polynomial_truncation",
                "--variable-orders", "0,1,2;1,0,2;2,0,1",
                "--trigger", "proactive_depth1_on_named_contexts",
                "--device", "cpu",
            ]
        )
        capture.install()
        try:
            summary = replay.run(args)
        finally:
            capture.restore()
    return capture, summary


def _node(
    term_id: str,
    *,
    kind: str,
    exponent: tuple[int, ...],
    coefficient: Any,
    parents: Sequence[str],
    operation: str,
    source_class: str,
    root_reason: str | None = None,
    first_intervalization: str | None = None,
    first_truncated_step: int | None = None,
) -> dict[str, Any]:
    return {
        "term_id": term_id,
        "node_kind": kind,
        "exponent_tuple": list(exponent),
        "degree": sum(exponent),
        "coefficient": _number(coefficient),
        "state_component": 1,
        "picard_iteration": "validation_attempt_1_raw_remainder_compat",
        "operation_type": operation,
        "parent_ids": list(parents),
        "source_class": source_class,
        "first_produced_accepted_step": 307,
        "first_cutoff_or_truncate_step": first_truncated_step,
        "first_symbolic_to_interval_operation": first_intervalization,
        "terminal_y_remainder_path": [
            "call44_discarded_polynomial_range",
            "raw_rhs_y_remainder",
            "integration_by_tau_[0,h]",
            "base_remainder_plus_poly_diff",
            "self_map_image_y",
            "terminal_subset_margin_y",
        ],
        "root_reason": root_reason,
    }


def _build_lineage(capture: _Capture) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if capture.call44 is None:
        raise RuntimeError("range call 44 was not captured")
    call44 = capture.call44
    op_index = int(call44["multiplication_operation_index"])
    if op_index != 11 or op_index - 1 not in capture.operations:
        raise RuntimeError(f"call44 multiplication identity changed: operation {op_index}")
    op43 = capture.operations[op_index - 1]
    op44 = capture.operations[op_index]
    basis = op44["basis"]
    exponents = _exponents(basis)
    left44 = op44["left"][0, 0]
    right44 = op44["right"][0, 0]
    x43 = op43["left"][0, 0]
    if op43["left_hash"] != op43["right_hash"]:
        raise RuntimeError("operation before call44 is no longer x*x")
    if int(op43["max_degree"]) != 3 or int(op44["max_degree"]) != 3:
        raise RuntimeError("raw-remainder effective degree changed")

    nodes: list[dict[str, Any]] = []
    x_roots: dict[int, str] = {}
    y_roots: dict[int, str] = {}
    for index, exponent in enumerate(exponents):
        x_id = _term_id("candidate-x", exponent, _number(x43[index]))
        y_id = _term_id("candidate-y", exponent, _number(right44[index]))
        x_roots[index] = x_id
        y_roots[index] = y_id
        root_explanation = (
            "Picard-candidate polynomial root. Its boundary ancestry is the fresh normalized current-state polynomial; "
            "historical right-map/insertion dependence was already enclosed at accepted boundary step 307."
        )
        nodes.append(
            _node(
                x_id, kind="root_polynomial_term", exponent=exponent, coefficient=x43[index], parents=[],
                operation="terminal_picard_candidate_x", source_class="current_state_polynomial",
                root_reason=root_explanation, first_intervalization="ancestral_normalized_insertion_boundary_step_307",
            )
        )
        nodes.append(
            _node(
                y_id, kind="root_polynomial_term", exponent=exponent, coefficient=right44[index], parents=[],
                operation="terminal_picard_candidate_y", source_class="current_state_polynomial",
                root_reason=root_explanation, first_intervalization="ancestral_normalized_insertion_boundary_step_307",
            )
        )

    kept_left, kept_right, kept_out, *_ = basis.multiplication_plan_for_degree(3)
    kept_left_l = kept_left.detach().cpu().tolist()
    kept_right_l = kept_right.detach().cpu().tolist()
    kept_out_l = kept_out.detach().cpu().tolist()
    op43_routes_by_out: dict[int, list[str]] = {index: [] for index in range(len(exponents))}
    for route, (left_index, right_index, out_index) in enumerate(zip(kept_left_l, kept_right_l, kept_out_l)):
        coefficient = x43[left_index] * x43[right_index]
        route_id = _term_id("x-square-route", route, exponents[left_index], exponents[right_index], _number(coefficient))
        op43_routes_by_out[out_index].append(route_id)
        nodes.append(
            _node(
                route_id, kind="multiplication_route", exponent=exponents[out_index], coefficient=coefficient,
                parents=[x_roots[left_index], x_roots[right_index]], operation="x_times_x_kept_route",
                source_class="current_state_polynomial",
            )
        )

    lhs_ids: dict[int, str] = {}
    cutoff_threshold = 1e-10
    op43_output = op43["output"][0, 0]
    for index, exponent in enumerate(exponents):
        aggregate_id = _term_id("x-square-aggregate", exponent, _number(op43_output[index]))
        parents = op43_routes_by_out[index]
        nodes.append(
            _node(
                aggregate_id, kind="coefficient_aggregate", exponent=exponent, coefficient=op43_output[index],
                parents=parents, operation="scatter_add_equal_exponents", source_class="current_state_polynomial",
                root_reason=None if parents else "structurally zero basis slot above effective degree 3",
            )
        )
        cutoff_id = _term_id("x-square-cutoff", exponent, _number(left44[index]))
        lhs_ids[index] = cutoff_id
        nodes.append(
            _node(
                cutoff_id, kind="cutoff_result", exponent=exponent, coefficient=left44[index],
                parents=[aggregate_id], operation="cutoff_keep_or_zero", source_class="current_state_polynomial",
                first_intervalization=("cutoff_interval_remainder" if abs(float(op43_output[index])) <= cutoff_threshold and float(op43_output[index]) != 0.0 else None),
                first_truncated_step=(307 if abs(float(op43_output[index])) <= cutoff_threshold and float(op43_output[index]) != 0.0 else None),
            )
        )

    _, _, _, dropped_left, dropped_right, dropped_merge, dropped_unique = basis.multiplication_plan_for_degree(3)
    dropped_left_l = dropped_left.detach().cpu().tolist()
    dropped_right_l = dropped_right.detach().cpu().tolist()
    dropped_merge_l = dropped_merge.detach().cpu().tolist()
    unique_exponents = [tuple(int(value) for value in row) for row in dropped_unique.detach().cpu().tolist()]
    route_coefficients = left44.index_select(0, dropped_left.cpu()) * right44.index_select(0, dropped_right.cpu())
    captured_coefficients = call44["coeffs"][0, 0]
    if not torch.equal(route_coefficients, captured_coefficients):
        raise RuntimeError("call44 route coefficients do not reconstruct the captured range input")
    monomial_lo, monomial_hi = dense._monomial_interval_bounds_for_exponents(
        call44["domain_lo"], call44["domain_hi"], call44["exponents"]
    )
    term_lo, term_hi = dense._interval_mul(
        captured_coefficients,
        captured_coefficients,
        monomial_lo[0],
        monomial_hi[0],
    )
    terms: list[dict[str, Any]] = []
    unique_parents: dict[int, list[str]] = {index: [] for index in range(len(unique_exponents))}
    for route, (left_index, right_index, merge_index) in enumerate(zip(dropped_left_l, dropped_right_l, dropped_merge_l)):
        exponent = tuple(int(value) for value in call44["exponents"][route].tolist())
        coefficient = captured_coefficients[route]
        route_id = _term_id(
            "call44-discarded-route", route, exponent, exponents[left_index], exponents[right_index], _number(coefficient)
        )
        parents = [lhs_ids[left_index], y_roots[right_index]]
        unique_parents[merge_index].append(route_id)
        nodes.append(
            _node(
                route_id, kind="discarded_multiplication_route", exponent=exponent, coefficient=coefficient,
                parents=parents, operation="x_squared_times_y_degree_gt_3", source_class="current_state_polynomial",
                first_intervalization="call44_polynomial_truncation_range", first_truncated_step=307,
            )
        )
        terms.append(
            {
                "term_id": route_id,
                "route_index": route,
                "stable_call_hash": "",
                "exponent_tuple": json.dumps(list(exponent), separators=(",", ":")),
                "degree": sum(exponent),
                "coefficient_decimal": _number(coefficient)["decimal"],
                "coefficient_hex": _number(coefficient)["hex"],
                "term_range_lower_decimal": _number(term_lo[route])["decimal"],
                "term_range_lower_hex": _number(term_lo[route])["hex"],
                "term_range_upper_decimal": _number(term_hi[route])["decimal"],
                "term_range_upper_hex": _number(term_hi[route])["hex"],
                "left_parent_id": parents[0],
                "right_parent_id": parents[1],
                "source_class": "current_state_polynomial",
                "first_truncated_step": 307,
                "first_intervalization_operation": "call44_polynomial_truncation_range",
            }
        )

    merged_coefficients = dense._merge_coefficients_by_index(
        captured_coefficients.view(1, 1, -1), call44["exponents"].new_tensor(dropped_merge_l), len(unique_exponents)
    )[0, 0]
    for index, exponent in enumerate(unique_exponents):
        aggregate_id = _term_id("call44-discarded-aggregate", exponent, _number(merged_coefficients[index]))
        nodes.append(
            _node(
                aggregate_id, kind="discarded_exponent_aggregate", exponent=exponent,
                coefficient=merged_coefficients[index], parents=unique_parents[index],
                operation="canonical_equal_exponent_aggregation_with_outward_rounding_for_range",
                source_class="current_state_polynomial", first_intervalization="call44_polynomial_truncation_range",
                first_truncated_step=307,
            )
        )

    identity_payload = {
        "stage": "validation_attempt_1_raw_remainder_compat",
        "component": 1,
        "rhs_operation": "negative_x_squared_times_y",
        "multiplication_operation_index": op_index,
        "range_context": call44["context"],
        "range_call_index": TARGET_RANGE_CALL,
        "left_input_sha256": op44["left_hash"],
        "right_input_sha256": op44["right_hash"],
        "basis_fingerprint": basis.fingerprint,
        "effective_max_degree": 3,
        "route_coefficient_sha256": _tensor_hash(call44["coeffs"]),
        "route_exponent_sha256": _tensor_hash(call44["exponents"]),
    }
    stable_hash = hashlib.sha256(_json_bytes(identity_payload)).hexdigest()
    for term in terms:
        term["stable_call_hash"] = stable_hash
    return nodes, terms, {"identity_payload": identity_payload, "stable_call_hash": stable_hash}


def _first_intervalization_rows(
    checkpoint: Mapping[str, Any],
    call44: Mapping[str, Any],
    prefix_segments: Path | None,
) -> list[dict[str, Any]]:
    normal = checkpoint["normal_state"]
    diagnostics = normal["diagnostics"]
    right = normal["tmv_right"]["models"]
    result = call44["result"]
    rows: list[dict[str, Any]] = []
    if prefix_segments is not None:
        with prefix_segments.open(newline="", encoding="utf-8") as handle:
            prefix = list(csv.DictReader(handle))
        if not prefix or prefix[0].get("status") != "accepted":
            raise ValueError("prefix segments do not begin with an accepted transition")
        first = prefix[0]
        first_truncation = next(
            (
                row for row in prefix
                if max(
                    abs(float(row.get("carry_insertion_truncation_width_x") or 0.0)),
                    abs(float(row.get("carry_insertion_truncation_width_y") or 0.0)),
                ) > 1e-300
            ),
            None,
        )
        rows.append(
            {
                "event_index": 0, "accepted_step": int(first["segment_index"]), "component": "x,y",
                "operation": "first_normalized_insertion_right_map_remainder_enclosure",
                "lower": "", "upper": "",
                "lineage_role": (
                    "first trajectory-wide symbolic-to-independent-interval boundary; "
                    f"output remainder widths x={float(first['carry_output_remainder_width_x']):.17g};"
                    f"y={float(first['carry_output_remainder_width_y']):.17g}"
                ),
            }
        )
        if first_truncation is not None:
            rows.append(
                {
                    "event_index": 1, "accepted_step": int(first_truncation["segment_index"]), "component": "x,y",
                    "operation": "first_non_underflow_normalized_insertion_degree_truncation",
                    "lower": "", "upper": "",
                    "lineage_role": (
                        f"truncation widths x={float(first_truncation['carry_insertion_truncation_width_x']):.17g};"
                        f"y={float(first_truncation['carry_insertion_truncation_width_y']):.17g}"
                    ),
                }
            )
    next_index = len(rows)
    rows.extend([
        {
            "event_index": next_index, "accepted_step": normal["step_index"], "component": "x,y",
            "operation": "normalized_insertion_right_map_interval_enclosure",
            "lower": f"{right[0]['remainder']['lo_hex']};{right[1]['remainder']['lo_hex']}",
            "upper": f"{right[0]['remainder']['hi_hex']};{right[1]['remainder']['hi_hex']}",
            "lineage_role": "ancestral historical right-map dependency; nonlinear interval attribution is not additive",
        },
        {
            "event_index": next_index + 1, "accepted_step": normal["step_index"], "component": "x,y",
            "operation": "normalized_insertion_degree_truncation",
            "lower": "", "upper": "",
            "lineage_role": f"sound interval widths x={diagnostics['insertion_truncation_width_x']:.17g};y={diagnostics['insertion_truncation_width_y']:.17g}",
        },
        {
            "event_index": next_index + 2, "accepted_step": 307, "component": "y",
            "operation": "call44_polynomial_truncation_range",
            "lower": _number(result.selected_lo[0, 0])["decimal"],
            "upper": _number(result.selected_hi[0, 0])["decimal"],
            "lineage_role": "direct call44 discarded polynomial enclosure entering raw RHS y remainder",
        },
    ])
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    state_path = checkpoint_path / "terminal_state.json" if checkpoint_path.is_dir() else checkpoint_path
    checkpoint_manifest = _validate_checkpoint_identity(state_path)
    checkpoint = json.loads(state_path.read_text(encoding="utf-8"))
    scheduler = checkpoint["scheduler"]
    if float(scheduler["current_time"]) != TARGET_T_PRE or float(scheduler["h_attempted"]) != TARGET_H:
        raise ValueError("frozen terminal scheduler identity changed")
    capture, summary = _capture_replay(checkpoint_path)
    if summary["candidate_hashes"]["coefficient_sha256"] != TARGET_CANDIDATE_SHA256:
        raise ValueError("candidate coefficient hash changed")
    if summary["candidate_hashes"]["exponent_support_sha256"] != TARGET_SUPPORT_SHA256:
        raise ValueError("candidate support hash changed")
    nodes, terms, identity = _build_lineage(capture)
    call44 = capture.call44
    assert call44 is not None
    result = call44["result"]
    reconstructed = capture.original_range(
        call44["coeffs"], call44["exponents"], call44["domain_lo"], call44["domain_hi"],
        policy=call44["policy"], context=call44["context"], trace=None,
    )
    selected = (float(result.selected_lo[0, 0]), float(result.selected_hi[0, 0]))
    reconstruction = (float(reconstructed.selected_lo[0, 0]), float(reconstructed.selected_hi[0, 0]))
    historical_error = max(abs(selected[0] - HISTORICAL_SELECTED[0]), abs(selected[1] - HISTORICAL_SELECTED[1]))
    reconstruction_error = max(abs(selected[0] - reconstruction[0]), abs(selected[1] - reconstruction[1]))
    node_ids = {node["term_id"] for node in nodes}
    missing_parents = sorted({parent for node in nodes for parent in node["parent_ids"] if parent not in node_ids})
    discarded_ids = {term["term_id"] for term in terms}
    discarded_nodes = [node for node in nodes if node["term_id"] in discarded_ids]
    classified = [node for node in discarded_nodes if node["source_class"] in {
        "current_state_polynomial", "historical_right_map_carry", "insertion_carry", "interval_remainder"
    }]
    identity_document = {
        "schema": "vdp_call44_identity_v1",
        "t_pre": _number(TARGET_T_PRE),
        "h": _number(TARGET_H),
        "checkpoint_full_sha256": TARGET_CHECKPOINT_SHA256,
        "checkpoint_payload_sha256": checkpoint_manifest["payload_sha256"],
        "candidate_coefficient_sha256": TARGET_CANDIDATE_SHA256,
        "candidate_exponent_support_sha256": TARGET_SUPPORT_SHA256,
        **identity,
        "ordinal_is_secondary": True,
        "discarded_route_count": len(terms),
        "discarded_unique_exponent_count": int(call44["exponents"].unique(dim=0).shape[0]),
        "flowstar_correspondence": {
            "one_to_one_call": None,
            "corresponding_stage": "Picard_ctrunc_normal inside stock symbolic-remainder adaptive step",
            "reason": "Flowstar does not expose an algebraically identical call ordinal/basis at the observation hook; no Flowstar call44 is fabricated.",
        },
    }
    _write_json(output / "call44_identity.json", identity_document)
    with (output / "call44_terms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(terms[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(terms)
    _write_jsonl(output / "call44_lineage.jsonl", nodes)
    events = _first_intervalization_rows(checkpoint, call44, args.prefix_segments.resolve() if args.prefix_segments else None)
    with (output / "first_intervalization_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(events[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(events)
    coverage = {
        "schema": "vdp_call44_lineage_coverage_v1",
        "discarded_route_count": len(terms),
        "discarded_routes_with_source_class": len(classified),
        "source_class_coverage": len(classified) / len(terms),
        "lineage_node_count": len(nodes),
        "missing_parent_ids": missing_parents,
        "parent_chain_complete": not missing_parents,
        "captured_selected_range": {"lower": selected[0], "upper": selected[1], "width": selected[1] - selected[0]},
        "reconstructed_selected_range": {"lower": reconstruction[0], "upper": reconstruction[1], "width": reconstruction[1] - reconstruction[0]},
        "historical_selected_range": {"lower": HISTORICAL_SELECTED[0], "upper": HISTORICAL_SELECTED[1], "width": HISTORICAL_SELECTED[1] - HISTORICAL_SELECTED[0]},
        "reconstruction_max_abs_error": reconstruction_error,
        "historical_max_abs_error": historical_error,
        "rounding_tolerance": 1e-14,
        "reconstruction_passed": reconstruction_error == 0.0 and historical_error <= 1e-14,
        "structural_lineage_complete": not missing_parents and len(classified) == len(terms),
        "causal_attribution_performed": False,
        "causal_attribution_reason": "interval/Horner/subdivision operations are nonlinear enclosures; per-term ranges are not asserted to add to the final violation",
    }
    _write_json(output / "lineage_coverage.json", coverage)
    global_first = events[0] if args.prefix_segments else None
    global_first_text = (
        f"The trajectory-wide first observed intervalization is accepted step {global_first['accepted_step']} "
        f"(`{global_first['operation']}`). "
        if global_first is not None else
        "No prefix trajectory was supplied, so the trajectory-wide first event is not asserted here. "
    )
    summary_text = f"""# Terminal call 44 lineage

Stable identity: `{identity['stable_call_hash']}`. The ordinal 44 is secondary to the stage/component/operation/input hashes in `call44_identity.json`.

- Frozen point: `t_pre={TARGET_T_PRE:.17g}`, `h={TARGET_H:.17g}`.
- Operation: validation-attempt-1 raw-remainder compatibility evaluation, y component, discarded part of `x²*y`, effective degree 3.
- Discarded routes: {len(terms)}; unique exponent groups: {identity_document['discarded_unique_exponent_count']}.
- Source classification: {len(classified)}/{len(terms)} routes (`current_state_polynomial`). The interval target remainder is an arithmetic side input, not a polynomial parent.
- Selected range: `[{selected[0]:.17g}, {selected[1]:.17g}]`, width `{selected[1] - selected[0]:.17g}`.
- Reconstruction error: `{reconstruction_error:.17g}`; historical-evidence error: `{historical_error:.17g}` (allowed `1e-14`).
- Parent-chain gaps: {len(missing_parents)}.

This DAG is structural lineage, not a linear causal allocation of the terminal y violation. Subdivision and Horner interval choices are non-additive, so the audit does not fabricate per-term contributions whose sum allegedly equals the final margin.

{global_first_text}The immediate ancestry of the terminal roots crosses the accepted-step-307 normalized-insertion boundary: historical right-map and insertion-truncation structure is enclosed before the next fresh affine state is formed. Call 44 then intervalizes its own discarded degree-overflow polynomial. Stock Flowstar has a corresponding Picard truncation stage but no demonstrated one-to-one call/basis identity, so no “Flowstar call 44” is claimed.
"""
    (output / "call44_lineage_summary.md").write_text(summary_text, encoding="utf-8")
    result_summary = {
        "stable_call_hash": identity["stable_call_hash"],
        "discarded_route_count": len(terms),
        "coverage": coverage["source_class_coverage"],
        "parent_chain_complete": coverage["parent_chain_complete"],
        "reconstruction_passed": coverage["reconstruction_passed"],
    }
    _write_json(output / "lineage_result.json", result_summary)
    return result_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-segments", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
