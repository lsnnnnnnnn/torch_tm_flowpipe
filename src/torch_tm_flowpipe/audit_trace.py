"""Observation-only, fail-closed transition trace serialization."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .flowpipe import FlowpipeSegment, FlowstarNormalFlowpipeState
from .interval import Interval
from .tm_vector import TMVector


SCHEMA = "vdp_transition_trace_schema_v1"
SOURCE_COMMIT = "a1fb3527bb7c12ce23aa2fb49d66f6380c463c90"
REQUIRED_COMMON = (
    "tool",
    "source_commit",
    "run_id",
    "accepted_step_index",
    "attempt_index",
    "retry_index",
    "t_pre",
    "h_attempt",
    "accepted",
    "rejection_reason",
    "state_component",
    "stage",
)
REQUIRED_STAGES = (
    "step_pre_state",
    "raw_picard_image",
    "truncation_cutoff",
    "insertion_input",
    "insertion_output",
    "right_map_input",
    "right_map_output",
    "normalized_reset_input",
    "normalized_reset_output",
    "next_step_pre_state",
    "acceptance_predicate",
)


def encode_float(value: Any) -> dict[str, str]:
    """Serialize one finite binary64 value in max-digits and hex forms."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("trace numbers must be scalar")
        value = value.detach().cpu().item()
    number = float(value)
    if not torch.isfinite(torch.tensor(number, dtype=torch.float64)):
        raise ValueError("trace numbers must be finite")
    return {"decimal": format(number, ".17g"), "hex": number.hex()}


def decode_float(value: Mapping[str, Any]) -> float:
    """Decode and cross-check the decimal/hex representations."""
    if set(value) != {"decimal", "hex"}:
        raise ValueError("encoded float must contain exactly decimal and hex")
    decimal = float(str(value["decimal"]))
    hexadecimal = float.fromhex(str(value["hex"]))
    if decimal != hexadecimal:
        raise ValueError("decimal and hex trace encodings differ")
    return decimal


def encode_interval(interval: Interval) -> dict[str, dict[str, str]]:
    lower = encode_float(interval.lo)
    upper = encode_float(interval.hi)
    if decode_float(lower) > decode_float(upper):
        raise ValueError("trace interval is reversed")
    return {"lower": lower, "upper": upper}


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def _interval_or_none(interval: Interval | None) -> dict[str, Any] | None:
    return encode_interval(interval) if interval is not None else None


def _scalar(value: Any) -> float:
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _model_constant(model: Any) -> float:
    zero = (0,) * model.n_vars
    coefficient = model.polynomial.terms.get(zero)
    return 0.0 if coefficient is None else _scalar(coefficient)


class TransitionTraceWriter:
    """Write deterministic JSONL/CSV without modifying numerical objects."""

    def __init__(self, output_dir: Path, *, run_id: str, source_commit: str = SOURCE_COMMIT):
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"refusing non-empty trace directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.source_commit = str(source_commit)
        self._transitions = (self.output_dir / "transitions.jsonl").open("w", encoding="utf-8")
        self._terms = (self.output_dir / "polynomial_terms.jsonl").open("w", encoding="utf-8")
        self._discarded = (self.output_dir / "discarded_terms.jsonl").open("w", encoding="utf-8")
        self._remainders = (self.output_dir / "remainders.jsonl").open("w", encoding="utf-8")
        self._attempt_handle = (self.output_dir / "acceptance_attempts.csv").open("w", newline="", encoding="utf-8")
        self._attempt_fields = [
            "tool", "source_commit", "run_id", "accepted_step_index", "attempt_index", "retry_index",
            "t_pre_decimal", "t_pre_hex", "h_attempt_decimal", "h_attempt_hex", "accepted",
            "rejection_reason", "state_component", "stage", "validation_status", "subset_margin_x", "subset_margin_y",
        ]
        self._attempt_writer = csv.DictWriter(self._attempt_handle, fieldnames=self._attempt_fields, lineterminator="\n")
        self._attempt_writer.writeheader()
        self._attempt_index = 0
        self.counts = {"transitions": 0, "polynomial_terms": 0, "discarded_terms": 0, "remainders": 0, "attempts": 0}
        self._seen_terms: set[tuple[Any, ...]] = set()
        self._closed = False
        schema = {
            "schema": SCHEMA,
            "required_common_fields": list(REQUIRED_COMMON),
            "required_stages": list(REQUIRED_STAGES),
            "number_encoding": {"decimal": "binary64 max_digits10 string", "hex": "Python hexfloat string"},
            "exponent_order": "tuple/list in the recorded basis_variable_order",
            "null_policy": "unavailable fields are null with an explicit reason; cross-field substitution is forbidden",
        }
        (self.output_dir / "trace_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _base(
        self,
        *,
        step: int,
        attempt: int,
        retry: int,
        t_pre: float,
        h_attempt: float,
        accepted: bool,
        rejection_reason: str,
        component: int,
        stage: str,
    ) -> dict[str, Any]:
        return {
            "tool": "torch",
            "source_commit": self.source_commit,
            "run_id": self.run_id,
            "accepted_step_index": int(step),
            "attempt_index": int(attempt),
            "retry_index": int(retry),
            "t_pre": encode_float(t_pre),
            "h_attempt": encode_float(h_attempt),
            "accepted": bool(accepted),
            "rejection_reason": str(rejection_reason),
            "state_component": int(component),
            "stage": str(stage),
        }

    def _write_transition(self, row: dict[str, Any]) -> None:
        missing = set(REQUIRED_COMMON) - row.keys()
        if missing:
            raise ValueError(f"transition missing required fields: {sorted(missing)}")
        self._transitions.write(_json_line(row))
        self.counts["transitions"] += 1

    def emit_missing(self, *, reason: str, **base: Any) -> None:
        row = self._base(**base)
        row.update(
            {
                "basis_variable_order": ["u0", "u1", "tau"],
                "center": None,
                "normalization_scale": None,
                "support_size": None,
                "polynomial_range": None,
                "remainder": None,
                "cutoff_discarded_terms": None,
                "truncation_discarded_terms": None,
                "insertion_truncation_remainder": None,
                "right_map_remainder": None,
                "integration_overflow": None,
                "self_map_candidate_box": None,
                "self_map_image": None,
                "violation_margin": None,
                "unavailable_reason": reason,
            }
        )
        self._write_transition(row)

    def emit_tmv(
        self,
        tmv: TMVector,
        *,
        centers: Sequence[float] | None = None,
        scales: Sequence[float] | None = None,
        **base: Any,
    ) -> None:
        if not isinstance(tmv, TMVector) or not tmv.models:
            raise ValueError("trace TMVector must be nonempty")
        basis = [f"u{index}" for index in range(tmv.n_vars)]
        if tmv.n_vars == 3:
            basis = ["u0", "u1", "tau"]
        for component, model in enumerate(tmv):
            component_base = dict(base, component=component)
            center = centers[component] if centers is not None and component < len(centers) else _model_constant(model)
            scale = scales[component] if scales is not None and component < len(scales) else None
            polynomial_range = model.polynomial.evaluate_interval(model.domain)
            row = self._base(**component_base)
            row.update(
                {
                    "basis_variable_order": basis,
                    "center": encode_float(center),
                    "normalization_scale": encode_float(scale) if scale is not None else None,
                    "support_size": len(model.polynomial.terms),
                    "polynomial_range": encode_interval(polynomial_range),
                    "remainder": encode_interval(model.remainder),
                    "cutoff_discarded_terms": None,
                    "truncation_discarded_terms": None,
                    "insertion_truncation_remainder": None,
                    "right_map_remainder": encode_interval(model.remainder) if base["stage"] == "right_map_output" else None,
                    "integration_overflow": None,
                    "self_map_candidate_box": None,
                    "self_map_image": None,
                    "violation_margin": None,
                }
            )
            self._write_transition(row)
            remainder_row = {**self._base(**component_base), "record_type": "remainder", "interval": encode_interval(model.remainder)}
            self._remainders.write(_json_line(remainder_row))
            self.counts["remainders"] += 1
            for term_index, (exponent, coefficient) in enumerate(sorted(model.polynomial.terms.items())):
                key = (
                    base["step"],
                    base["attempt"],
                    base["stage"],
                    component,
                    tuple(exponent),
                )
                if key in self._seen_terms:
                    raise ValueError(f"duplicate trace exponent: {key}")
                self._seen_terms.add(key)
                term_row = {
                    **self._base(**component_base),
                    "record_type": "polynomial_term",
                    "term_index": term_index,
                    "exponent_tuple": list(exponent),
                    "degree": sum(exponent),
                    "coefficient": {"lower": encode_float(coefficient), "upper": encode_float(coefficient)},
                }
                self._terms.write(_json_line(term_row))
                self.counts["polynomial_terms"] += 1

    def _record_attempts(self, *, step: int, t_pre: float, diagnostics: Sequence[Mapping[str, Any]], segment: FlowpipeSegment) -> int:
        validation = [row for row in diagnostics if row.get("phase") == "remainder_validation"]
        if not validation:
            validation = [{}]
        final_attempt = self._attempt_index + len(validation) - 1
        for retry, row in enumerate(validation):
            h = float(row.get("h_try", row.get("h", segment.h)))
            status = str(row.get("validation_status", ""))
            accepted = status.lower() == "validated" and retry == len(validation) - 1 and segment.status == "validated"
            reason = str(row.get("rejection_reason", "" if accepted else segment.message))
            row_lo = row.get("picard_image_remainder_lo")
            row_hi = row.get("picard_image_remainder_hi")
            image_pair: list[Any]
            if (
                isinstance(row_lo, Sequence)
                and isinstance(row_hi, Sequence)
                and row_lo
                and row_hi
                and isinstance(row_lo[0], Sequence)
                and isinstance(row_hi[0], Sequence)
            ):
                image_pair = [list(row_lo[0]), list(row_hi[0])]
            else:
                image_pair = list(segment.picard_image_remainder or [])
            component_margins = []
            if len(image_pair) == 2:
                component_margins = [
                    min(float(image_pair[0][index]) + 1e-4, 1e-4 - float(image_pair[1][index]))
                    for index in range(min(len(image_pair[0]), len(image_pair[1])))
                ]
            encoded_t = encode_float(t_pre)
            encoded_h = encode_float(h)
            self._attempt_writer.writerow(
                {
                    "tool": "torch",
                    "source_commit": self.source_commit,
                    "run_id": self.run_id,
                    "accepted_step_index": step,
                    "attempt_index": self._attempt_index,
                    "retry_index": retry,
                    "t_pre_decimal": encoded_t["decimal"],
                    "t_pre_hex": encoded_t["hex"],
                    "h_attempt_decimal": encoded_h["decimal"],
                    "h_attempt_hex": encoded_h["hex"],
                    "accepted": accepted,
                    "rejection_reason": reason,
                    "state_component": -1,
                    "stage": "acceptance_predicate",
                    "validation_status": status,
                    "subset_margin_x": component_margins[0] if len(component_margins) > 0 else "",
                    "subset_margin_y": component_margins[1] if len(component_margins) > 1 else "",
                }
            )
            self._attempt_index += 1
            self.counts["attempts"] += 1
        return final_attempt

    def record_step(
        self,
        *,
        step: int,
        t_pre: float,
        current: TMVector | Sequence[Interval],
        previous_state: FlowstarNormalFlowpipeState | None,
        segment: FlowpipeSegment,
        diagnostics: Sequence[Mapping[str, Any]],
        accepted: bool,
        attempted_h: float,
        order: int,
    ) -> None:
        final_attempt = self._record_attempts(step=step, t_pre=t_pre, diagnostics=diagnostics, segment=segment)
        retry = max(int(segment.step_rejections), 0)
        h_final = float(segment.h if segment.h else attempted_h)
        reason = "" if accepted else str(segment.message)
        common = {
            "step": step,
            "attempt": final_attempt,
            "retry": retry,
            "t_pre": t_pre,
            "h_attempt": h_final,
            "accepted": accepted,
            "rejection_reason": reason,
        }
        if isinstance(current, TMVector):
            current_tmv = current
        else:
            current_tmv = TMVector.identity(list(current), order=order)
        previous_centers = previous_state.center if previous_state is not None else None
        previous_scales = previous_state.scales if previous_state is not None else None
        self.emit_tmv(current_tmv, stage="step_pre_state", centers=previous_centers, scales=previous_scales, **common)
        if previous_state is not None:
            self.emit_tmv(previous_state.tmv_pre, stage="right_map_input", centers=previous_centers, scales=previous_scales, **common)
        else:
            self.emit_missing(stage="right_map_input", component=-1, reason="initial step has no historical right-map state", **common)
            self.emit_missing(stage="right_map_output", component=-1, reason="initial step has no historical right-map state", **common)

        self.emit_tmv(segment.tm, stage="raw_picard_image", **common)
        self.emit_missing(
            stage="truncation_cutoff",
            component=-1,
            reason="aggregate range records exist in runner range_trace; per-term records are emitted by the call44 lineage recorder only",
            **common,
        )
        if segment.endpoint_raw_tm is not None:
            self.emit_tmv(segment.endpoint_raw_tm, stage="insertion_input", **common)
            self.emit_tmv(segment.endpoint_raw_tm, stage="normalized_reset_input", **common)
        else:
            self.emit_missing(stage="insertion_input", component=-1, reason="rejected segment did not publish raw endpoint", **common)
            self.emit_missing(stage="normalized_reset_input", component=-1, reason="rejected segment did not publish raw endpoint", **common)
        next_state = segment.flowstar_normal_state
        if next_state is not None:
            self.emit_tmv(next_state.tmv_pre, stage="insertion_output", centers=next_state.center, scales=next_state.scales, **common)
            self.emit_tmv(next_state.tmv_right, stage="right_map_output", centers=next_state.center, scales=next_state.scales, **common)
        else:
            self.emit_missing(stage="insertion_output", component=-1, reason="rejected segment has no next normalized state", **common)
        if segment.reset_tm is not None:
            centers = next_state.center if next_state is not None else None
            scales = next_state.scales if next_state is not None else None
            self.emit_tmv(segment.reset_tm, stage="normalized_reset_output", centers=centers, scales=scales, **common)
            self.emit_tmv(segment.reset_tm, stage="next_step_pre_state", centers=centers, scales=scales, **common)
        else:
            self.emit_missing(stage="normalized_reset_output", component=-1, reason="rejected segment has no reset", **common)
            self.emit_missing(stage="next_step_pre_state", component=-1, reason="rejected segment has no next pre-state", **common)

        image_pair = list(segment.picard_image_remainder or [])
        component_count = min(len(image_pair[0]), len(image_pair[1])) if len(image_pair) == 2 else 0
        target = [Interval(-1e-4, 1e-4) for _ in range(max(component_count, 2))]
        images = [Interval(image_pair[0][index], image_pair[1][index]) for index in range(component_count)]
        margins = [
            min(float(image.lo) + 1e-4, 1e-4 - float(image.hi))
            for image in images
        ]
        for component in range(max(len(images), 2)):
            row = self._base(component=component, stage="acceptance_predicate", **common)
            image = images[component] if component < len(images) else None
            candidate = target[component] if component < len(target) else None
            margin = margins[component] if component < len(margins) else None
            row.update(
                {
                    "basis_variable_order": ["u0", "u1", "tau"],
                    "center": None,
                    "normalization_scale": None,
                    "support_size": None,
                    "polynomial_range": None,
                    "remainder": _interval_or_none(image),
                    "cutoff_discarded_terms": None,
                    "truncation_discarded_terms": None,
                    "insertion_truncation_remainder": None,
                    "right_map_remainder": None,
                    "integration_overflow": None,
                    "self_map_candidate_box": _interval_or_none(candidate),
                    "self_map_image": _interval_or_none(image),
                    "violation_margin": encode_float(margin) if margin is not None else None,
                }
            )
            self._write_transition(row)

        discarded_row = {
            **self._base(component=-1, stage="truncation_cutoff", **common),
            "record_type": "discarded_term",
            "availability": False,
            "discarded_term": None,
            "reason": "transition trace does not fabricate per-term attribution; terminal call44 lineage is recorded separately",
        }
        self._discarded.write(_json_line(discarded_row))
        self.counts["discarded_terms"] += 1
        for handle in (self._transitions, self._terms, self._discarded, self._remainders, self._attempt_handle):
            handle.flush()

    def close(self, *, result_summary: Mapping[str, Any]) -> None:
        if self._closed:
            return
        for handle in (self._transitions, self._terms, self._discarded, self._remainders, self._attempt_handle):
            handle.close()
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(self.output_dir.iterdir())
            if path.is_file() and path.name != "run_metadata.json"
        }
        metadata = {
            "schema": "vdp_torch_observation_run_v1",
            "tool": "torch",
            "source_commit": self.source_commit,
            "run_id": self.run_id,
            "observation_only": True,
            "counts": self.counts,
            "result_summary": {key: value for key, value in result_summary.items() if key != "runtime_s"},
            "file_sha256": hashes,
        }
        (self.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._closed = True
