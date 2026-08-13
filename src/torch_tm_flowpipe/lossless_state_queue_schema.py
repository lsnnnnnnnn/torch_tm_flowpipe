"""Canonical binary helpers for the audit-only Flow*/Torch state bridge.

The on-disk schema is intentionally simple: ordered UTF-8 ``key=value``
records.  Every numeric payload is an exact MPFR-style dyadic tuple
``precision:sign:hex_mantissa:binary_exponent``.  Decimal text is never a
canonical coefficient representation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

from .flowpipe import FlowstarNormalFlowpipeState
from .interval import Interval
from .tm_vector import TMVector


SCHEMA = "flowstar_lossless_state_queue_v1"


def encode_binary64(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("canonical dyadic values must be finite")
    sign = -1 if math.copysign(1.0, value) < 0.0 else 1
    if value == 0.0:
        # MPFR's canonical zero exponent is MPFR_EXP_MIN on this pinned ABI.
        return f"53:{sign}:0:-1073741823"
    fraction, exponent = math.frexp(abs(value))
    mantissa = int(math.ldexp(fraction, 53))
    if not 2**52 <= mantissa < 2**53:
        raise AssertionError("binary64 significand normalization failed")
    return f"53:{sign}:{mantissa:x}:{exponent - 53}"


def decode_binary64_exact(encoded: str) -> float:
    fields = encoded.split(":")
    if len(fields) != 4:
        raise ValueError("canonical dyadic must contain four fields")
    precision_text, sign_text, mantissa_text, exponent_text = fields
    if precision_text != "53" or sign_text not in {"-1", "1"}:
        raise ValueError("only finite precision-53 bridge values map to binary64")
    try:
        mantissa = int(mantissa_text, 16)
        exponent = int(exponent_text, 10)
    except ValueError as error:
        raise ValueError("invalid canonical dyadic integer") from error
    if mantissa < 0:
        raise ValueError("canonical mantissa must be unsigned")
    sign = int(sign_text)
    if mantissa == 0:
        value = math.copysign(0.0, float(sign))
    else:
        try:
            value = math.ldexp(float(mantissa), exponent)
        except OverflowError as error:
            raise ValueError("canonical dyadic is outside binary64 range") from error
        value = math.copysign(value, float(sign))
    if not math.isfinite(value) or encode_binary64(value) != encoded:
        raise ValueError("canonical dyadic is not exactly representable as binary64")
    return value


def parse_records(text: str) -> dict[str, str]:
    records: dict[str, str] = {}
    if not text or not text.endswith("\n"):
        raise ValueError("canonical schema must be nonempty and newline-terminated")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.count("=") != 1:
            raise ValueError(f"invalid key=value record at line {line_number}")
        key, value = line.split("=", 1)
        if not key or not value:
            raise ValueError(f"empty key/value at line {line_number}")
        if key in records:
            raise ValueError(f"duplicate field: {key}")
        records[key] = value
    if records.get("schema") != SCHEMA:
        raise ValueError("wrong or missing schema")
    return records


def parse_file(path: Path) -> dict[str, str]:
    return parse_records(path.read_text(encoding="utf-8"))


def _interval_values(interval: Interval) -> tuple[float, float]:
    return float(interval.lo.detach().cpu()), float(interval.hi.detach().cpu())


def _append_interval(rows: list[tuple[str, str]], key: str, interval: Interval) -> None:
    lower, upper = _interval_values(interval)
    if lower > upper:
        raise ValueError(f"inverted Torch interval: {key}")
    rows.append((f"{key}.lo", encode_binary64(lower)))
    rows.append((f"{key}.hi", encode_binary64(upper)))


def _append_tmv(rows: list[tuple[str, str]], prefix: str, tmv: TMVector) -> None:
    rows.append((f"{prefix}.component_count", str(len(tmv))))
    for component, model in enumerate(tmv):
        base = f"{prefix}.component.{component}"
        terms = sorted(model.polynomial.terms.items())
        rows.append((f"{base}.term_count", str(len(terms))))
        for term_index, (exponents, coefficient) in enumerate(terms):
            term = f"{base}.term.{term_index}"
            rows.append((f"{term}.exponents", ",".join(str(int(v)) for v in exponents)))
            rows.append((f"{term}.total_degree", str(sum(int(v) for v in exponents))))
            rows.append(
                (
                    f"{term}.coefficient",
                    encode_binary64(float(coefficient.detach().cpu())),
                )
            )
        _append_interval(rows, f"{base}.remainder", model.remainder)


def _settings_rows(template: Mapping[str, str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in template.items():
        if key.startswith("settings."):
            rows.append((key, value))
    required = {
        "settings.order",
        "settings.term_ordering",
        "settings.cutoff.lo",
        "settings.cutoff.hi",
        "settings.local_step",
    }
    if not required <= {key for key, _ in rows}:
        raise ValueError("template is missing frozen setting fields")
    return rows


def export_torch_initial_state(template_path: Path, output_path: Path) -> dict[str, object]:
    """Export a real Torch float64 normal state into the shared schema."""

    state = FlowstarNormalFlowpipeState.from_initial_box(
        [Interval(1.1, 1.4), Interval(2.35, 2.45)],
        order=4,
    )
    template = parse_file(template_path)
    rows: list[tuple[str, str]] = [
        ("schema", SCHEMA),
        ("producer", "torch_binary64"),
        ("phase", "torch_state"),
        ("step", str(state.step_index)),
        ("local_time", encode_binary64(0.0)),
        ("state_dimension", str(len(state.tmv_pre))),
        ("variable_dimension", str(len(state.domain))),
    ]
    rows.extend(_settings_rows(template))
    rows.extend(
        [
            ("flowpipe.safety", "1"),
            ("flowpipe.constrained", "0"),
            ("flowpipe.domain_count", str(len(state.domain))),
        ]
    )
    for index, interval in enumerate(state.domain):
        _append_interval(rows, f"flowpipe.domain.{index}", interval)
    _append_tmv(rows, "flowpipe.tmvPre", state.tmv_pre)
    _append_tmv(rows, "flowpipe.tmv", state.tmv_right)
    rows.extend(
        [
            ("queue.max_size", str(state.symbolic_queue_max_size)),
            ("queue.scalars_count", str(len(state.center))),
        ]
    )
    for index in range(len(state.center)):
        rows.append((f"queue.scalar.{index}", encode_binary64(1.0)))
    rows.extend([("queue.J_count", "0"), ("queue.Phi_L_count", "0")])
    keys = [key for key, _ in rows]
    if len(keys) != len(set(keys)):
        raise AssertionError("Torch exporter generated duplicate fields")
    output_path.write_text(
        "".join(f"{key}={value}\n" for key, value in rows),
        encoding="utf-8",
    )
    coefficient_count = sum(
        1 for key, _ in rows if key.endswith(".coefficient")
    )
    remainder_endpoint_count = sum(
        1
        for key, _ in rows
        if ".remainder." in key and key.endswith((".lo", ".hi"))
    )
    return {
        "producer": "torch_binary64",
        "state_dimension": len(state.tmv_pre),
        "variable_dimension": len(state.domain),
        "coefficient_count": coefficient_count,
        "remainder_endpoint_count": remainder_endpoint_count,
    }


def iter_canonical_dyadics(records: Mapping[str, str]) -> Iterable[tuple[str, str]]:
    for key, value in records.items():
        if key == "local_time" or key == "settings.local_step":
            yield key, value
        elif key.endswith((".coefficient", ".lo", ".hi")) or key.startswith("queue.scalar."):
            yield key, value


__all__ = [
    "SCHEMA",
    "decode_binary64_exact",
    "encode_binary64",
    "export_torch_initial_state",
    "iter_canonical_dyadics",
    "parse_file",
    "parse_records",
]
