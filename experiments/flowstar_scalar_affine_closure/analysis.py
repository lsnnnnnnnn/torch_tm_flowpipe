"""Strict analysis helpers for the Flow* scalar-affine correctness closure."""

from __future__ import annotations

import math
import re
from decimal import Decimal, localcontext
from typing import Any, Iterable


_INTEGER = re.compile(r"^-?[0-9]+$")
_FLOAT = re.compile(r"^-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?$")


def _value(text: str) -> Any:
    if _INTEGER.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    return text


def _fields(tokens: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = _value(value)
    return result


def parse_trace(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "config": {},
        "official": {},
        "domains": {},
        "stages": {},
        "intervals": [],
        "checks": {},
        "unavailable": {},
        "status": {},
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        kind = parts[0]
        fields = _fields(parts[1:])
        if kind == "TRACE_CONFIG":
            result["config"] = fields
        elif kind == "TRACE_OFFICIAL":
            result["official"] = fields
        elif kind == "TRACE_STATUS":
            result["status"] = fields
        elif kind == "TRACE_DOMAIN":
            stage = str(fields.pop("stage"))
            index = str(fields.pop("index"))
            result["domains"].setdefault(stage, {})[index] = fields
        elif kind in {"TRACE_TERM", "TRACE_REMAINDER", "TRACE_BOX"}:
            stage = str(fields.pop("stage"))
            stage_record = result["stages"].setdefault(
                stage, {"terms": [], "remainders": {}, "boxes": {}}
            )
            if kind == "TRACE_TERM":
                exponents = str(fields.pop("exponents", ""))
                fields["exponents"] = (
                    [] if not exponents else [int(item) for item in exponents.split(",")]
                )
                stage_record["terms"].append(fields)
            elif kind == "TRACE_REMAINDER":
                state = str(fields.pop("state"))
                stage_record["remainders"][state] = fields
            else:
                scope = str(fields.pop("scope"))
                state = str(fields.pop("state"))
                stage_record["boxes"].setdefault(scope, {})[state] = fields
        elif kind == "TRACE_INTERVAL":
            result["intervals"].append(fields)
        elif kind == "TRACE_CHECK":
            result["checks"][str(fields["name"])] = fields["value"]
        elif kind == "TRACE_UNAVAILABLE":
            result["unavailable"][str(fields["stage"])] = str(fields["reason"])
    return result


def parse_oracle(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {"meta": {}, "bounds": {}}
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        raw_fields = {
            token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in parts[1:]
            if "=" in token
        }
        if parts[0] == "ORACLE_META":
            result["meta"] = {key: _value(value) for key, value in raw_fields.items()}
        elif parts[0] == "ORACLE_BOUND":
            name = raw_fields.pop("name")
            result["bounds"][name] = {
                "direction": raw_fields["direction"],
                "decimal": raw_fields["decimal"],
                "binary64": float(raw_fields["binary64"]),
                "binary64_hex": raw_fields["binary64_hex"],
            }
    return result


def high_precision_outward_oracle(
    x0_lower: str, x0_upper: str, h: str
) -> dict[str, Any]:
    """Independent >=256-bit fallback used by unit tests, not the MPFR run."""

    with localcontext() as context:
        context.prec = 100
        half = Decimal("0.5")
        time = Decimal(h)

        def exact(x0: str, at: Decimal) -> Decimal:
            return (Decimal(x0) + half) * (Decimal(2) * at).exp() - half

        endpoint_lower_exact = exact(x0_lower, time)
        endpoint_upper_exact = exact(x0_upper, time)
        tube_lower_exact = exact(x0_lower, Decimal(0))
        tube_upper_exact = endpoint_upper_exact

    def down(value: Decimal) -> float:
        return math.nextafter(float(value), -math.inf)

    def up(value: Decimal) -> float:
        return math.nextafter(float(value), math.inf)

    return {
        "label": "high_precision_outward_oracle",
        "precision_decimal_digits": 100,
        "endpoint": [down(endpoint_lower_exact), up(endpoint_upper_exact)],
        "tube": [down(tube_lower_exact), up(tube_upper_exact)],
        "monotonicity": monotonicity_certificate(float(x0_lower), float(x0_upper), float(h)),
    }


def monotonicity_certificate(
    x0_lower: float, x0_upper: float, h: float
) -> dict[str, Any]:
    valid_domain = x0_lower >= -0.5 and x0_lower <= x0_upper and h >= 0
    return {
        "verified": valid_domain,
        "x0_interval": [x0_lower, x0_upper],
        "time_interval": [0.0, h],
        "dx_dx0": "exp(2*t) > 0",
        "dx_dt": "(1 + 2*x0)*exp(2*t) >= 0 on x0 >= -1/2",
        "endpoint_extrema": "(x0_lower,h), (x0_upper,h)",
        "tube_extrema": "(x0_lower,0), (x0_upper,h)",
    }


def containment_defect(
    exported: list[float] | tuple[float, float],
    oracle: list[float] | tuple[float, float],
) -> dict[str, Any]:
    lower_defect = max(0.0, float(exported[0]) - float(oracle[0]))
    upper_defect = max(0.0, float(oracle[1]) - float(exported[1]))
    return {
        "lower_defect": lower_defect,
        "upper_defect": upper_defect,
        "max_defect": max(lower_defect, upper_defect),
        "contained": lower_defect == 0.0 and upper_defect == 0.0,
        "tolerance": None,
    }


def box(trace: dict[str, Any], stage: str, scope: str, state: int = 0) -> list[float]:
    record = trace["stages"][stage]["boxes"][scope][str(state)]
    return [float(record["lower"]), float(record["upper"])]


def endpoint_corner_defect(
    trace: dict[str, Any], stage: str, oracle_endpoint: list[float]
) -> dict[str, Any]:
    lower_box = box(trace, stage, "endpoint_lower_corner")
    upper_box = box(trace, stage, "endpoint_upper_corner")
    lower_defect = max(0.0, lower_box[0] - oracle_endpoint[0], oracle_endpoint[0] - lower_box[1])
    upper_defect = max(0.0, upper_box[0] - oracle_endpoint[1], oracle_endpoint[1] - upper_box[1])
    return {
        "lower_corner_box": lower_box,
        "upper_corner_box": upper_box,
        "lower_defect": lower_defect,
        "upper_defect": upper_defect,
        "max_defect": max(lower_defect, upper_defect),
        "contained": lower_defect == 0.0 and upper_defect == 0.0,
        "tolerance": None,
    }


def first_loss_rows(
    trace: dict[str, Any], oracle_endpoint: list[float]
) -> list[dict[str, Any]]:
    refinement_stages = sorted(
        (
            name
            for name in trace["stages"]
            if re.fullmatch(r"refinement_[0-9]+_accepted_tmv", name)
        ),
        key=lambda name: int(name.split("_")[1]),
    )
    stages = [
        "candidate_remainder_seed",
        "picard_ctrunc_raw",
        "validated_picard_image",
        *refinement_stages,
        "accepted_native_tmv_pre",
        "composed_flowpipe",
    ]
    source_locations = {
        "candidate_remainder_seed": "Continuous.cpp:961-969",
        "picard_ctrunc_raw": "Continuous.cpp:966-982",
        "validated_picard_image": "Continuous.cpp:969-1005",
        "accepted_native_tmv_pre": "Continuous.cpp:1038-1040",
        "composed_flowpipe": "Continuous.cpp:386-390",
    }
    rows: list[dict[str, Any]] = []
    loss_seen = False
    for stage in stages:
        defect = endpoint_corner_defect(trace, stage, oracle_endpoint)
        first_loss = not defect["contained"] and not loss_seen
        loss_seen = loss_seen or not defect["contained"]
        if stage.startswith("refinement_"):
            source = "Continuous.cpp:1013-1029"
            mathematical_object = "accepted refinement of stored interval remainder"
        elif stage == "candidate_remainder_seed":
            source = source_locations[stage]
            mathematical_object = "order-limited Picard polynomial plus candidate remainder"
        elif stage == "validated_picard_image":
            source = source_locations[stage]
            mathematical_object = "candidate Picard image after cutoff/truncation and roundoff interval"
        elif stage == "picard_ctrunc_raw":
            source = source_locations[stage]
            mathematical_object = "Picard image with combined cutoff/truncation remainder before roundoff difference"
        elif stage == "accepted_native_tmv_pre":
            source = source_locations[stage]
            mathematical_object = "accepted preconditioned flowpipe before composition"
        else:
            source = source_locations[stage]
            mathematical_object = "composed accepted flowpipe"
        rows.append(
            {
                "path": "generated-stock diagnostic",
                "stage": stage,
                "flowstar_source": source,
                "mathematical_object": mathematical_object,
                "oracle_object": "closed-form endpoint corners at requested h",
                **defect,
                "first_loss": first_loss,
            }
        )
    return rows


def validate_field_separation(trace: dict[str, Any]) -> dict[str, Any]:
    endpoint_raw = box(trace, "endpoint_raw", "full_initial_interval")
    endpoint_collapsed = box(trace, "endpoint_collapsed", "domain")
    full_tube = box(trace, "composed_flowpipe", "domain")
    required_unavailable = {
        "endpoint_tightened": "no_distinct_stock_field",
        "repaired_hull": "prohibited_not_computed",
    }
    if any(trace["unavailable"].get(key) != value for key, value in required_unavailable.items()):
        raise ValueError("tightened/repaired field contract is not explicit")
    return {
        "endpoint_raw": endpoint_raw,
        "endpoint_collapsed": endpoint_collapsed,
        "endpoint_tightened": {"availability": "unavailable"},
        "repaired_hull": {"availability": "unavailable", "computed": False},
        "last_segment": full_tube,
        "full_tube": full_tube,
        "fields_are_separate": True,
    }


def term_map(trace: dict[str, Any], stage: str) -> dict[tuple[int, ...], tuple[float, float]]:
    return {
        tuple(term["exponents"]): (
            float(term["coefficient_lower"]),
            float(term["coefficient_upper"]),
        )
        for term in trace["stages"][stage]["terms"]
        if int(term["state"]) == 0
    }
