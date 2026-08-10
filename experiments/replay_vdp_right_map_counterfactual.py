#!/usr/bin/env python3
"""Replay a Flow* right map through the authoritative Torch next-step path."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_vdp_dense_backend as authoritative
from torch_tm_flowpipe import DenseRangePolicy, Interval, Polynomial, PolynomialODE, TaylorModel, TMVector
from torch_tm_flowpipe.flowpipe import flowpipe_step_flowstar_style_adaptive


TARGET_STEP = 12


def _load_map(path: Path, order: int) -> TMVector:
    data = json.loads(path.read_text(encoding="utf-8"))
    models = []
    for component in data["components"]:
        domain = [Interval(*bounds) for bounds in component["domain"]]
        terms = {
            tuple(int(value) for value in term["exponents"]): float(term["coefficient"])
            for term in component["terms"]
        }
        models.append(
            TaylorModel(
                Polynomial(terms, n_vars=len(domain)),
                Interval(*component["remainder"]),
                domain,
                order=order,
            )
        )
    return TMVector(models)


def _tmv_payload(value: TMVector | None) -> Any:
    if value is None:
        return None
    return [
        {
            "terms": [
                {"exponents": list(exponent), "coefficient": float(coefficient)}
                for exponent, coefficient in sorted(model.polynomial.terms.items())
            ],
            "remainder": [float(model.remainder.lo), float(model.remainder.hi)],
            "domain": [[float(item.lo), float(item.hi)] for item in model.domain],
        }
        for model in value
    ]


def _tmv_sha(value: TMVector | None) -> str | None:
    if value is None:
        return None
    encoded = json.dumps(_tmv_payload(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _run_lane(replacement_map: TMVector | None) -> dict[str, Any]:
    contract = authoritative.load_contract()
    policy = DenseRangePolicy(
        method="adaptive_subdivision",
        max_depth=1,
        max_leaves=4,
        split_vars=(0, 1),
        trigger="proactive_depth1_on_named_contexts",
        named_contexts=("polynomial_truncation",),
        variable_orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)),
    )
    ode = PolynomialODE.from_system_spec(contract["canonical_system_spec"])
    current: TMVector | list[Interval] = [Interval(*bounds) for bounds in contract["initial_box"]]
    normal_state = None
    h_next = float(contract["h_max"])
    current_time = 0.0
    rows = []
    for step in range(TARGET_STEP + 2):
        substituted = False
        if step == TARGET_STEP and replacement_map is not None:
            if normal_state is None:
                raise RuntimeError("target step has no normal state")
            normal_state = replace(normal_state, tmv_right=replacement_map)
            substituted = True
        h_try = min(h_next, float(contract["h_max"]))
        diagnostics: list[dict[str, Any]] = []
        segment = flowpipe_step_flowstar_style_adaptive(
            ode,
            current,
            h=h_try,
            h_min=contract["h_min"],
            h_max=contract["h_max"],
            order=contract["requested_order"],
            target_remainder_radius=contract["target_remainder_radius"],
            cutoff_threshold=contract["cutoff"],
            max_validation_attempts=2,
            validation_eps=1e-12,
            validation_mode=contract["validation_mode"],
            reset_mode="normalized_insertion",
            step_policy_mode="flowstar_compat",
            flowstar_normal_state=normal_state,
            right_map_center_mode="constant",
            right_map_range_mode="standard",
            tm_backend="dense",
            dense_device="cpu",
            dense_range_policy=policy,
            diagnostics=diagnostics,
            diagnostics_context={"segment_index": step, "t_before": current_time, "mode": "dense"},
        )
        accepted = segment.status == "validated" and segment.reset_tm is not None
        row = {
            "step": step,
            "t_pre": current_time,
            "h_attempt": h_try,
            "accepted": accepted,
            "h_accepted": float(segment.h) if accepted else None,
            "internal_rejections": int(segment.step_rejections),
            "right_map_substituted": substituted,
            "subset_margin": _jsonable(segment.subset_margin),
            "candidate_remainder": _jsonable(segment.candidate_remainder),
            "picard_image_remainder": _jsonable(segment.picard_image_remainder),
            "endpoint_sha256": _tmv_sha(segment.endpoint_raw_tm),
            "reset_sha256": _tmv_sha(segment.reset_tm),
            "next_right_map_sha256": (
                _tmv_sha(segment.flowstar_normal_state.tmv_right)
                if segment.flowstar_normal_state is not None
                else None
            ),
            "next_normalization_center": (
                list(segment.flowstar_normal_state.center)
                if segment.flowstar_normal_state is not None
                else None
            ),
            "next_normalization_scale": (
                list(segment.flowstar_normal_state.scales)
                if segment.flowstar_normal_state is not None
                else None
            ),
        }
        rows.append(row)
        if not accepted:
            break
        current_time += float(segment.h)
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state
        h_next = float(segment.next_h)
    return {
        "completed_through_step_13": len(rows) == TARGET_STEP + 2 and rows[-1]["accepted"],
        "rows": rows,
    }


def replay(right_map_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = authoritative.load_contract()
    replacement = _load_map(right_map_path, int(contract["requested_order"]))
    native = _run_lane(None)
    counterfactual = _run_lane(replacement)
    native_target = native["rows"][TARGET_STEP]
    counter_target = counterfactual["rows"][TARGET_STEP]
    if any(
        native_target[key] != counter_target[key]
        for key in ("accepted", "h_accepted", "subset_margin", "picard_image_remainder")
    ):
        raise RuntimeError("right-map-only substitution perturbed the current validation decision")
    result = {
        "schema": "vdp_flowstar_right_map_to_torch_next_step_v1",
        "right_map_input": str(right_map_path),
        "right_map_input_sha256": hashlib.sha256(right_map_path.read_bytes()).hexdigest(),
        "current_step_validation_unchanged": True,
        "native": native,
        "counterfactual": counterfactual,
    }
    output = output_dir / "right_map_counterfactual.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "current_step_validation_unchanged": True,
        "native_completed_through_step_13": native["completed_through_step_13"],
        "counterfactual_completed_through_step_13": counterfactual["completed_through_step_13"],
        "next_step_native_accepted": native["rows"][-1]["accepted"],
        "next_step_counterfactual_accepted": counterfactual["rows"][-1]["accepted"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(replay(args.right_map.resolve(), args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
