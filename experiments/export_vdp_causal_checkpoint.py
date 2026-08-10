#!/usr/bin/env python3
"""Export the authoritative Torch side of the first Flow*/Torch schedule split."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import run_vdp_dense_backend as authoritative
from torch_tm_flowpipe import DenseRangePolicy, Interval, PolynomialODE, TMVector
from torch_tm_flowpipe.batched_dense_tm import (
    BatchedTaylorModel,
    DenseExecutionCounters,
    dense_picard_validate_step,
    dense_polynomial_picard,
    sparse_tmvector_to_dense,
)
from torch_tm_flowpipe.flowpipe import flowpipe_step_flowstar_style_adaptive


TARGET_STEP = 12
TARGET_TIME = 0.18187433604506256
TARGET_H = 0.019615177354506262


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    return str(value)


def _sha_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _interval(value: Interval) -> list[float]:
    return [float(value.lo.detach().cpu()), float(value.hi.detach().cpu())]


def _sparse_tmv(value: TMVector | None) -> Any:
    if value is None:
        return None
    components = []
    for model in value:
        components.append(
            {
                "terms": [
                    {"exponents": list(exponent), "coefficient": float(coefficient.detach().cpu())}
                    for exponent, coefficient in sorted(model.polynomial.terms.items())
                ],
                "remainder": _interval(model.remainder),
                "domain": [_interval(item) for item in model.domain],
            }
        )
    return {"components": components, "variables": value.n_vars, "states": len(value.models)}


def _dense_model(value: BatchedTaylorModel) -> dict[str, Any]:
    active = torch.any(value.poly.coeffs != 0, dim=(0, 1))
    slots = torch.nonzero(active, as_tuple=False).flatten()
    return {
        "batch": value.poly.batch,
        "states": value.poly.out_dim,
        "variables": value.poly.basis.dim,
        "order": value.poly.basis.order,
        "basis_hash": value.poly.basis.fingerprint,
        "exponents": value.poly.basis.exponents[slots].detach().cpu().tolist(),
        "coefficients": value.poly.coeffs[:, :, slots].detach().cpu().tolist(),
        "coefficient_sha256": _sha_tensor(value.poly.coeffs),
        "remainder_lo": value.rem_lo.detach().cpu().tolist(),
        "remainder_hi": value.rem_hi.detach().cpu().tolist(),
        "domain_lo": value.domain_lo.detach().cpu().tolist(),
        "domain_hi": value.domain_hi.detach().cpu().tolist(),
        "ledger_intervals": _jsonable(value.ledger.intervals()),
    }


def _make_dense_base(
    current: TMVector,
    h: float,
    order: int,
    policy: DenseRangePolicy,
) -> BatchedTaylorModel:
    return sparse_tmvector_to_dense(
        current.extend_domain(Interval(0.0, h)),
        order=order,
        device="cpu",
        dtype=torch.float64,
        counters=DenseExecutionCounters(),
        segment_boundary=True,
        range_policy=policy,
        range_trace=[],
    )


def export(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
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
    h_next = contract["h_max"]
    current_time = 0.0
    schedule: list[dict[str, Any]] = []
    checkpoint: dict[str, Any] | None = None
    previous_segment = None
    previous_step_input_state = None

    for step in range(TARGET_STEP + 1):
        input_state_for_segment = normal_state
        h_try = min(float(h_next), contract["h_max"])
        manual: dict[str, Any] | None = None
        if step == TARGET_STEP:
            if current_time != TARGET_TIME or h_try != TARGET_H:
                raise RuntimeError(
                    f"authoritative checkpoint moved: t={current_time!r}, h={h_try!r}"
                )
            if normal_state is None:
                raise RuntimeError("target checkpoint has no normalized carry state")
            current_tm = normal_state.normalized_initial_tm(contract["requested_order"])
            base_for_iterations = _make_dense_base(
                current_tm, h_try, contract["requested_order"], policy
            )
            iterations: list[dict[str, Any]] = []

            def observer(
                iteration: int,
                pre_cutoff: BatchedTaylorModel,
                retained: BatchedTaylorModel,
            ) -> None:
                iterations.append(
                    {
                        "iteration": iteration,
                        "pre_cutoff": _dense_model(pre_cutoff),
                        "retained": _dense_model(retained),
                    }
                )

            candidate, trace = dense_polynomial_picard(
                ode,
                base_for_iterations.without_remainder(),
                tau_index=current_tm.n_vars,
                order=contract["requested_order"],
                iterations=contract["requested_order"],
                cutoff_threshold=contract["cutoff"],
                observer=observer,
            )
            validation_base = _make_dense_base(
                current_tm, h_try, contract["requested_order"], policy
            )
            validation = dense_picard_validate_step(
                ode,
                validation_base,
                h=h_try,
                order=contract["requested_order"],
                tau_index=current_tm.n_vars,
                target_remainder_radius=contract["target_remainder_radius"],
                cutoff_threshold=contract["cutoff"],
                max_validation_attempts=2,
                validation_eps=1e-12,
                validation_mode=contract["validation_mode"],
            )
            manual = {
                "input_normalized_tm": _sparse_tmv(current_tm),
                "validation_base": _dense_model(validation_base),
                "previous_accepted_exact_time_endpoint": (
                    _sparse_tmv(previous_segment.endpoint_raw_tm)
                    if previous_segment is not None
                    else None
                ),
                "previous_step_normalization_center": (
                    list(previous_step_input_state.center)
                    if previous_step_input_state is not None
                    else None
                ),
                "previous_step_normalization_scale": (
                    list(previous_step_input_state.scales)
                    if previous_step_input_state is not None
                    else None
                ),
                "normalization_center": list(normal_state.center),
                "normalization_scale": list(normal_state.scales),
                "right_map_input": _sparse_tmv(normal_state.tmv_right),
                "pre_map_input": _sparse_tmv(normal_state.tmv_pre),
                "normal_domain": [_interval(item) for item in normal_state.domain],
                "picard_iterations": iterations,
                "picard_trace": _jsonable(trace),
                "candidate": _dense_model(candidate),
                "validation_status": validation.status,
                "validation_message": validation.message,
                "candidate_remainder_lo": validation.candidate_remainder_lo.detach().cpu().tolist(),
                "candidate_remainder_hi": validation.candidate_remainder_hi.detach().cpu().tolist(),
                "picard_image_remainder_lo": validation.picard_image_remainder_lo.detach().cpu().tolist(),
                "picard_image_remainder_hi": validation.picard_image_remainder_hi.detach().cpu().tolist(),
                "subset_margin": validation.subset_margin.detach().cpu().tolist(),
                "accepted_remainder": _dense_model(validation.segment_tm),
                "exact_time_endpoint": (
                    _dense_model(validation.raw_endpoint)
                    if validation.raw_endpoint is not None
                    else None
                ),
                "symbolic_queue": {
                    "present": False,
                    "reason": "authoritative complete Torch baseline has no symbolic queue",
                },
            }

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
        schedule.append(
            {
                "step": step,
                "t_pre": current_time,
                "h_attempt": h_try,
                "h_accepted": float(segment.h) if accepted else None,
                "accepted": accepted,
                "internal_rejections": int(segment.step_rejections),
            }
        )
        if not accepted:
            raise RuntimeError(f"authoritative prefix rejected at step {step}: {segment.message}")
        if step == TARGET_STEP:
            assert manual is not None
            production_picard = [
                row for row in (segment.backend_trace or []) if row.get("phase") == "polynomial_picard"
            ]
            manual_hash = manual["candidate"]["coefficient_sha256"]
            production_hash = production_picard[-1]["coefficient_sha256"]
            if manual_hash != production_hash:
                raise RuntimeError("observation replay and production Picard coefficients differ")
            checkpoint = {
                "schema": "torch_vdp_causal_checkpoint_v1",
                "source_commit": _git("rev-parse", "HEAD"),
                "tracked_diff_sha256": hashlib.sha256(
                    subprocess.run(
                        ["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True, capture_output=True
                    ).stdout
                ).hexdigest(),
                "observation_only": True,
                "accepted_step_index": step,
                "t_pre": current_time,
                "h_attempt": h_try,
                "production_accepted": accepted,
                "production_subset_margin": _jsonable(segment.subset_margin),
                "production_candidate_remainder": _jsonable(segment.candidate_remainder),
                "production_picard_image_remainder": _jsonable(segment.picard_image_remainder),
                "production_picard_coefficient_sha256": production_hash,
                "manual_picard_coefficient_sha256": manual_hash,
                "next_step_pre_map": _sparse_tmv(segment.flowstar_normal_state.tmv_pre),
                "next_step_right_map": _sparse_tmv(segment.flowstar_normal_state.tmv_right),
                "next_normalization_center": list(segment.flowstar_normal_state.center),
                "next_normalization_scale": list(segment.flowstar_normal_state.scales),
                "endpoint": _sparse_tmv(segment.endpoint_raw_tm),
                "full_segment": _sparse_tmv(segment.tm),
                **manual,
            }
        current_time += float(segment.h)
        previous_segment = segment
        previous_step_input_state = input_state_for_segment
        current = segment.reset_tm
        normal_state = segment.flowstar_normal_state
        h_next = float(segment.next_h)

    if checkpoint is None:
        raise RuntimeError("target checkpoint was not exported")
    checkpoint["schedule_prefix"] = schedule
    output = output_dir / "torch_causal_checkpoint.json"
    output.write_text(json.dumps(_jsonable(checkpoint), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "output": str(output.relative_to(ROOT)),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "accepted_step_index": TARGET_STEP,
        "t_pre": TARGET_TIME,
        "h_attempt": TARGET_H,
        "production_accepted": True,
        "picard_iterations": len(checkpoint["picard_iterations"]),
        "observer_matches_production": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = export(parse_args(argv).output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
