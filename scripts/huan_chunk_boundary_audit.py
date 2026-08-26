#!/usr/bin/env python3
"""Exercise several awkward member chunks and B=1-in-B>1 identity."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any


def run(engine_root: Path, device: str) -> dict[str, Any]:
    sys.path.insert(0, str(engine_root / "src"))
    torch = importlib.import_module("torch")
    poly = importlib.import_module("flowstar_gpu.polynomial")
    safety = importlib.import_module("flowstar_gpu.safety")
    composition = importlib.import_module("flowstar_gpu.composition")
    config = importlib.import_module("flowstar_gpu.config")
    monomials = importlib.import_module("flowstar_gpu.monomials")

    n, order, batch = 3, 4, 2
    tables = monomials.build_tables(n, order).to(device)
    step = poly.build_step_tables(tables, 0.1)
    schedule = composition.build_schedule(n, order, device)
    generator = torch.Generator().manual_seed(20260826)
    ts = int(tables.sp_prefix_len_cpu[order + 1])
    coeffs = (0.1 * torch.randn(batch, n, ts, generator=generator, dtype=torch.float64)).to(device)
    radii = (1e-8 * torch.rand(batch, n, generator=generator, dtype=torch.float64)).to(device)
    rem = torch.stack((-radii, radii), dim=-1)
    settings = config.Settings(step=0.1, order=order, cutoff=1e-10, device=device)
    pair_count = int(tables.sp_pair_i.shape[0])

    original_budget = safety._SPEC_IMG_BUDGET_BYTES
    rows: list[dict[str, Any]] = []
    try:
        safety._SPEC_IMG_BUDGET_BYTES = float(2**63)
        reference = safety._spatial_images(coeffs, rem, tables, step, schedule, settings)
        per_member_b2 = batch * pair_count * coeffs.element_size()
        for chunk in (1, 2, 3, 5, 7):
            safety._SPEC_IMG_BUDGET_BYTES = float(per_member_b2 * chunk)
            actual = safety._spatial_images(coeffs, rem, tables, step, schedule, settings)
            rows.append(
                {
                    "requested_member_chunk": chunk,
                    "coefficients_bitwise": bool(torch.equal(actual[0], reference[0])),
                    "remainders_bitwise": bool(torch.equal(actual[1], reference[1])),
                    "awkward_nondivisor_of_degree4_level": 15 % chunk != 0,
                }
            )

        safety._SPEC_IMG_BUDGET_BYTES = float(pair_count * coeffs.element_size() * 3)
        solo = safety._spatial_images(coeffs[:1], rem[:1], tables, step, schedule, settings)
        safety._SPEC_IMG_BUDGET_BYTES = float(per_member_b2 * 3)
        batched = safety._spatial_images(coeffs, rem, tables, step, schedule, settings)
    finally:
        safety._SPEC_IMG_BUDGET_BYTES = original_budget

    lane = {
        "coefficients_bitwise": bool(torch.equal(solo[0][0], batched[0][0])),
        "remainders_bitwise": bool(torch.equal(solo[1][0], batched[1][0])),
    }
    passed = all(row["coefficients_bitwise"] and row["remainders_bitwise"] for row in rows) and all(lane.values())
    return {
        "schema": "torch_tm_flowpipe.huan_chunk_boundary_audit/1",
        "engine_root": str(engine_root.resolve()),
        "device": device,
        "basis_size": ts,
        "degree4_level_members": 15,
        "chunk_cases": rows,
        "b1_embedded_in_b2": lane,
        "gate_passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.engine_root, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"device": args.device, "gate_passed": payload["gate_passed"]}, sort_keys=True))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
