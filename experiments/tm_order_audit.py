#!/usr/bin/env python3
"""Audit the actual polynomial dependency degree retained by torch_tm_flowpipe.

This script is intentionally diagnostic-only.  The comparison CSV has an
``order`` column, but this audit makes explicit what the code actually retains:
all monomials whose *total degree* is <= order over the active dependency
variables.  For dependency-preserving runs, each segment temporarily adds a
local time variable tau; after endpoint substitution tau is dropped.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required for tm_order_audit.py") from exc

import torch

torch.set_num_threads(1)

from torch_tm_flowpipe import Interval, TMVector, flowpipe_multi_step
from torch_tm_flowpipe.ode_examples import harmonic_oscillator_ode, scalar_quadratic_ode, van_der_pol_ode

CONFIG_DIR = REPO_ROOT / "comparisons" / "flowstar" / "configs"
DEFAULT_CONFIGS = [
    CONFIG_DIR / "scalar_quadratic.yaml",
    CONFIG_DIR / "harmonic_oscillator.yaml",
    CONFIG_DIR / "van_der_pol.yaml",
    CONFIG_DIR / "affine_controlled.yaml",
]

FIELDS = [
    "system", "mode", "h", "steps", "requested_order", "order_semantics",
    "status", "final_width_sum", "final_width_max", "max_final_degree",
    "degree_by_dim", "term_count_by_dim", "remainder_radius_by_dim",
    "active_vars_by_dim", "segment_max_degree", "segment_tau_active_after_drop",
    "validation_attempts",
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def interval_box_from_config(cfg: Mapping[str, Any]) -> list[Interval]:
    init = cfg["initial"]
    return [Interval(init[var][0], init[var][1]) for var in cfg["state_vars"]]


def affine_controlled_folded_ode(x: TMVector, u: TMVector | None = None) -> TMVector:
    return TMVector([0.5 * x[0] + x[1], 0.0 * x[1]])


def ode_for_config(cfg: Mapping[str, Any]) -> Callable[..., TMVector]:
    name = cfg.get("torch_ode", cfg["system"])
    if name == "scalar_quadratic":
        return scalar_quadratic_ode
    if name == "harmonic_oscillator":
        return harmonic_oscillator_ode
    if name == "van_der_pol":
        return van_der_pol_ode
    if name == "affine_controlled_folded":
        return affine_controlled_folded_ode
    raise ValueError(f"unknown torch_ode: {name}")


def metric_indices(cfg: Mapping[str, Any]) -> list[int]:
    state_vars = list(cfg["state_vars"])
    metric_vars = list(cfg.get("metric_vars", state_vars))
    return [state_vars.index(v) for v in metric_vars]


def width(iv: Interval) -> float:
    return float(iv.width().detach().cpu())


def radius(iv: Interval) -> float:
    return float(iv.radius().detach().cpu())


def audit_row(cfg: Mapping[str, Any], *, h: float, steps: int, order: int, mode: str) -> dict[str, Any]:
    result = flowpipe_multi_step(
        ode_for_config(cfg),
        interval_box_from_config(cfg),
        h=h,
        steps=steps,
        order=order,
        mode=mode,
    )
    indices = metric_indices(cfg)
    final_box = result.final_tm.range_box()
    final_widths = [width(final_box[i]) for i in indices]
    final_models = [result.final_tm[i] for i in indices]
    final_degrees = [m.polynomial.degree() for m in final_models]
    final_terms = [len(m.polynomial.terms) for m in final_models]
    final_rems = [radius(m.remainder) for m in final_models]
    final_active = [sorted(m.active_variables()) for m in final_models]
    seg_degrees = []
    tau_active_after_drop = []
    for seg in result.segments:
        seg_degrees.append(max((m.polynomial.degree() for m in seg.tm.models), default=0))
        if seg.tau_index is None:
            tau_active_after_drop.append(False)
        else:
            tau_active_after_drop.append(any(seg.tau_index in m.active_variables() for m in seg.final_tm.models))
    return {
        "system": cfg["system"],
        "mode": mode,
        "h": h,
        "steps": steps,
        "requested_order": order,
        "order_semantics": "total_degree_over_dependency_vars_plus_local_tau_per_segment",
        "status": result.status,
        "final_width_sum": sum(final_widths),
        "final_width_max": max(final_widths) if final_widths else 0.0,
        "max_final_degree": max(final_degrees) if final_degrees else 0,
        "degree_by_dim": repr(final_degrees),
        "term_count_by_dim": repr(final_terms),
        "remainder_radius_by_dim": repr(final_rems),
        "active_vars_by_dim": repr(final_active),
        "segment_max_degree": repr(seg_degrees),
        "segment_tau_active_after_drop": repr(tau_active_after_drop),
        "validation_attempts": result.validation_attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit retained Taylor-model polynomial dependency degree.")
    parser.add_argument("--config", nargs="*", default=None, help="config paths; defaults to bundled Flow* comparison configs")
    parser.add_argument("--all", action="store_true", help="run full configured grids; otherwise first h/steps/order per config")
    parser.add_argument("--csv", default="outputs/tm_order_audit.csv")
    args = parser.parse_args()

    config_paths = [Path(p) for p in args.config] if args.config else DEFAULT_CONFIGS
    rows = []
    for path in config_paths:
        cfg = load_config(path)
        hs = list(cfg["h"])
        steps_list = list(cfg["steps"])
        orders = list(cfg["order"])
        cases = [(float(hs[0]), int(steps_list[0]), int(orders[0]))]
        if args.all:
            cases = [(float(h), int(s), int(o)) for h in hs for s in steps_list for o in orders]
        for h, steps, order in cases:
            for mode in ["range_only", "dependency_preserving"]:
                row = audit_row(cfg, h=h, steps=steps, order=order, mode=mode)
                rows.append(row)
                print(f"{row['system']} {mode} h={h:g} steps={steps} order={order} "
                      f"degree={row['degree_by_dim']} terms={row['term_count_by_dim']} "
                      f"status={row['status']}")
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
