#!/usr/bin/env python3
"""Extract and cross-check the VDP order-4 Torch/stock-Flow* audit contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "benchmarks/canonical.yaml"
MATCHED = ROOT / "benchmarks/three_tool_matched_contract.yaml"
RUNNER = ROOT / "experiments/run_vdp_dense_backend.py"
FLOWPIPE = ROOT / "src/torch_tm_flowpipe/flowpipe.py"
CHECKPOINT = (
    ROOT
    / "evidence/vdp_terminal_range_closure/20260805T055556Z/05_fresh_horizons/"
    "t6p5_proactive_d1_truncation/terminal_checkpoint/terminal_state.json"
)
CHECKPOINT_MANIFEST = CHECKPOINT.with_name("terminal_state_manifest.json")
FROZEN_CHECKPOINT_SHA256 = "dcb8f646d45c9742e0cff23fea596c12e53d8ccd00d1544f70564a44a7463420"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: float) -> dict[str, Any]:
    value = float(value)
    return {"decimal": value, "binary64_hex": value.hex()}


def _field(torch: Any, flowstar: Any, matched: bool, reason: str) -> dict[str, Any]:
    return {"torch": torch, "flowstar": flowstar, "matched": bool(matched), "reason": reason}


def _require(text: str, fragment: str, source: str) -> None:
    if fragment not in text:
        raise ValueError(f"required source fragment missing from {source}: {fragment}")


def build(flowstar_root: Path) -> dict[str, Any]:
    benchmark = flowstar_root / "benchmarks/continuous/vanderpol/vanderpol.cpp"
    continuous_cpp = flowstar_root / "flowstar-toolbox/Continuous.cpp"
    continuous_h = flowstar_root / "flowstar-toolbox/Continuous.h"
    interval_cpp = flowstar_root / "flowstar-toolbox/Interval.cpp"
    include_h = flowstar_root / "flowstar-toolbox/include.h"
    source_paths = [benchmark, continuous_cpp, continuous_h, interval_cpp, include_h]
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    canonical = _read_yaml(CANONICAL)
    preregistered = _read_yaml(MATCHED)
    torch_vdp = canonical["systems"]["van_der_pol"]
    contract_vdp = preregistered["systems"]["van_der_pol"]
    checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    if checkpoint_manifest.get("full_checkpoint_sha256") != FROZEN_CHECKPOINT_SHA256:
        raise ValueError("formal T=6.5 checkpoint full SHA256 changed")
    if checkpoint_manifest.get("payload_sha256") != _sha256(CHECKPOINT):
        raise ValueError("formal T=6.5 checkpoint payload does not match its manifest")
    benchmark_text = benchmark.read_text(encoding="utf-8")
    continuous_text = continuous_cpp.read_text(encoding="utf-8")
    include_text = include_h.read_text(encoding="utf-8")
    interval_text = interval_cpp.read_text(encoding="utf-8")
    runner_text = RUNNER.read_text(encoding="utf-8")
    flowpipe_text = FLOWPIPE.read_text(encoding="utf-8")

    required = [
        (benchmark_text, 'ODE<Real> ode({"y", "(1 - x^2) * y - x", "1"}, vars);', str(benchmark)),
        (benchmark_text, "Interval init_x(1.1, 1.4), init_y(2.35, 2.45);", str(benchmark)),
        (benchmark_text, "Symbolic_Remainder sr(initialSet, 100);", str(benchmark)),
        (continuous_text, "setAdaptiveStepsize(0.002, 0.1, 4);", str(continuous_cpp)),
        (continuous_text, "Interval cutoff_threshold(-1e-10,1e-10);", str(continuous_cpp)),
        (continuous_text, "Interval I(-1e-4,1e-4);", str(continuous_cpp)),
        (include_text, "normal_precision\t=\t53", str(include_h)),
        (interval_text, "MPFR_RNDU", str(interval_cpp)),
        (interval_text, "MPFR_RNDD", str(interval_cpp)),
        (runner_text, '"validation_mode": "flowstar_raw_remainder_compat"', str(RUNNER)),
        (runner_text, '"reset_mode": "normalized_insertion"', str(RUNNER)),
        (flowpipe_text, "FLOWSTAR_COMPAT_STEP_SHRINK = 0.5", str(FLOWPIPE)),
        (flowpipe_text, "FLOWSTAR_COMPAT_STEP_GROW = 1.1", str(FLOWPIPE)),
    ]
    for text, fragment, source in required:
        _require(text, fragment, source)

    if torch_vdp["initial_box"] != contract_vdp["initial_set"]:
        raise ValueError("canonical and preregistered Torch initial sets differ")
    if contract_vdp["ode"] != ["y", "(1 - x^2) * y - x"]:
        raise ValueError("unexpected preregistered VDP ODE")
    if checkpoint["contract"]["initial_box"] != contract_vdp["initial_set"]:
        raise ValueError("checkpoint initial set differs from source contract")
    if checkpoint["contract"]["requested_order"] != contract_vdp["requested_order"]:
        raise ValueError("checkpoint order differs from source contract")

    native_step = contract_vdp["step_policy"]["native_flowstar"]
    initial_remainder = contract_vdp["remainder_initialization"]
    authoritative_switches = {
        "backend_lane": "hybrid_dense_core",
        "tm_backend": "dense",
        "device": "cpu",
        "dtype": "float64",
        "dense_range_method": "adaptive_subdivision",
        "dense_range_trigger": "proactive_depth1_on_named_contexts",
        "dense_range_max_depth": 1,
        "dense_range_max_leaves": 4,
        "dense_range_split_vars": [0, 1],
        "dense_range_contexts": ["polynomial_truncation"],
        "dense_range_variable_orders": [[0, 1, 2], [1, 0, 2], [2, 0, 1]],
        "reset_mode": "normalized_insertion",
        "right_map_center_mode": "constant",
        "right_map_range_mode": "standard",
        "validation_mode": "flowstar_raw_remainder_compat",
        "validation_eps": _number(1e-12),
        "max_validation_attempts": 2,
        "step_policy_mode": "flowstar_compat",
        "endpoint_repair": False,
        "endpoint_tightening": False,
        "external_endpoint_substitution": False,
        "sparse_fallback": False,
        "symbolic_queue_enabled": False,
        "symbolic_queue_max_size_inert": 100,
    }

    fields = {
        "ode": _field(
            contract_vdp["ode"],
            ["y", "(1 - x^2) * y - x", "1"],
            True,
            "The physical x/y equations match exactly; stock adds the clock state t'=1.",
        ),
        "physical_state_order": _field(
            ["x", "y"], ["x", "y"], True, "position/velocity aliases are mapped to stock x/y."
        ),
        "stored_state_order": _field(
            ["x", "y"], ["x", "y", "t"], False, "stock stores an extra clock state used for plotting/safety."
        ),
        "initial_set": _field(
            [[_number(v) for v in bounds] for bounds in contract_vdp["initial_set"]],
            [[_number(1.1), _number(1.4)], [_number(2.35), _number(2.45)], [_number(0.0), _number(0.0)]],
            True,
            "The physical box is identical; stock's clock starts at the point interval zero.",
        ),
        "taylor_order": _field(4, 4, True, "Both use fixed Taylor order four."),
        "retained_degree": _field(
            {"complete_total_degree": 4, "candidate_rhs_degree": 3, "time_integration_restores_degree": 4},
            {"complete_total_degree": 4, "normal_ctrunc_order": 4},
            True,
            "Both retain complete total degree <=4; the Torch raw-RHS replay uses degree 3 before tau integration.",
        ),
        "local_variable_order": _field(
            ["u0", "u1", "tau"], ["tau", "r0", "r1", "r2"], False,
            "Stock places local time at domain index 0 and includes one normalized generator for each x/y/t state; Torch omits the constant clock generator and places tau last.",
        ),
        "local_time_domain": _field(
            {"tau": "[0,h]", "index": 2}, {"tau": "[0,h]", "index": 0}, True,
            "Domains match after the explicit index permutation in field_map.md.",
        ),
        "initial_step": _field(_number(0.1), _number(0.1), True, "Both start at h_max."),
        "h_min": _field(_number(native_step["minimum"]), _number(0.002), True, "Source defaults match."),
        "h_max": _field(_number(native_step["maximum"]), _number(0.1), True, "Source defaults match."),
        "step_shrink": _field(_number(0.5), _number(0.5), True, "Rejected attempts are halved."),
        "step_growth": _field(_number(1.1), _number(1.1), True, "Accepted steps propose 1.1*h capped by h_max."),
        "initial_candidate_remainder": _field(
            [[_number(v) for v in bounds] for bounds in initial_remainder],
            [[_number(-1e-4), _number(1e-4)] for _ in range(3)],
            True,
            "The physical x/y candidate intervals match; stock also assigns the default interval to its clock component.",
        ),
        "cutoff": _field(_number(contract_vdp["cutoff"]), _number(1e-10), True, "Absolute cutoff threshold matches."),
        "truncation": _field(
            {"complete_total_degree": 4, "cutoff": "intervalized", "selected_range": "depth-1 x/y subdivision only for polynomial_truncation"},
            {"complete_total_degree": 4, "cutoff": "intervalized", "selected_range": "stock natural interval evaluation"},
            False,
            "The truncation support and cutoff match, but the authoritative Torch lane evaluates dropped polynomial ranges with four-leaf subdivision; stock does not.",
        ),
        "picard_validation": _field(
            {"iterations": 1, "max_validation_attempts": 2, "predicate": "flowstar_raw_remainder_compat subset of [-1e-4,1e-4]", "epsilon": _number(1e-12)},
            {"iterations": "stock fixed-order Picard construction plus remainder subset validation", "max_validation_attempts": None, "predicate": "stock remainder estimation subset check", "epsilon": None},
            False,
            "The Torch predicate replays stock raw-remainder algebra, but iteration/control fields are not exposed as an identical public stock contract; nulls are not substituted.",
        ),
        "normalized_insertion": _field(
            {"enabled": True, "right_map_center": "constant", "right_map_range": "standard", "reset": "fresh affine center/scale variables"},
            {"enabled": True, "right_map_center": "constant-part removal", "right_map_range": "normal evaluation", "reset": "normalized insertion with tmvPre/tmvRight"},
            False,
            "The broad normalized-insertion structure is analogous, but the evaluated range operators and stored state dimensions are not identical.",
        ),
        "symbolic_remainder_queue": _field(
            {"enabled": False, "checkpoint_present": checkpoint["normal_state"]["symbolic_queue_present"], "configured_max_size_but_inert": 100},
            {"enabled": True, "max_size": 100, "reset_when_J_size_reaches_max": True},
            False,
            "This is a material cross-step representation difference, not a matched field.",
        ),
        "numeric_backend": _field(
            {"scalar": "IEEE-754 binary64 via torch.float64", "interval": "nextafter-safeguarded float64", "proof_level": "safeguarded, not machine-checked directed rounding"},
            {"scalar": "MPFR", "precision_bits": 53, "interval": "MPFR_RNDD/MPFR_RNDU directed rounding"},
            False,
            "Precision is nominally 53 bits, but the interval implementations and guarantees differ.",
        ),
        "endpoint": _field(
            "raw endpoint TM evaluated at tau=h before reset", "composed end-of-segment flowpipe at tau=h", True,
            "Semantic endpoints align; no segment box is substituted for an endpoint.",
        ),
        "segment": _field(
            "validated TM over tau in [0,h]", "validated TM flowpipe over local time in [0,h]", True, "Segment semantics align."
        ),
        "tube": _field(
            "hull of validated segment x/y boxes", "ordered vector of validated Taylor-model flowpipes (plus plot projections)", False,
            "Torch stores an aggregate x/y hull and segment records; stock retains the full ordered flowpipe vector. These are mapped but not equated.",
        ),
    }

    return {
        "schema": "vdp_cross_step_matched_contract_v1",
        "status": "partially_matched_fail_closed",
        "scope": "Van der Pol order-4 native stock Flow* versus authoritative Torch H1/R4 lane",
        "source_identity": {
            "torch_report_head": "455146df23940caa6f168877ffe6ec6f508c43a4",
            "torch_formal_numerical_source": "a1fb3527bb7c12ce23aa2fb49d66f6380c463c90",
            "torch_checkpoint_generating_source": checkpoint["provenance"],
            "flowstar_commit": "b85a3211748cb77b736fe4ad42ee02d8d2b81148",
            "files": {
                str(path): _sha256(path)
                for path in [CANONICAL, MATCHED, RUNNER, FLOWPIPE, CHECKPOINT, CHECKPOINT_MANIFEST, *source_paths]
            },
        },
        "fields": fields,
        "authoritative_torch_switches": authoritative_switches,
        "checkpoint_freeze": {
            "full_sha256": FROZEN_CHECKPOINT_SHA256,
            "current_time": _number(checkpoint["scheduler"]["current_time"]),
            "attempted_h": _number(checkpoint["scheduler"]["h_attempted"]),
            "accepted_segments": checkpoint["scheduler"]["accepted_segment_count"],
            "symbolic_queue_present": checkpoint["normal_state"]["symbolic_queue_present"],
        },
        "all_fields_matched": all(item["matched"] for item in fields.values()),
        "behavioral_comparison_requires_lane_labels": [
            "native_stock_flowstar",
            "authoritative_torch_proactive_subdivision",
            "torch_natural_shadow",
        ],
    }


def _render_markdown(contract: dict[str, Any]) -> str:
    rows = []
    for name, field in contract["fields"].items():
        rows.append(f"| `{name}` | {'yes' if field['matched'] else 'no'} | {field['reason']} |")
    return "\n".join(
        [
            "# Matched VDP order-4 execution contract",
            "",
            "This file is generated from the stock Flow* benchmark/default sources, the Torch runner/config, and the frozen terminal checkpoint. `matched=no` is intentional and fail-closed; it is never filled by a look-alike field.",
            "",
            "| Field | Matched | Evidence-based reason |",
            "|---|---:|---|",
            *rows,
            "",
            "The physical ODE, initial x/y box, order, step bounds, half-on-reject/1.1-on-accept scheduler, candidate remainder radius, and cutoff match. The authoritative end-to-end executions do not form one fully matched numerical contract because Torch has no active symbolic-remainder queue, uses a different interval backend, has a different stored/local basis layout, and proactively subdivides the polynomial-truncation range.",
            "",
        ]
    )


def _render_field_map() -> str:
    return """# Cross-tool field map

| Meaning | Torch field | Flow* field | Mapping status |
|---|---|---|---|
| physical position | `x`, state 0, local generator `u0` | `x`, state 0, local generator `r0` after time | exact after basis permutation |
| physical velocity | `y`, state 1, local generator `u1` | `y`, state 1, local generator `r1` after time | exact after basis permutation |
| local time | `tau`, domain index 2 | domain index 0 | exact after index permutation |
| clock state | absent | `t`, state 2, normalized generator `r2` | no Torch equivalent; use `null` |
| endpoint | `endpoint_raw_tm` at `tau=h` | composed flowpipe at local-time supremum | semantic match; never replace with segment range |
| segment | `segment.tm` over `[0,h]` | `TaylorModelFlowpipe` over domain index 0 | semantic match |
| tube | aggregate x/y hull plus ordered segment CSV | ordered `tmv_flowpipes` plus plot projection | not identical; compare ordered segment boxes, not aggregate hull |
| candidate remainder | per-x/y `[-1e-4,1e-4]` | per-x/y/t default `[-1e-4,1e-4]` | physical components match; clock field is `null` on Torch |
| Picard subfields not exported by stock trace | named Torch ledger fields | unavailable | Flow* value must remain `null`; aggregate remainder is not a substitute |
| symbolic carry | inactive (`symbolic_queue_present=false`) | `Symbolic_Remainder{J,Phi_L,scalars}`, max 100 | material mismatch |

Common monomial comparison uses `[u0,u1,tau] -> [r0,r1,time]`. The stock clock generator `r2` must have exponent zero for a physical x/y term to be comparable. Any nonzero `r2` exponent is reported as a stock-only term rather than silently projected away.
"""


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = build(args.flowstar_root.resolve())
    (output_dir / "matched_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "matched_contract.md").write_text(_render_markdown(contract), encoding="utf-8")
    (output_dir / "field_map.md").write_text(_render_field_map(), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowstar-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
