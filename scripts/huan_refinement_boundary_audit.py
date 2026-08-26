#!/usr/bin/env python3
"""Independent control-flow probes for Huan's shipped refinement loop."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any, Callable


class UnitStep:
    def __init__(self, torch: Any, device: str):
        self._torch = torch
        self._device = device

    def dt(self, _ndim: int) -> Any:
        return self._torch.tensor([1.0, 1.0], dtype=self._torch.float64, device=self._device)


def run(engine_root: Path, device: str) -> dict[str, Any]:
    sys.path.insert(0, str(engine_root / "src"))
    torch = importlib.import_module("torch")
    fp = importlib.import_module("flowstar_gpu.flowpipe")
    config = importlib.import_module("flowstar_gpu.config")
    determinism = importlib.import_module("flowstar_gpu.determinism")
    determinism.assert_gradual_underflow(device)

    accepted = torch.tensor([-1.0, 1.0], dtype=torch.float64, device=device).expand(1, 3, 2).contiguous()
    ok = torch.ones(1, dtype=torch.bool, device=device)
    bad = torch.zeros(1, dtype=torch.bool, device=device)
    cache = torch.empty(1, 0, 2, dtype=torch.float64, device=device)
    tails = torch.empty(1, 0, 2, dtype=torch.float64, device=device)
    int_diff = torch.zeros_like(accepted)
    step = UnitStep(torch, device)

    original: Callable[..., Any] = fp.exec_replay

    def invoke(
        max_steps: int,
        replay: Callable[..., Any],
        initial_ok: Any = ok,
        trace: list[dict[str, object]] | None = None,
    ):
        calls = 0

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return replay(*args, **kwargs)

        fp.exec_replay = counted
        try:
            settings = config.Settings(
                step=1.0,
                order=2,
                device=device,
                max_refinement_steps=max_steps,
                stop_ratio=0.99,
                refinement_callback=trace.append if trace is not None else None,
            )
            cur, bad_out = fp.refine_loop(
                None, accepted.clone(), initial_ok.clone(), bad.clone(), cache,
                tails, False, int_diff, step, settings,
            )
        finally:
            fp.exec_replay = original
        return cur, bad_out, calls

    shrink = lambda _code, cur, *_args, **_kwargs: cur * 0.5
    cur490, bad490, calls490 = invoke(490, shrink)
    cur491, bad491, calls491 = invoke(491, shrink)
    cur_fail, bad_fail, calls_fail = invoke(491, shrink, torch.zeros_like(ok))

    proposal = torch.tensor(
        [[[-0.5, 0.5], [-2.0, 2.0], [-0.25, 0.25]]],
        dtype=torch.float64,
        device=device,
    )
    partial = lambda *_args, **_kwargs: proposal
    partial_trace: list[dict[str, object]] = []
    cur_partial, bad_partial, calls_partial = invoke(1, partial, trace=partial_trace)
    component_rows = [
        row for row in partial_trace if row["event"] == "refinement_component"
    ]
    final_rows = [
        row for row in partial_trace if row["event"] == "final_remainder_owner"
    ]
    partial_semantics = {
        "calls": calls_partial,
        "dim0_committed": bool(not torch.equal(cur_partial[:, 0], accepted[:, 0])),
        "failing_dim1_unchanged": bool(torch.equal(cur_partial[:, 1], accepted[:, 1])),
        "later_dim2_unchanged": bool(torch.equal(cur_partial[:, 2], accepted[:, 2])),
        "classification": "FLOWSTAR_SEQUENTIAL_FIRST_FAIL_PARTIAL_VECTOR_COMMIT",
        "component_ledger": component_rows,
        "final_owner": final_rows,
    }

    stale_generation_rejected = False
    stale_owner_rejected = False
    settings_tamper = config.Settings(step=1.0, order=2, device=device)
    for state, token in (
        (
            fp.RefinementCacheState(
                "tampered-cache", "tails", 7, accepted.clone(), accepted.clone()
            ),
            "generation",
        ),
        (
            fp.RefinementCacheState(
                "tampered-cache", "tails", 0, accepted + 1.0, accepted.clone()
            ),
            "owner",
        ),
    ):
        try:
            fp.refine_loop(
                None, accepted.clone(), ok.clone(), bad.clone(), cache, tails,
                False, int_diff, step, settings_tamper, cache_state=state,
            )
        except RuntimeError as exc:
            if token == "generation":
                stale_generation_rejected = "stale refinement cache generation" in str(exc)
            else:
                stale_owner_rejected = "stale refinement cache owner" in str(exc)

    reach_trace: list[dict[str, object]] = []
    reach_settings = config.Settings(
        step=0.03125,
        order=3,
        device=device,
        max_refinement_steps=1,
        refinement_callback=reach_trace.append,
    )
    reach_result = fp.reach(
        ["1"], ["x"],
        torch.tensor([[[-0.5, 0.5]]], dtype=torch.float64, device=device),
        0.03125,
        reach_settings,
    )
    initial_rows = [row for row in reach_trace if row["event"] == "initial_self_map"]
    reach_final_rows = [
        row for row in reach_trace if row["event"] == "final_remainder_owner"
    ]

    signature = inspect.signature(fp.refine_loop)
    source = inspect.getsource(fp.refine_loop)
    defaults = config.Settings(step=0.01, order=3)
    behavioral_passed = all(
        (
            defaults.max_refinement_steps == 490,
            defaults.stop_ratio == 0.99,
            calls490 == 490,
            calls491 == 491,
            bool((torch.abs(cur491) <= torch.abs(cur490)).all()),
            not bool(bad490.any()),
            not bool(bad491.any()),
            calls_fail == 0,
            torch.equal(cur_fail, accepted),
            not bool(bad_fail.any()),
            not bool(bad_partial.any()),
            all(
                value for key, value in partial_semantics.items()
                if key.endswith(("committed", "unchanged"))
            ),
            len(component_rows) == 3,
            [row["commit_result"] for row in component_rows] == [True, False, False],
            len(final_rows) == 1,
            final_rows[0]["final_accepted_remainder"] == cur_partial[0].tolist(),
            stale_generation_rejected,
            stale_owner_rejected,
            bool(initial_rows),
            bool(reach_final_rows),
            int(reach_result.steps_completed[0]) == 1,
        )
    )
    return {
        "schema": "torch_tm_flowpipe.huan_refinement_boundary_audit/2",
        "engine_root": str(engine_root.resolve()),
        "device": device,
        "defaults": {"max_refinement_steps": defaults.max_refinement_steps, "stop_ratio": defaults.stop_ratio},
        "cap_boundary": {
            "calls_490": calls490,
            "calls_491": calls491,
            "iteration_491_subset_of_490": bool((torch.abs(cur491) <= torch.abs(cur490)).all()),
        },
        "initial_self_map_failure": {"replay_calls": calls_fail, "unchanged": bool(torch.equal(cur_fail, accepted))},
        "partial_vector_semantics": partial_semantics,
        "cache_freshness": {
            "immutable_cache_contract": "cache/tails hold fixed polynomial ranges and truncation tails; every replay receives the current candidate remainder explicitly",
            "stale_generation_rejected": stale_generation_rejected,
            "stale_owner_rejected": stale_owner_rejected,
        },
        "production_trace": {
            "initial_self_map_rows": initial_rows,
            "final_owner_rows": reach_final_rows,
            "steps_completed": int(reach_result.steps_completed[0]),
        },
        "api_contract": {
            "signature": str(signature),
            "proposal_commit_ledger_exposed": "refinement_callback" in source.lower(),
            "remainder_cache_freshness_metadata_exposed": any(token in source.lower() for token in ("generation", "freshness", "cache_version")),
            "returns_only_remainder_and_bad": "tuple[torch.Tensor, torch.Tensor]" in source,
        },
        "behavioral_passed": behavioral_passed,
        "contract_gate_passed": behavioral_passed,
        "contract_gap": None if behavioral_passed else "refinement ledger/cache contract check failed",
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
    print(json.dumps({"behavioral_passed": payload["behavioral_passed"], "contract_gate_passed": payload["contract_gate_passed"]}, sort_keys=True))
    return 0 if payload["behavioral_passed"] and payload["contract_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
