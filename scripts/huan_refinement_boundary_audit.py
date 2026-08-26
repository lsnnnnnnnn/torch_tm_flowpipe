#!/usr/bin/env python3
"""Independent control-flow probes for Huan's shipped refinement loop."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
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

    accepted = torch.tensor([-1.0, 1.0], dtype=torch.float64, device=device).expand(1, 3, 2).contiguous()
    ok = torch.ones(1, dtype=torch.bool, device=device)
    bad = torch.zeros(1, dtype=torch.bool, device=device)
    cache = torch.empty(1, 0, 2, dtype=torch.float64, device=device)
    tails = torch.empty(1, 0, 2, dtype=torch.float64, device=device)
    int_diff = torch.zeros_like(accepted)
    step = UnitStep(torch, device)

    original: Callable[..., Any] = fp.exec_replay

    def invoke(max_steps: int, replay: Callable[..., Any], initial_ok: Any = ok):
        calls = 0

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return replay(*args, **kwargs)

        fp.exec_replay = counted
        try:
            settings = SimpleNamespace(max_refinement_steps=max_steps, stop_ratio=0.99)
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
    cur_partial, bad_partial, calls_partial = invoke(1, partial)
    partial_semantics = {
        "calls": calls_partial,
        "dim0_committed": bool(not torch.equal(cur_partial[:, 0], accepted[:, 0])),
        "failing_dim1_unchanged": bool(torch.equal(cur_partial[:, 1], accepted[:, 1])),
        "later_dim2_unchanged": bool(torch.equal(cur_partial[:, 2], accepted[:, 2])),
        "classification": "FLOWSTAR_SEQUENTIAL_FIRST_FAIL_PARTIAL_VECTOR_COMMIT",
    }

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
            all(value for key, value in partial_semantics.items() if key.endswith(("committed", "unchanged"))),
        )
    )
    return {
        "schema": "torch_tm_flowpipe.huan_refinement_boundary_audit/1",
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
        "api_contract": {
            "signature": str(signature),
            "proposal_commit_ledger_exposed": "ledger" in source.lower() or "ledger" in signature.parameters,
            "remainder_cache_freshness_metadata_exposed": any(token in source.lower() for token in ("generation", "freshness", "cache_version")),
            "returns_only_remainder_and_bad": "tuple[torch.Tensor, torch.Tensor]" in source,
        },
        "behavioral_passed": behavioral_passed,
        "contract_gate_passed": False,
        "contract_gap": "behavioral replay controls pass, but no public proposal/commit ledger or remainder-cache freshness metadata can establish the required final-ledger and stale-cache claims",
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
    return 0 if payload["behavioral_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
