#!/usr/bin/env python3
"""Count program-issued host scalar extraction through Torch dispatch."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torch_tm_flowpipe.batched_dense_tm import dense_transient_ledger_suppressed
from torch_tm_flowpipe.tora_q3 import (
    build_tora_q3_initial_model,
    compose_tora_q3_step,
    dense_tora_q3_dr_step,
    dense_validation_batch,
    identity_tora_q3_carry,
    normalize_tora_q3_boundary,
    project_tora_q3_endpoint_to_affine,
    tora_q3_boundary_from_model,
)


class ScalarDispatchAudit(TorchDispatchMode):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            source = "<non_repository>"
            for frame in reversed(traceback.extract_stack()):
                try:
                    relative = Path(frame.filename).resolve().relative_to(
                        self.root
                    )
                except ValueError:
                    continue
                if relative == Path("experiments/audit_tora_q3_dispatch_sync.py"):
                    continue
                if str(relative).startswith(("src/", "experiments/", "scripts/")):
                    source = f"{relative}:{frame.lineno}:{frame.name}"
                    break
            self.counts[source] += 1
        return func(*args, **(kwargs or {}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path.cwd().resolve()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    lower = torch.full((48,), 9.8, dtype=torch.float64, device=device)
    upper = torch.full((48,), 10.2, dtype=torch.float64, device=device)
    base = build_tora_q3_initial_model(lower, upper, device=device)
    boundary = tora_q3_boundary_from_model(base)
    local, carry = normalize_tora_q3_boundary(
        boundary, identity_tora_q3_carry(48, device=device)
    )
    audit = ScalarDispatchAudit(root)
    with torch.no_grad(), audit:
        with dense_validation_batch():
            with dense_transient_ledger_suppressed():
                local_step = dense_tora_q3_dr_step(
                    local,
                    capture_trace=False,
                    point_enclosure_backend="eager",
                )
            physical_step = compose_tora_q3_step(local_step, carry)
            projection = project_tora_q3_endpoint_to_affine(
                local_step.segment_tm
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if not local_step.accepted or not physical_step.accepted:
        raise RuntimeError("dispatch audit workload did not validate")
    payload = {
        "schema": "tora_q3_program_dispatch_sync_audit_v1",
        "status": "PASS",
        "scope": "B48 full logical one-step math",
        "method": (
            "TorchDispatchMode interception of aten._local_scalar_dense; "
            "profiler-internal events that do not pass dispatcher are excluded"
        ),
        "program_issued_host_scalar_sync_count": sum(audit.counts.values()),
        "source_call_sites": [
            {"source": source, "count": count}
            for source, count in audit.counts.most_common()
        ],
        "accepted": True,
        "projection_shape": list(projection.center.shape),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
