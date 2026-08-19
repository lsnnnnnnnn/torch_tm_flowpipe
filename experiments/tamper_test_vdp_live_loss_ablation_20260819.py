#!/usr/bin/env python3
"""Prove semantic tampering of the VDP live-loss ledger fails closed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Sequence

from verify_vdp_live_loss_ablation_20260819 import VerificationError, verify


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _expect_rejected(root: Path, mutate: Callable[[Path], None]) -> str:
    mutate(root)
    try:
        verify(root)
    except (VerificationError, KeyError, StopIteration, ValueError) as error:
        return str(error)
    raise AssertionError("tampered live-loss ledger passed verification")


def run(source: Path, output: Path | None = None) -> dict[str, Any]:
    source = source.resolve()
    cases: list[dict[str, Any]] = []

    def execute(name: str, mutate: Callable[[Path], None]) -> None:
        with tempfile.TemporaryDirectory(prefix=f"vdp-live-loss-{name}-") as temporary:
            copied = Path(temporary) / "gate"
            shutil.copytree(source, copied)
            message = _expect_rejected(copied, mutate)
            cases.append({"case": name, "rejected": True, "message": message})

    def delete_earlier(root: Path) -> None:
        path = root / "live_loss_ledger.json"
        value = _load(path)
        value["rows"].pop(4)
        _write(path, value)

    def reorder(root: Path) -> None:
        path = root / "live_loss_ledger.json"
        value = _load(path)
        value["rows"][10], value["rows"][11] = value["rows"][11], value["rows"][10]
        _write(path, value)

    def dead_as_live(root: Path) -> None:
        path = root / "live_loss_ledger.json"
        value = _load(path)
        dead = next(row for row in value["rows"] if row["decision_role"] == "ordinary_path_diagnostic")
        dead["final_subset_width_live"] = True
        dead["consumer_chain_to_final_subset"] = [dead["stage_id"], "torch.i4.c0.subset"]
        _write(path, value)

    def old_first_loss(root: Path) -> None:
        path = root / "summary.json"
        value = _load(path)
        value["first_live_strict_surplus"] = {
            **value["first_live_strict_surplus"],
            "stage_id": "raw.B1.x_squared",
            "output_stage": "raw.B1.x_squared",
        }
        _write(path, value)

    def reused_eps(root: Path) -> None:
        path = root / "live_loss_ledger.json"
        value = _load(path)
        payments = value["rounding_proof"]["payments"]
        payments[1]["payment_id"] = payments[0]["payment_id"]
        _write(path, value)

    execute("delete_earlier_stage", delete_earlier)
    execute("reorder_stages", reorder)
    execute("mark_dead_stage_live", dead_as_live)
    execute("restore_obsolete_raw_B1_x_squared_first_loss", old_first_loss)
    execute("reuse_validation_eps_payment", reused_eps)
    result = {
        "schema": "vdp_live_loss_tamper_tests_v1",
        "passed": all(row["rejected"] for row in cases),
        "cases": cases,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _write(output, result)
    print(json.dumps(result, sort_keys=True))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.root, args.output)
