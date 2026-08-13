#!/usr/bin/env python3
"""Exercise the shared lossless Flow*/Torch state schema and fail closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.lossless_state_queue_schema import (
    decode_binary64_exact,
    export_torch_initial_state,
    iter_canonical_dyadics,
    parse_file,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_bridge(binary: Path, *arguments: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "argv": [str(binary), *arguments],
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def mutate(base: str, mutation: str) -> str:
    lines = base.splitlines()
    if mutation == "missing_field":
        lines = [line for line in lines if not line.startswith("queue.max_size=")]
    elif mutation == "duplicate_field":
        lines.append(next(line for line in lines if line.startswith("step=")))
    elif mutation == "nan_coefficient":
        index = next(i for i, line in enumerate(lines) if ".coefficient=" in line)
        lines[index] = lines[index].split("=", 1)[0] + "=nan"
    elif mutation == "wrong_dimension":
        index = next(i for i, line in enumerate(lines) if line.startswith("state_dimension="))
        lines[index] = "state_dimension=99"
    elif mutation == "wrong_order":
        index = next(i for i, line in enumerate(lines) if line.startswith("settings.order="))
        lines[index] = "settings.order=5"
    elif mutation == "unknown_field":
        lines.append("unexpected.field=1")
    elif mutation == "inverted_interval":
        lo_index = next(i for i, line in enumerate(lines) if line.endswith(".lo=53:-1:10000000000000:-52"))
        key = lines[lo_index].split("=", 1)[0]
        lines[lo_index] = key + "=53:1:1fffffffffffff:100"
    else:
        raise ValueError(mutation)
    return "\n".join(lines) + "\n"


def audit(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    bridge = args.bridge_binary.resolve()
    template = args.flowstar_fixture.resolve()

    torch_state = output / "torch_initial.state"
    torch_summary = export_torch_initial_state(template, torch_state)
    torch_roundtrip = output / "torch_initial.roundtrip.state"
    roundtrip = run_bridge(bridge, "roundtrip", str(torch_state), str(torch_roundtrip))
    if roundtrip["exit_code"] != 0 or torch_state.read_bytes() != torch_roundtrip.read_bytes():
        raise RuntimeError("Torch -> schema -> Flow* -> schema roundtrip was not byte exact")

    fixture_root = template.parent
    state_files = sorted(fixture_root.glob("*.state"))
    exact_float_count = 0
    for path in [*state_files, torch_state]:
        records = parse_file(path)
        for key, encoded in iter_canonical_dyadics(records):
            decode_binary64_exact(encoded)
            exact_float_count += 1

    negative_results: dict[str, Any] = {}
    base = torch_state.read_text(encoding="utf-8")
    mutations = (
        "missing_field",
        "duplicate_field",
        "nan_coefficient",
        "wrong_dimension",
        "wrong_order",
        "unknown_field",
    )
    for name in mutations:
        path = output / f"negative_{name}.state"
        path.write_text(mutate(base, name), encoding="utf-8")
        result = run_bridge(bridge, "validate", str(path))
        negative_results[name] = result
        if result["exit_code"] == 0:
            raise RuntimeError(f"negative schema mutation was accepted: {name}")

    result = {
        "schema": "lossless_state_queue_cross_language_audit_v1",
        "torch_export": torch_summary,
        "torch_flowstar_roundtrip": roundtrip,
        "torch_flowstar_roundtrip_byte_exact": True,
        "flowstar_fixture_count": len(state_files),
        "binary64_exact_dyadic_count": exact_float_count,
        "all_precision53_values_exact_in_python_float": True,
        "negative_tests": negative_results,
        "negative_tests_all_rejected": True,
        "sampling_used_for_roundtrip": False,
        "common_box_reboxing": False,
        "status": "SAME_PRESTATE_LOSSLESS_BRIDGE_AVAILABLE",
    }
    write_json(output / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-binary", type=Path, required=True)
    parser.add_argument("--flowstar-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(audit(parse_args()), sort_keys=True, allow_nan=False))
