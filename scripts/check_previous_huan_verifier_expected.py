#!/usr/bin/env python3
"""Run the previous package verifier and require its two corrected D2 errors."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


EXPECTED = [
    "D2 cpu evidence lacks route-tagged schema v2",
    "D2 cuda evidence lacks route-tagged schema v2",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    script = args.repo_root.resolve() / "scripts/verify_huan_repro_package.py"
    spec = importlib.util.spec_from_file_location("previous_huan_verifier", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import previous Huan verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    errors = module.verify(args.repo_root.resolve(), args.output_root.resolve())
    ok = errors == EXPECTED
    print(
        json.dumps(
            {
                "schema": "torch_tm_flowpipe.previous_huan_verifier_expected_result/1",
                "ok": ok,
                "errors": errors,
                "interpretation": (
                    "legacy package is intentionally rejected only for its two schema-v1 "
                    "D2 labels; availability is not accepted as CUDA invocation"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
