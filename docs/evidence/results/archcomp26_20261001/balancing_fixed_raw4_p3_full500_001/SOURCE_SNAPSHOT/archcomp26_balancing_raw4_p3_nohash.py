#!/usr/bin/env python3
"""Working-P3 run for the fixed-repository Balancing profile."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import archcomp26_balancing_raw4_huan_nohash as balancing
import archcomp26_dp_p3_nohash as helper
import archcomp26_tora_remain_p3_nohash as p3_runtime


def prepare(backend):
    if backend != "p3":
        raise ValueError("this isolated entry only runs P3")
    torch, driver, _cap = p3_runtime.prepare()
    return torch, driver, helper.ENGINE, helper.CACHE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke1", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    balancing.common.prepare = prepare
    run_args = argparse.Namespace(backend="p3", mode=args.mode, output=args.output)
    try:
        return balancing.run(run_args)
    finally:
        if args.output.is_dir():
            (args.output / "P3_METHOD.json").write_text(json.dumps({
                "schema": "archcomp26-balancing-fixed-raw4-p3-method-nohash-v1",
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "profile": "balancing-fixed-repo-raw4",
                "method": "working P3 order 3 / validation order 4, strict endpoint and control injection, box same-slope CROWN, rpc-float32",
                "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "engine": str(helper.ENGINE), "shared_driver": str(helper.DRIVER),
                "source_files": [str(Path(__file__).resolve()),
                                 str(Path(balancing.__file__).resolve()),
                                 str(Path(p3_runtime.__file__).resolve()),
                                 str(Path(helper.__file__).resolve())],
                "refinement_trace": "Settings.refinement_callback selects the eager refinement path; no timing comparison",
                "qualification": ("500-period full-initial-box attempt with [8,10] s saved-box checks; no end-to-end floating-point NNCS proof"
                                  if args.mode == "full" else
                                  "single full-initial-box control period only; no property-window check or end-to-end floating-point NNCS proof"),
            }, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
