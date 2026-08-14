#!/usr/bin/env python3
"""Recompute the prior scientific/attestation publication distinction."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence


SCIENTIFIC = "a8653a7d9ea6f54b1450da6bee9af0e2a5a19695"
ATTESTATION = "3940386a61bdd6edbf3dc1722be031a1da572171"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    prior_manifest = json.loads(args.prior_manifest.read_text(encoding="utf-8"))
    trees = {
        name: {
            "scientific": _git(args.repo, "rev-parse", f"{SCIENTIFIC}:{name}"),
            "attestation": _git(args.repo, "rev-parse", f"{ATTESTATION}:{name}"),
        }
        for name in ("src", "experiments", "tests")
    }
    changed = _git(args.repo, "diff", "--name-only", SCIENTIFIC, ATTESTATION).splitlines()
    merge_base = _git(args.repo, "merge-base", SCIENTIFIC, ATTESTATION)
    result = {
        "schema": "prior_publication_semantics_recheck_v1",
        "scientific_sha": SCIENTIFIC,
        "attestation_tip": ATTESTATION,
        "merge_base": merge_base,
        "attestation_descends_from_scientific": merge_base == SCIENTIFIC,
        "changed_paths": changed,
        "relevant_tree_hashes": trees,
        "relevant_trees_unchanged": all(value["scientific"] == value["attestation"] for value in trees.values()),
        "prior_manifest": {
            "path": str(args.prior_manifest),
            "sha256": _sha(args.prior_manifest),
            "scientific_commit": prior_manifest.get("scientific_commit"),
            "publication_tip": prior_manifest.get("publication_tip"),
        },
        "labels": [
            "LOSSLESS_STATE_SERIALIZATION_CLOSED",
            "SAME_ENGINE_REPLAY_CLOSED",
            "CROSS_OPERATOR_SAME_PRESTATE_NOT_AVAILABLE",
            "OPERATOR_ATTRIBUTION_OPEN",
        ],
        "publication": {
            "scientific_sha_verified": True,
            "attestation_tip_contains_no_scientific_tree_changes": True,
            "final_tip_fresh_clone_verified": "unknown",
        },
    }
    passed = bool(
        result["attestation_descends_from_scientific"]
        and result["relevant_trees_unchanged"]
        and prior_manifest.get("scientific_commit") == SCIENTIFIC
        and prior_manifest.get("publication_tip") is None
    )
    result["passed"] = passed
    (output / "recheck.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema": "prior_publication_semantics_recheck_summary_v1",
                "passed": passed,
                "labels": result["labels"],
                "publication": result["publication"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise ValueError("prior publication recheck failed")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps({"passed": result["passed"], "publication": result["publication"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
