#!/usr/bin/env python3
"""Replay one canonical Brusselator TM through Torch and pinned Flow* ranges."""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch_tm_flowpipe.brusselator_canonical_exchange import (  # noqa: E402
    SCHEMA,
    object_sha256,
    read_records,
    take_tmv,
)
from torch_tm_flowpipe.step1_oracle import (  # noqa: E402
    RationalInterval,
    RationalPolynomial,
    fraction_text,
)


FLOWSTAR_SHA = "b85a3211748cb77b736fe4ad42ee02d8d2b81148"
HARNESS = ROOT / "experiments/flowstar_probe/flowstar_brusselator_canonical_range.cpp"
ACCESS_PATCH = ROOT / "experiments/flowstar_probe/flowstar_canonical_range_access.patch"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("same-object range matrix cannot be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_logged(command: Sequence[str], cwd: Path, log_dir: Path, name: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    stdout = log_dir / f"{name}.stdout.log"
    stderr = log_dir / f"{name}.stderr.log"
    stdout.write_text(result.stdout, encoding="utf-8")
    stderr.write_text(result.stderr, encoding="utf-8")
    return {
        "command": list(command),
        "cwd": str(cwd),
        "exit_code": result.returncode,
        "wall_seconds": elapsed,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
    }


def _build_harness(source: Path, output: Path) -> tuple[Path, dict[str, Any], tempfile.TemporaryDirectory[str]]:
    source = source.resolve()
    if subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True, check=True
    ).stdout.strip() != FLOWSTAR_SHA:
        raise ValueError("Flow* source is not at the pinned commit")
    compiler = shutil.which("g++-15")
    if compiler is None:
        raise FileNotFoundError("g++-15 is required by the frozen Flow* lane")
    temporary: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory(
        prefix="flowstar-brusselator-canonical-"
    )
    root = Path(temporary.name)
    clone = root / "repo"
    logs = output / "build_logs"
    logs.mkdir(parents=True, exist_ok=True)
    build: dict[str, Any] = {}
    build["clone"] = _run_logged(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(clone)],
        root,
        logs,
        "clone",
    )
    if build["clone"]["exit_code"] != 0:
        raise RuntimeError("disposable Flow* clone failed")
    build["checkout"] = _run_logged(
        ["git", "checkout", "--quiet", FLOWSTAR_SHA], clone, logs, "checkout"
    )
    build["patch"] = _run_logged(
        ["git", "apply", str(ACCESS_PATCH)], clone, logs, "access_patch"
    )
    if build["checkout"]["exit_code"] != 0 or build["patch"]["exit_code"] != 0:
        raise RuntimeError("pinned Flow* checkout/access patch failed")
    toolbox = clone / "flowstar-toolbox"
    build["toolbox"] = _run_logged(
        ["make", "-j1", f"CXX={compiler} -fpermissive"],
        toolbox,
        logs,
        "build_toolbox",
    )
    binary = root / "flowstar_brusselator_canonical_range"
    compile_command = [
        compiler,
        "-DFLOWSTAR_CANONICAL_RANGE_ACCESS",
        "-fpermissive",
        "-O3",
        "-std=c++11",
        "-I",
        str(toolbox),
        str(HARNESS),
        "-L",
        str(toolbox),
        "-L",
        "/usr/local/lib",
        "-o",
        str(binary),
        "-lflowstar",
        "-lmpfr",
        "-lgmp",
        "-lgsl",
        "-lgslcblas",
        "-lm",
        "-lglpk",
    ]
    build["harness"] = _run_logged(compile_command, root, logs, "build_harness")
    if build["toolbox"]["exit_code"] != 0 or build["harness"]["exit_code"] != 0:
        raise RuntimeError("Flow* canonical harness build failed")
    build.update(
        {
            "source_commit": FLOWSTAR_SHA,
            "access_patch_sha256": _sha256(ACCESS_PATCH),
            "harness_sha256": _sha256(HARNESS),
            "compiler": subprocess.run(
                [compiler, "--version"], text=True, capture_output=True, check=True
            ).stdout.splitlines()[0],
            "binary_sha256": _sha256(binary),
        }
    )
    return binary, build, temporary


def _flat_records(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise ValueError("Flow* composed object is not newline-terminated")
    for line in text.splitlines():
        if not line or line.count("=") != 1:
            raise ValueError("malformed Flow* composed object")
        key, value = line.split("=", 1)
        if key in result or not key or not value:
            raise ValueError("duplicate/empty Flow* composed field")
        result[key] = value
    return result


def _fraction(value: Any) -> Fraction:
    return Fraction.from_float(float(value.detach().cpu()) if hasattr(value, "detach") else float(value))


def _exact_range(model: Any, *, include_remainder: bool) -> RationalInterval:
    polynomial = RationalPolynomial(
        model.n_vars,
        {
            tuple(int(item) for item in exponent): _fraction(coefficient)
            for exponent, coefficient in model.polynomial.terms.items()
        },
    )
    domain = [RationalInterval(_fraction(item.lo), _fraction(item.hi)) for item in model.domain]
    result = polynomial.bernstein_range(domain)
    if include_remainder:
        result = result + RationalInterval(
            _fraction(model.remainder.lo), _fraction(model.remainder.hi)
        )
    return result


def _interval_record(interval: Any) -> dict[str, Any]:
    lo = float(interval.lo.detach().cpu())
    hi = float(interval.hi.detach().cpu())
    return {"lo": lo, "hi": hi, "lo_hex": lo.hex(), "hi_hex": hi.hex()}


def _contains_exact(interval: Mapping[str, Any], exact: RationalInterval) -> bool:
    lo = Fraction.from_float(float.fromhex(str(interval["lo_hex"])))
    hi = Fraction.from_float(float.fromhex(str(interval["hi_hex"])))
    return lo <= exact.lo and exact.hi <= hi


def _matrix_row(
    *,
    step: int,
    object_sha: str,
    operator: str,
    implementation: str,
    stage_class: str,
    channel: str,
    component: int,
    interval: Mapping[str, Any],
    exact: RationalInterval,
) -> dict[str, Any]:
    lo = float.fromhex(str(interval["lo_hex"]))
    hi = float.fromhex(str(interval["hi_hex"]))
    return {
        "accepted_step": step,
        "canonical_object_sha256": object_sha,
        "operator": operator,
        "implementation": implementation,
        "stage_class": stage_class,
        "channel": channel,
        "component": component,
        "lo": lo,
        "hi": hi,
        "lo_hex": str(interval["lo_hex"]),
        "hi_hex": str(interval["hi_hex"]),
        "width": hi - lo,
        "exact_bernstein_lo": fraction_text(exact.lo),
        "exact_bernstein_hi": fraction_text(exact.hi),
        "exact_local_outward_contained": _contains_exact(interval, exact),
    }


def _support_sha(value: Any) -> str:
    payload = [
        [
            [list(exponent), float(coefficient.detach().cpu()).hex()]
            for exponent, coefficient in sorted(model.polynomial.terms.items())
        ]
        for model in value
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _replay_one(
    canonical: Path,
    binary: Path,
    output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_records = read_records(canonical)
    if source_records["schema"] != SCHEMA:
        raise ValueError("wrong canonical exchange schema")
    step = int(source_records["accepted_step"])
    object_sha = object_sha256(canonical)
    endpoint = take_tmv(dict(source_records), "tm.segment_endpoint_raw")
    endpoint_pre_cutoff = take_tmv(
        dict(source_records), "tm.segment_endpoint_pre_cutoff"
    )
    tube = take_tmv(dict(source_records), "tm.segment_tube")
    torch_inserted = take_tmv(dict(source_records), "tm.boundary_torch_inserted")
    torch_endpoint_poly = [_interval_record(model.polynomial.evaluate_interval(model.domain)) for model in endpoint]
    torch_endpoint_full = [_interval_record(model.range_box()) for model in endpoint]
    torch_tube_poly = [_interval_record(model.polynomial.evaluate_interval(model.domain)) for model in tube]
    torch_tube_full = [_interval_record(model.range_box()) for model in tube]
    torch_composition_poly = [
        _interval_record(model.polynomial.evaluate_interval(model.domain)) for model in torch_inserted
    ]
    torch_composition_full = [_interval_record(model.range_box()) for model in torch_inserted]

    flow_output = output / "flowstar_composed" / f"accepted_step_{step:04d}.canonical"
    flow_output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [str(binary), str(canonical), str(flow_output)],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Flow* canonical replay failed at step {step}: {process.stderr.strip()}")
    flow_result = json.loads(process.stdout)
    output_records = _flat_records(flow_output)
    if output_records.pop("schema", None) != "flowstar.brusselator_canonical_composition/1":
        raise ValueError("Flow* composed object schema mismatch")
    if int(output_records.pop("accepted_step")) != step:
        raise ValueError("Flow* composed object step mismatch")
    if output_records.pop("source.flowstar_commit") != FLOWSTAR_SHA:
        raise ValueError("Flow* composed object source mismatch")
    output_records.pop("source.input_checkpoint_sha256")
    output_records.pop("boundary.composition_branch")
    flow_inserted = take_tmv(output_records, "tm.flowstar_inserted")

    exact_endpoint_poly = [_exact_range(model, include_remainder=False) for model in endpoint]
    exact_endpoint_full = [_exact_range(model, include_remainder=True) for model in endpoint]
    exact_endpoint_pre_cutoff_full = [
        _exact_range(model, include_remainder=True) for model in endpoint_pre_cutoff
    ]
    exact_tube_poly = [_exact_range(model, include_remainder=False) for model in tube]
    exact_tube_full = [_exact_range(model, include_remainder=True) for model in tube]
    exact_torch_composition = [
        _exact_range(model, include_remainder=True) for model in torch_inserted
    ]
    exact_flow_composition = [
        _exact_range(model, include_remainder=True) for model in flow_inserted
    ]
    rows: list[dict[str, Any]] = []
    specifications = (
        ("A", "Torch", "reporting_endpoint", "endpoint", torch_endpoint_poly, exact_endpoint_poly),
        ("B", "Flow*", "reporting_endpoint", "endpoint", flow_result["endpoint_polynomial"], exact_endpoint_poly),
        ("C", "Torch", "reporting_tube", "tube", torch_tube_poly, exact_tube_poly),
        ("D", "Flow*", "reporting_tube", "tube", flow_result["tube_polynomial"], exact_tube_poly),
        ("E", "Torch", "reporting_endpoint", "endpoint", torch_endpoint_full, exact_endpoint_full),
        ("E", "Torch", "reporting_tube", "tube", torch_tube_full, exact_tube_full),
        ("F", "Flow*", "reporting_endpoint", "endpoint", flow_result["endpoint_full"], exact_endpoint_full),
        ("F", "Flow*", "reporting_tube", "tube", flow_result["tube_full"], exact_tube_full),
        ("G", "Torch", "boundary_normalization", "boundary", torch_composition_full, exact_torch_composition),
        ("H", "Flow*", "boundary_normalization", "boundary", flow_result["composition_full"], exact_flow_composition),
        (
            "X1",
            "Torch",
            "cutoff_payment",
            "endpoint",
            torch_endpoint_full,
            exact_endpoint_pre_cutoff_full,
        ),
        (
            "X2",
            "Flow*",
            "cutoff_payment",
            "endpoint",
            flow_result["endpoint_pre_cutoff_full"],
            exact_endpoint_pre_cutoff_full,
        ),
        (
            "X3",
            "Flow*_diagnostic_cutoff_normal",
            "cutoff_payment",
            "endpoint",
            flow_result["endpoint_flowstar_cutoff_full"],
            exact_endpoint_pre_cutoff_full,
        ),
    )
    for operator, implementation, stage_class, channel, intervals, exacts in specifications:
        for component, (interval, exact) in enumerate(zip(intervals, exacts, strict=True)):
            rows.append(
                _matrix_row(
                    step=step,
                    object_sha=object_sha,
                    operator=operator,
                    implementation=implementation,
                    stage_class=stage_class,
                    channel=channel,
                    component=component,
                    interval=interval,
                    exact=exact,
                )
            )
    result = {
        "accepted_step": step,
        "canonical_object": canonical.name,
        "canonical_object_sha256": object_sha,
        "flowstar_stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
        "flowstar_composed_filename": flow_output.relative_to(output).as_posix(),
        "flowstar_composed_sha256": _sha256(flow_output),
        "torch_inserted_support_sha256": _support_sha(torch_inserted),
        "flowstar_inserted_support_sha256": _support_sha(flow_inserted),
        "all_exact_local_outward_checks": all(row["exact_local_outward_contained"] for row in rows),
    }
    return rows, result


def run(args: argparse.Namespace) -> dict[str, Any]:
    objects = args.objects_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((objects / "index.json").read_text(encoding="utf-8"))
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.binary is None:
        binary, build, temporary = _build_harness(args.flowstar_source, output)
    else:
        binary = args.binary.resolve()
        if not binary.is_file():
            raise FileNotFoundError(binary)
        build = {
            "prebuilt_binary": str(binary),
            "binary_sha256": _sha256(binary),
            "source_commit": FLOWSTAR_SHA,
        }
    try:
        rows: list[dict[str, Any]] = []
        replay_records: list[dict[str, Any]] = []
        for item in index["objects"]:
            canonical = objects / item["filename"]
            if _sha256(canonical) != item["sha256"]:
                raise ValueError(f"canonical object hash mismatch: {canonical}")
            item_rows, item_result = _replay_one(canonical, binary, output)
            rows.extend(item_rows)
            replay_records.append(item_result)
        matrix = output / "same_object_range_matrix.csv"
        _write_csv(matrix, rows)
        result = {
            "schema": "torch_tm_flowpipe.brusselator_same_object_range_replay/1",
            "status": "SAME_OBJECT_RANGE_REPLAY_COMPLETE",
            "exchange_index_sha256": _sha256(objects / "index.json"),
            "flowstar_commit": FLOWSTAR_SHA,
            "build": build,
            "objects": replay_records,
            "matrix_rows": len(rows),
            "matrix_sha256": _sha256(matrix),
            "all_exact_local_outward_checks": all(
                row["exact_local_outward_contained"] for row in rows
            ),
        }
        _write_json(output / "range_replay.json", result)
        return result
    finally:
        if temporary is not None:
            temporary.cleanup()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", required=True, type=Path)
    parser.add_argument("--flowstar-source", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.binary is None and args.flowstar_source is None:
        parser.error("provide --flowstar-source or --binary")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["all_exact_local_outward_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
