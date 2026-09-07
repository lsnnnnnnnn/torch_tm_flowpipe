"""The goal's exact endpoint witness at our two production substitution entries.

This is a local map check, not a new dynamical benchmark or a numerical repair.
The published dense lane converts its accepted model to TMVector and then uses
substitute_const/drop_variable/apply_cutoff (flowpipe.py). The dense validator
also calls BatchedTaylorModel.endpoint (batched_dense_tm.py).
"""
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

BASE = "f23c375f7935dc2f85a37a0ad69814ab052168bb"
PREVIOUS = "4939fb288c941a67f55cc191f4d75f8594692f47"
STOP = "OUR_REFERENCE_DEFECT_FOUND__PERFORMANCE_STOP"
REPO = Path(__file__).resolve().parents[2]
ENTRIES = ("published_endpoint", "dense_internal_endpoint")
SOURCE_FILES = (
    "src/torch_tm_flowpipe/batched_dense_tm.py",
    "src/torch_tm_flowpipe/flowpipe.py",
    "src/torch_tm_flowpipe/tm_vector.py",
    "src/torch_tm_flowpipe/taylor_model.py",
    "src/torch_tm_flowpipe/polynomial.py",
    "src/torch_tm_flowpipe/interval.py",
)


def endpoint_witness(entry):
    import torch
    from torch_tm_flowpipe import Interval, Polynomial, TaylorModel, TMVector
    from torch_tm_flowpipe.batched_dense_tm import (
        sparse_tmvector_to_dense, dense_to_sparse_tmvector,
    )

    if entry not in ENTRIES:
        raise ValueError(entry)
    time = float.fromhex("0x1.47ae147ae147bp-7")
    domain = [Interval(-1., 1.), Interval(-1., 1.), Interval(0., time)]
    model = TMVector([
        TaylorModel(Polynomial({(0, 0, 0): -1., (0, 0, 1): 100.}, 3),
                    Interval.zero(), domain, order=4)
        for _ in range(2)
    ])
    dense = sparse_tmvector_to_dense(model, order=4, device="cpu", dtype=torch.float64)
    if entry == "published_endpoint":
        segment = dense_to_sparse_tmvector(dense)
        endpoint = segment.substitute_const(2, time).drop_variable(2).apply_cutoff(1e-10)
    else:
        endpoint = dense_to_sparse_tmvector(dense.endpoint(2, time))
    exact = Fraction(-1) + Fraction(100) * Fraction(time)
    return model, endpoint, exact


def canonical(model, label):
    from torch_tm_flowpipe.brusselator_canonical_exchange import _append_tmv
    from experiments.xiangru_adoption.common import from_existing_canonical

    rows = []
    names = ("ux", "uy", "tau") if model.n_vars == 3 else ("ux", "uy")
    _append_tmv(rows, label, model, variable_order=names)
    return from_existing_canonical(dict(rows), label)


def collect():
    import torch
    import torch_tm_flowpipe as core
    from experiments.xiangru_adoption.common import measure

    git = lambda *args: subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    assert not git("status", "--porcelain"), "A clean committed driver is required"
    assert not git("diff", BASE, "--", "src"), "Stop branch cannot change solver code"
    assert Path(core.__file__).resolve().is_relative_to(REPO / "src")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = []
    for entry in ENTRIES:
        before, after, exact = endpoint_witness(entry)
        output = canonical(after, "endpoint")
        full = [[float(v.lo).hex(), float(v.hi).hex()] for v in after.range_box()]
        exact_representation = measure(output)
        rows.append({
            "entry": entry,
            "input_model": canonical(before, "segment"),
            "output_model": output,
            "full_range_hex": full,
            "represented_range_hex": [[x.hex() for x in pair] for pair in exact_representation],
            "contains": [Fraction(float.fromhex(lo)) <= exact <= Fraction(float.fromhex(hi))
                         for lo, hi in full],
        })
    return {
        "schema": "our_solver_endpoint_witness/1",
        "source_sha": git("rev-parse", "HEAD"),
        "source_clean": True,
        "numerical_base_sha": BASE,
        "source_root": str(REPO),
        "imported_package": core.__file__,
        "source_files": {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in SOURCE_FILES},
        "python": sys.executable, "python_version": sys.version,
        "torch_version": torch.__version__, "device": "cpu", "dtype": "float64", "batch": 1,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "time_hex": (0.01).hex(), "time_fraction": str(Fraction(0.01)),
        "exact_value": str(Fraction(-1) + 100 * Fraction(0.01)),
        "cutoff_hex": (1e-10).hex(), "order": 4,
        "rows": rows,
        "scope": "Actual local substitution methods; no ODE solve, history or normalization.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": STOP if any(not all(r["contains"]) for r in data["rows"]) else "CONTAINED",
                      "exact": data["exact_value"], "rows": data["rows"]}, indent=2))
    # Recording the evidence succeeded. The regression assertion is intentionally
    # left as a normal failing pytest test; a zero driver exit is not a safety pass.


if __name__ == "__main__":
    main()
