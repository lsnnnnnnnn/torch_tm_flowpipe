"""One CPU control/ownership check; no solver, GPU, old checker, or digest.

Frozen refine_accepted is the oracle, with only its plan and interval-map calls
replaced by tiny deterministic fixtures. This is not a CUDA qualification.
"""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2]
sys.path.insert(0, str(RESEARCH / "p3_speed_tightness_20261005"))
from run_quad_candidate import guard_digests
guard_digests()
import torch
from weighted_fused_batch import _run_chunks, _two_rounds, _finish, install

SOURCE = RESEARCH / "gpu_verified_20260930/source/engine/src/flowstar_gpu"
PACKAGE = "fused_batch_control_reference"
package = ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package
for name in ("elementary", "support", "cuda_kernels"):
    sys.modules[PACKAGE + "." + name] = ModuleType(PACKAGE + "." + name)


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, SOURCE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rounding, interval, reference = [load(name) for name in ("rounding", "interval", "weighted_validation")]


def byte_equal(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def check_case(batch, rows, case):
    # Lane-dependent radii expose accidentally reused graph-static output buffers.
    x = torch.arange(batch, dtype=torch.float64).reshape(batch, 1, 1)
    initial = x.clone()
    radius = 1 + x[:, :, 0] / (2 * batch)
    current = torch.stack((-radius, radius), -1)
    original = current * 2
    eligible = torch.ones(batch, dtype=torch.bool)
    target = rows + 1 if batch > rows else batch - 1
    if case == "ineligible":
        eligible[target] = False
    elif case == "initial_mismatch":
        initial[target] += 1
    elif case == "nonfinite":
        x[target] = float("nan")
    inputs = (x, initial, original, current, eligible)
    snapshots, versions = [v.clone() for v in inputs], [v._version for v in inputs]
    plan = dict(zero_positions=torch.tensor([0]), initial_positions=torch.tensor([0]),
                input_positions=torch.tensor([0]), sup=SimpleNamespace(size=1))
    engine = SimpleNamespace()

    def map_fixture(code, coefficients, point, candidate, plan, eng, cutoff):
        ids = point[:, 0, 0]
        threshold = 0.75 * (1 + ids / (2 * batch))
        first = candidate[:, 0, 1] > threshold
        selected = ids == target
        factor = torch.full_like(ids, 0.5)
        if case == "first_failure":
            factor = torch.where(selected & first, 2.0, factor)
        elif case == "second_failure":
            factor = torch.where(selected & ~first, 2.0, factor)
        bad = selected if case == "bad_map" else torch.zeros_like(selected)
        return candidate * factor[:, None, None], bad

    def original_call():
        return reference.refine_accepted(None, x, None, initial, None, None, original,
                                         current, eligible, engine, 0.0,
                                         chunk_size=rows, use_graph=False)

    # This replay deliberately overwrites the SAME result+flag buffers every
    # call, as GraphCache does. The last chunk must not overwrite earlier rows.
    statics, replay_count = [], 0
    def replay(values):
        nonlocal replay_count
        value, flags = _two_rounds(torch, reference, rounding.next_down, None,
                                  *values, plan, engine, 0.0)
        if not statics:
            statics.extend([torch.empty_like(value), torch.empty_like(flags)])
        statics[0].copy_(value)
        statics[1].copy_(flags)
        replay_count += 1
        return tuple(statics)

    with patch.object(reference, "_plan", return_value=plan), patch.object(reference, "_map", map_fixture):
        expected, expected_info = original_call()
        candidate, flags = _run_chunks(torch, inputs, rows, replay)
        # Verify the packed flags also own storage after a later reuse.
        statics[1].fill_(True)
        value, info, fast = _finish(candidate, flags, original_call)
    assert fast == (case == "success"), (batch, rows, case, flags.tolist())
    assert byte_equal(value, expected), (batch, rows, case, "output bytes")
    assert info == expected_info, (batch, rows, case, "statistics", info, expected_info)
    for tensor, snapshot, version in zip(inputs, snapshots, versions):
        assert tensor._version == version and byte_equal(tensor, snapshot), "input was modified"
    statics[0].fill_(19)
    assert byte_equal(value, expected), "public output aliases reusable graph storage"
    assert replay_count == (batch + rows - 1) // rows
    return dict(batch=batch, rows=rows, case=case, committed=fast,
                byte_equal=True, statistics_equal=True, input_bytes_unchanged=True,
                public_outputs_owned=True, replays=replay_count)


def check_install_restore():
    graphing = ModuleType("flowstar_gpu.graphing")
    graphing.GraphCache = lambda *a: (_ for _ in ()).throw(AssertionError("unexpected capture"))
    graphing.WARMUP = 2
    se = ModuleType("flowstar_gpu.sparse_exec")
    se.__file__ = __file__
    modules = {"flowstar_gpu.weighted_validation": reference, se.__name__: se,
               graphing.__name__: graphing, "flowstar_gpu.rounding": rounding}
    raw_refine, raw_evaluate = reference.refine_accepted, reference._evaluate_map
    with patch.dict(sys.modules, modules), patch.object(reference, "__name__", "flowstar_gpu.weighted_validation"), \
            patch.object(torch, "__version__", "2.5.1+CPU-fixture"):
        binding = install(torch, reference, se, batch=25, rows=32)
        try:
            install(torch, reference, se, batch=25, rows=32)
        except RuntimeError:
            pass
        else:
            raise AssertionError("duplicate install accepted")
        installed = reference.refine_accepted
        reference.refine_accepted = lambda: None
        try:
            binding.restore()
        except RuntimeError:
            pass
        else:
            raise AssertionError("foreign owner overwritten")
        reference.refine_accepted = installed
        binding.restore()
        assert reference.refine_accepted is raw_refine and reference._evaluate_map is raw_evaluate
    return dict(duplicate_install_refused=True, foreign_restore_refused=True,
                originals_restored=True, graph_calls=0)


def check_metadata_plan_admission():
    graphing = ModuleType("flowstar_gpu.graphing")
    graphing.GraphCache = lambda *a: (_ for _ in ()).throw(AssertionError("unexpected capture"))
    graphing.WARMUP = 2
    se = ModuleType("flowstar_gpu.sparse_exec")
    se.__file__ = __file__
    backend = ModuleType("fullbatch_p3_metadata_backend")
    backend.__file__ = str(RESEARCH.parent / "tools/archcomp26_quad_p3_nohash/metadata_backend.py")
    backend.wv, backend.se, backend._installed = reference, se, True
    original = reference._plan
    backend._original_weighted = original
    # Admission fixture with the same module/function ownership boundary.
    # No metadata factory, engine, original checker, or numerical plan runs.
    exec(compile("def weighted_plan(*args): return _original_weighted(*args)\n",
                 backend.__file__, "exec"), backend.__dict__)
    modules = {"flowstar_gpu.weighted_validation": reference, se.__name__: se,
               graphing.__name__: graphing, "flowstar_gpu.rounding": rounding,
               backend.__name__: backend}
    with patch.dict(sys.modules, modules), patch.object(reference, "__name__", "flowstar_gpu.weighted_validation"), \
            patch.object(torch, "__version__", "2.5.1+CPU-fixture"), \
            patch.object(reference, "_plan", backend.weighted_plan):
        binding = install(torch, reference, se, batch=1024, rows=256)
        assert reference._plan is backend.weighted_plan
        assert binding.receipt["plan_admission"]["installed_function_retained"]
        binding.restore()
        assert reference._plan is backend.weighted_plan
        for attribute, bad in (("_installed", False), ("wv", object()), ("se", object()),
                               ("_original_weighted", lambda: None)):
            with patch.object(backend, attribute, bad):
                try:
                    install(torch, reference, se, batch=1024, rows=256)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("invalid metadata owner accepted: " + attribute)
    assert reference._plan is original
    return dict(installed_metadata_plan_preserved=True, original_map_preserved=True,
                wrong_owner_or_uninstalled_or_foreign_original_rejected=4,
                graph_calls=0, numerical_plan_calls=0)


if __name__ == "__main__":
    cases = [check_case(batch, rows, case) for batch, rows in ((25, 32), (1024, 256))
             for case in ("success", "first_failure", "second_failure", "ineligible",
                          "initial_mismatch", "nonfinite", "bad_map")]
    print(json.dumps(dict(status="PASSED", scope="CPU control with deterministic map fixtures only",
                         torch_version=torch.__version__, cases=cases,
                         binding=check_install_restore(), metadata_plan=check_metadata_plan_admission(), gpu_calls=0,
                         old_checker_runs=0, digest_operations=0), indent=2))
