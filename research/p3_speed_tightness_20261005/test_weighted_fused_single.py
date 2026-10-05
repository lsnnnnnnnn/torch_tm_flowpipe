"""Real Torch CPU control tests with map stubs; no numerical/GPU qualification.

The oracle is the preserved weighted Python control flow, loaded without its
engine or a checker. Only _plan and _map are replaced by small deterministic
fixtures. All content digest operations are disabled before importing Torch.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from run_quad_candidate import guard_digests
guard_digests()
import torch

from weighted_fused_single import _two_rounds, _finish, install


SOURCE = Path(__file__).resolve().parents[1] / "gpu_verified_20260930/source/engine/src/flowstar_gpu"
PACKAGE = "fused_semantics_reference"
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


rounding, interval, reference = (load(name) for name in ("rounding", "interval", "weighted_validation"))


class FusedSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[[0.0]]], dtype=torch.float64)
        self.initial = self.x.clone()
        self.original = torch.tensor([[[-2.0, 2.0]]], dtype=torch.float64)
        self.current = torch.tensor([[[-1.0, 1.0]]], dtype=torch.float64)
        self.eligible = torch.tensor([True])
        self.plan = dict(zero_positions=torch.tensor([0]), initial_positions=torch.tensor([0]),
                         input_positions=torch.tensor([0]), sup=SimpleNamespace(size=1))
        self.engine = SimpleNamespace()

    def map_stub(self, images, bad=None):
        calls = []

        def evaluate(code, coefficients, point, candidate, plan, eng, cutoff):
            index = min(len(calls), len(images) - 1)
            calls.append(candidate.clone())
            return (torch.tensor([[images[index]]], dtype=torch.float64),
                    torch.tensor([False if bad is None else bad[index]]))
        return evaluate, calls

    def raw(self, images, bad=None):
        evaluate, calls = self.map_stub(images, bad)
        with patch.object(reference, "_plan", return_value=self.plan), \
                patch.object(reference, "_map", evaluate):
            result = reference.refine_accepted(
                None, self.x, None, self.initial, None, None, self.original,
                self.current, self.eligible, self.engine, 0.0, use_graph=False)
        return result, calls

    def fused(self, images, bad=None):
        evaluate, calls = self.map_stub(images, bad)
        with patch.object(reference, "_map", evaluate):
            value, flags = _two_rounds(
                torch, reference, rounding.next_down, None, self.x, self.initial,
                self.original, self.current, self.eligible, self.plan, self.engine, 0.0)
        return value, flags, calls

    def compare(self, images, *, fast, bad=None):
        inputs = (self.x, self.initial, self.original, self.current, self.eligible)
        versions = [value._version for value in inputs]
        snapshots = [value.clone() for value in inputs]
        expected, _ = self.raw(images, bad)
        value, flags, calls = self.fused(images, bad)
        fallback_calls = []

        def fallback():
            fallback_calls.append(True)
            return self.raw(images, bad)[0]

        actual, info, committed = _finish(value, flags, fallback)
        self.assertEqual(committed, fast)
        self.assertEqual(len(fallback_calls), int(not fast))
        self.assertEqual(len(calls), 2)
        self.assertTrue(torch.equal(actual.view(torch.uint8), expected[0].view(torch.uint8)))
        self.assertEqual(info, expected[1])
        for tensor, snapshot, version in zip(inputs, snapshots, versions):
            self.assertEqual(tensor._version, version)
            self.assertTrue(torch.equal(tensor.view(torch.uint8), snapshot.view(torch.uint8)))
        return actual, info, value

    def test_success_records_bytes_and_owned_outputs(self):
        actual, info, internal = self.compare([[-0.5, 0.6], [-0.25, 0.3]], fast=True)
        self.assertEqual(info["rounds"], [dict(eligible=1, attempted=1, recovered=1,
            evaluations=1, initial_mismatch=0, max_attempts=1)] * 2)
        internal.fill_(19)
        self.assertTrue(torch.equal(actual, torch.tensor([[[-0.25, 0.3]]], dtype=torch.float64)))

    def test_first_failure_keeps_current_and_one_round_record(self):
        actual, info, _ = self.compare([[-2.0, 2.0]], fast=False)
        self.assertTrue(torch.equal(actual, self.current))
        self.assertEqual(len(info["rounds"]), 1)
        self.assertEqual(info["rounds"][0]["recovered"], 0)

    def test_second_failure_keeps_first_bound(self):
        actual, info, _ = self.compare([[-0.5, 0.6], [-1.0, 1.0]], fast=False)
        self.assertTrue(torch.equal(actual, torch.tensor([[[-0.5, 0.6]]], dtype=torch.float64)))
        self.assertEqual([r["recovered"] for r in info["rounds"]], [1, 0])

    def test_preconditions_and_bad_maps_fall_back(self):
        self.eligible[0] = False
        self.compare([[-0.5, 0.6]], fast=False)
        self.eligible[0] = True
        self.initial[0, 0, 0] = 1
        self.compare([[-0.5, 0.6]], fast=False)
        self.initial[0, 0, 0] = 0
        self.x[0, 0, 0] = float("nan")
        self.compare([[-0.5, 0.6]], fast=False)
        self.x[0, 0, 0] = 0
        self.compare([[-0.5, 0.6]], bad=[True], fast=False)

    def test_first_disjoint_uses_original_exception(self):
        # Correct zero-containing intervals cannot be disjoint here. Inject a
        # corrupted intersection to exercise the production defensive branch.
        maximum = torch.maximum

        def corrupted_intersection():
            calls = []

            def operation(a, b):
                calls.append(True)
                return torch.full_like(a, 10.0) if len(calls) == 3 else maximum(a, b)
            return operation

        images = [[-0.5, 0.6], [-0.25, 0.3]]
        with patch.object(torch, "maximum", corrupted_intersection()):
            value, flags, _ = self.fused(images)
        self.assertFalse(bool(flags[2]))

        def fallback():
            with patch.object(torch, "maximum", corrupted_intersection()):
                return self.raw(images)[0]

        with self.assertRaisesRegex(FloatingPointError, "two validated same-polynomial bounds are disjoint"):
            _finish(value, flags, fallback)

    def test_install_cpu_trace_dispatch_and_restore(self):
        # Runtime identity is mocked only for adapter dispatch; tensor math
        # above uses this machine's real CPU Torch version.
        wv = ModuleType("flowstar_gpu.weighted_validation")
        wv.__file__ = reference.__file__
        for name in ("refine_accepted", "_evaluate_map", "_map", "_plan"):
            setattr(wv, name, getattr(reference, name))
        se = ModuleType("flowstar_gpu.sparse_exec")
        se.__file__ = __file__
        graphing = ModuleType("flowstar_gpu.graphing")
        graphing.GraphCache = lambda *a: self.fail("CPU/trace fallback must not capture")
        graphing.WARMUP = 2
        modules = {m.__name__: m for m in (wv, se, graphing)}
        modules["flowstar_gpu.rounding"] = rounding
        with patch.dict(sys.modules, modules), patch.object(torch, "__version__", "2.5.1+fixture"):
            binding = install(torch, wv, se)
            with self.assertRaises(RuntimeError):
                install(torch, wv, se)
            events = []
            evaluate, _ = self.map_stub([[-0.5, 0.6], [-0.25, 0.3]])
            with patch.object(reference, "_plan", return_value=self.plan), \
                    patch.object(reference, "_map", evaluate):
                value, info = wv.refine_accepted(None, self.x, None, self.initial, None,
                    None, self.original, self.current, self.eligible, self.engine, 0.0,
                    trace=events.append)
            self.assertEqual(len(events), 2)
            self.assertEqual([event["refinement_round"] for event in events], [0, 1])
            self.assertEqual(binding.counters["fallback_calls"], 1)
            self.assertEqual(binding.counters["graph_map_calls"], 0)
            installed = wv.refine_accepted
            wv.refine_accepted = lambda: None
            with self.assertRaises(RuntimeError):
                binding.restore()
            wv.refine_accepted = installed
            binding.restore()
            self.assertIs(wv.refine_accepted, reference.refine_accepted)
            with self.assertRaises(RuntimeError):
                binding.restore()


if __name__ == "__main__":
    print(f"CPU Torch {torch.__version__}; map stubs only; no GPU/checker/digest calls")
    unittest.main()
