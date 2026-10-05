"""CPU fake-module checks of wrapper dispatch/ownership, not numerical parity."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from weighted_small_batch import install, pad_rows


def prohibited(*args, **kwargs):
    raise AssertionError("digests and numerical engines are forbidden in wrapper tests")


class Tensor:
    dtype, device, is_cuda = "float64", "cuda:0", True

    def __init__(self, data):
        self.data = data
        self.shape = (len(data), 1)

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def expand(self, rows, *shape):
        return Tensor(self.data * rows)

    def clone(self):
        return Tensor([row.copy() for row in self.data])


def _evaluate_map(*args):
    return "original_evaluate"


def refine_accepted(code, x, sup, initial, initial_sup, var_sups, original, current,
                    eligible, eng, cutoff, *, rounds=2, chunk_size=512, use_graph=True, trace=None):
    if trace is not None:
        trace.update(rounds=rounds, chunk_size=chunk_size, use_graph=use_graph)
        if trace.get("raise"):
            raise ValueError("fixture error")
    if not use_graph or not x.is_cuda:
        return "original_refine"
    module = sys.modules["flowstar_gpu.weighted_validation"]
    plan = object()
    for _ in range(rounds):
        result = module._evaluate_map(code, x, x, current, plan, eng, cutoff, use_graph)
    return result


class Cache:
    def __init__(self, device):
        self._segs = {}

    def run(self, key, fn, values, keepalive):
        self._segs[key] = True
        self.values = values
        self.output = fn(*values)
        return self.output


class SmallBatchTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
            if hasattr(hashlib, name):
                self.stack.enter_context(patch.object(hashlib, name, prohibited))
        self.torch = SimpleNamespace(cat=lambda values, dim: Tensor(
            [row for value in values for row in value.data]))
        self.wv = ModuleType("flowstar_gpu.weighted_validation")
        self.wv.__file__ = str(Path(__file__))
        self.wv._evaluate_map, self.wv.refine_accepted = _evaluate_map, refine_accepted
        self.wv._map = lambda code, a, b, c, plan, eng, cutoff: (a, b, c)
        self.se = ModuleType("flowstar_gpu.sparse_exec")
        self.se.__file__ = str(Path(__file__))
        self.se._code_serial = lambda code: "fixture"
        graphing = ModuleType("flowstar_gpu.graphing")
        graphing.GraphCache = Cache
        self.stack.enter_context(patch.dict(sys.modules, {
            m.__name__: m for m in (self.wv, self.se, graphing)}))

    def call(self, batch, engine, **kwargs):
        tensor = Tensor([[i] for i in range(batch)])
        return self.wv.refine_accepted(None, tensor, None, None, None, None,
                                      tensor, tensor, None, engine, 0.0, **kwargs)

    def test_selected_rows_rounds_owned_clones_and_restore(self):
        for rows, batch in ((1, 1), (16, 12), (32, 25), (32, 32)):
            with self.subTest(rows=rows, batch=batch):
                binding = install(self.torch, self.wv, self.se, rows=rows)
                engine, trace = SimpleNamespace(), {}
                result = self.call(batch, engine, trace=trace)
                self.assertEqual(trace, dict(rounds=2, chunk_size=rows, use_graph=True))
                self.assertEqual(binding.counters, dict(refine_calls=1, graph_map_calls=2,
                    target_rows=2 * batch, padded_rows=2 * (rows - batch), chunk_size=rows))
                for value in engine._weighted_graphs.values:
                    self.assertEqual(value.shape[0], rows)
                    self.assertEqual(value.data[batch:], [[batch - 1]] * (rows - batch))
                engine._weighted_graphs.output[0].data[0][0] = -99
                self.assertEqual(result[0].data, [[i] for i in range(batch)])
                binding.restore()
                self.assertIs(self.wv._evaluate_map, _evaluate_map)
                self.assertIs(self.wv.refine_accepted, refine_accepted)
                with self.assertRaises(RuntimeError):
                    binding.restore()

    def test_fallback_and_scope_guards(self):
        binding = install(self.torch, self.wv, self.se)
        self.assertEqual(self.wv._evaluate_map(*([None] * 8)), "original_evaluate")
        self.assertEqual(self.call(100, SimpleNamespace(), use_graph=False, rounds=7), "original_refine")
        for kwargs, batch in (({}, 0), ({}, 33), ({"rounds": 1}, 25), ({"chunk_size": 32}, 25)):
            with self.assertRaises(RuntimeError):
                self.call(batch, SimpleNamespace(), **kwargs)
        with self.assertRaises(RuntimeError):
            self.call(25, SimpleNamespace(_weighted_graphs=Cache("cuda:0")))
        self.call(25, SimpleNamespace())
        with self.assertRaises(RuntimeError):
            self.call(25, SimpleNamespace())
        binding.restore()

    def test_duplicate_foreign_owner_and_exception_restore(self):
        binding = install(self.torch, self.wv, self.se)
        with self.assertRaises(RuntimeError):
            install(self.torch, self.wv, self.se)
        with self.assertRaises(ValueError):
            self.call(25, SimpleNamespace(), trace={"raise": True})
        selected = self.wv._evaluate_map
        self.wv._evaluate_map = prohibited
        with self.assertRaises(RuntimeError):
            binding.restore()
        self.wv._evaluate_map = selected
        binding.restore()

    def test_invalid_rows(self):
        for rows in (0, 2, 64):
            with self.assertRaises(ValueError):
                install(self.torch, self.wv, self.se, rows=rows)
        for rows, batch in ((1, 2), (16, 17), (32, 33), (32, 0)):
            with self.assertRaises(ValueError):
                pad_rows(self.torch, Tensor([[0]] * batch), rows)


if __name__ == "__main__":
    unittest.main()
