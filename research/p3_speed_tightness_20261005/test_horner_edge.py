"""CPU-only binding checks; no numerical engine, Torch, GPU or digests."""

from contextlib import ExitStack
import hashlib
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from horner_edge import EXTENSION_NAME, install


def prohibited(*args, **kwargs):
    raise AssertionError("loading, arithmetic, JIT and digests are forbidden")


class HornerBindingTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
            if hasattr(hashlib, name):
                self.stack.enter_context(patch.object(hashlib, name, prohibited))
        self.torch = ModuleType("torch")
        self.torch.__version__ = "2.5.1+cu121"
        self.torch.float64 = object()
        self.torch.device = lambda value: value
        class Tensor:
            is_cuda = True
            dtype = self.torch.float64
            device = "cuda:0"
        self.torch.Tensor = Tensor
        class Engine:
            def __init__(self):
                self.device = "cuda:0"
                self._cutoff_zero = Tensor()
        self.Engine = Engine
        self.support = ModuleType("flowstar_gpu.support")
        self.support.SparseEngine = Engine
        self.se = ModuleType("flowstar_gpu.sparse_exec")
        self.se.COMPOSITION_MODE = "horner"
        self.edge = ModuleType("flowstar_gpu.horner_edge_kernels")
        self.edge.NAME = EXTENSION_NAME
        self.edge._ext = ModuleType(EXTENSION_NAME)
        self.edge._ext.horner_edge = prohibited
        self.edge.load = prohibited
        modules = (self.torch, self.support, self.se, self.edge, self.edge._ext)
        self.stack.enter_context(patch.dict(sys.modules, {
            module.__name__: module for module in modules
        }))

    def bind(self, engine):
        return install(self.torch, engine, self.se, self.edge)

    def test_absent_and_none_restore_without_other_mutation(self):
        for present in (False, True):
            with self.subTest(present=present):
                engine = self.Engine()
                if present:
                    engine.horner_edge_kernel = None
                before = vars(engine).copy()
                binding = self.bind(engine)
                self.assertEqual(vars(engine), {
                    **before, "horner_edge_kernel": self.edge._ext.horner_edge})
                self.assertFalse(binding.receipt["hash_or_jit_performed"])
                binding.restore()
                binding.restore()
                self.assertEqual(vars(engine), before)

    def test_rejections_leave_engine_unchanged(self):
        cases = (
            (self.torch, "__version__", "2.6.0"),
            (self.se, "COMPOSITION_MODE", "monomial"),
            (self.edge, "NAME", "unexpected"),
            (self.edge, "_ext", None),
            (self.edge._ext, "horner_edge", None),
        )
        for owner, field, value in cases:
            with self.subTest(field=field), patch.object(owner, field, value):
                engine = self.Engine()
                before = vars(engine).copy()
                with self.assertRaises(RuntimeError):
                    self.bind(engine)
                self.assertEqual(vars(engine), before)
        for field, value in (("_graphs", SimpleNamespace()),
                             ("_weighted_graphs", SimpleNamespace()),
                             ("horner_edge_kernel", prohibited)):
            engine = self.Engine()
            setattr(engine, field, value)
            before = vars(engine).copy()
            with self.assertRaises(RuntimeError):
                self.bind(engine)
            self.assertEqual(vars(engine), before)
        engine = self.Engine()
        engine._cutoff_zero.is_cuda = False
        with self.assertRaises(RuntimeError):
            self.bind(engine)
        with self.assertRaises(RuntimeError):
            self.bind(SimpleNamespace())

    def test_unregistered_module_rejected(self):
        with patch.dict(sys.modules, {EXTENSION_NAME: ModuleType(EXTENSION_NAME)}):
            with self.assertRaises(RuntimeError):
                self.bind(self.Engine())

    def test_duplicate_and_foreign_restore_rejected(self):
        engine = self.Engine()
        binding = self.bind(engine)
        with self.assertRaises(RuntimeError):
            self.bind(engine)
        foreign = lambda: None
        engine.horner_edge_kernel = foreign
        with self.assertRaises(RuntimeError):
            binding.restore()
        self.assertIs(engine.horner_edge_kernel, foreign)


if __name__ == "__main__":
    unittest.main()
