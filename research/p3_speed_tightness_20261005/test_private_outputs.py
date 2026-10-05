"""CPU-only binding tests: no Torch import, CUDA, extension load or digest."""

from contextlib import ExitStack
import hashlib
import json
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from private_outputs import BASE_NAME, EXPORTS, PRIVATE_NAME, install


def prohibited(*_args, **_kwargs):
    raise AssertionError("availability, source generation, loading and digests forbidden")


class PrivateOutputsTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
            if hasattr(hashlib, name):
                self.stack.enter_context(patch.object(hashlib, name, prohibited))
        self.torch = ModuleType("torch")
        self.torch.__version__ = "2.5.1+cu121"
        self.ck = ModuleType("flowstar_gpu.cuda_kernels")
        self.private = ModuleType("flowstar_gpu.private_output_kernels")
        self.base = ModuleType(BASE_NAME)
        self.selected = ModuleType(PRIVATE_NAME)
        for module in (self.base, self.selected):
            for name in EXPORTS:
                setattr(module, name, prohibited)  # installation must not execute kernels
        self.ck._ext, self.ck._tried = self.base, True
        self.private._ext, self.private._tried = self.selected, True
        self.private.base, self.private.NAME, self.private.SUPPORTED = self.ck, PRIVATE_NAME, True
        self.private.available = self.private.sources = self.ck.available = prohibited
        self.ck.load_cuda_extension = prohibited
        self.stack.enter_context(patch.dict(sys.modules, {
            module.__name__: module for module in (self.ck, self.private, self.base, self.selected)
        }))

    def test_switch_is_only_mutation_and_restore_is_idempotent(self):
        before_ck, before_private = vars(self.ck).copy(), vars(self.private).copy()
        binding = install(self.torch, self.ck, self.private)
        self.assertIs(self.ck._ext, self.selected)
        self.assertEqual(vars(self.ck), {**before_ck, "_ext": self.selected})
        self.assertEqual(vars(self.private), before_private)
        self.assertFalse(binding.receipt["hash_or_jit_performed"])
        self.assertEqual(json.loads(json.dumps(binding.receipt))["exports_checked"], list(EXPORTS))
        binding.restore()
        binding.restore()
        self.assertEqual(vars(self.ck), before_ck)

    def test_rejects_configuration_without_mutation(self):
        cases = [
            (self.torch, "__version__", "2.6.0+cu124"),
            (self.ck, "_tried", False), (self.private, "_tried", False),
            (self.private, "SUPPORTED", False), (self.private, "base", None),
            (self.private, "NAME", "poison_binary"),
            (self.ck, "_ext", self.selected), (self.private, "_ext", None),
            (self.base, "iv_sum", None), (self.selected, "iv_neg", None),
        ]
        for module, name, value in cases:
            with self.subTest(module=module.__name__, field=name):
                with patch.object(module, name, value):
                    before = vars(self.ck).copy()
                    with self.assertRaises(RuntimeError):
                        install(self.torch, self.ck, self.private)
                    self.assertEqual(vars(self.ck), before)

    def test_rejects_unregistered_extension(self):
        with patch.dict(sys.modules, {PRIVATE_NAME: ModuleType(PRIVATE_NAME)}):
            with self.assertRaises(RuntimeError):
                install(self.torch, self.ck, self.private)
        self.assertIs(self.ck._ext, self.base)

    def test_duplicate_install_and_foreign_restore_are_rejected(self):
        binding = install(self.torch, self.ck, self.private)
        with self.assertRaises(RuntimeError):
            install(self.torch, self.ck, self.private)
        foreign = ModuleType("other_extension")
        self.ck._ext = foreign
        with self.assertRaises(RuntimeError):
            binding.restore()
        self.assertIs(self.ck._ext, foreign)
        self.ck._ext = self.selected
        binding.restore()
        self.assertIs(self.ck._ext, self.base)


if __name__ == "__main__":
    unittest.main()
