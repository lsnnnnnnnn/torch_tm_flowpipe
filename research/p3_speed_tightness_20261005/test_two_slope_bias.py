"""CPU-only exact-rational conversion and CLI checks; no Torch or GPU import.

The interval stub provides exact directed binary64 addition. This checks the
adapter's algebra/dispatch, not the production interval kernel or CROWN.
"""

import argparse
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


def prohibited(*_args, **_kwargs):
    raise AssertionError("content digests forbidden")


def directed(exact, upward):
    rounded = float(exact)
    if (Fraction(rounded) < exact if upward else Fraction(rounded) > exact):
        return math.nextafter(rounded, math.inf if upward else -math.inf)
    return rounded


class ExactInterval:
    @staticmethod
    def from_point(value):
        return value, value

    @staticmethod
    def add(left, right):
        return tuple(directed(Fraction(a) + Fraction(b), upward)
                     for a, b, upward in zip(left, right, (False, True)))


class TwoSlopeBiasTest(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
            if hasattr(hashlib, name):
                stack.enter_context(patch.object(hashlib, name, prohibited))
        path = Path(__file__).resolve().parents[2] / "tools/archcomp26_quad_paper_p3_nohash.py"
        spec = importlib.util.spec_from_file_location("paper_quad_two_slope_test", path)
        self.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runner)

    def test_outward_bias_contains_exact_affine_shift(self):
        tiny = math.ulp(0.0)
        cases = [(0.0, (-0.25, 0.75)), (1.0, (-2.0**-54, 2.0**-54)),
                 (-1.0, (-2.0**-54, 2.0**-54)), (1e100, (-1.0, 3.0)),
                 (1.0, (-1.0, -0.5)), (tiny, (-tiny, tiny))]
        for center, residual in cases:
            with self.subTest(center=center, residual=residual):
                lower, upper = self.runner.two_slope_bias_bounds(center, residual, ExactInterval)
                exact_lower = Fraction(center) + Fraction(residual[0])
                exact_upper = Fraction(center) + Fraction(residual[1])
                self.assertLessEqual(Fraction(lower), exact_lower)
                self.assertGreaterEqual(Fraction(upper), exact_upper)
                self.assertLessEqual(lower, upper)
        # Ordinary RN addition loses both sides of this nonzero interval.
        lower, upper = self.runner.two_slope_bias_bounds(1.0, (-2.0**-54, 2.0**-54), ExactInterval)
        self.assertLess(lower, 1.0)
        self.assertGreater(upper, 1.0)

    def test_cli_defaults_and_explicit_method(self):
        base = ["paper_quad", "--mode", "batch2", "--source-config", "source.yaml", "--output", "new-run"]
        for extra, expected in (([], "same-slope"), (["--crown-relax", "two-slope"], "two-slope")):
            with patch.object(sys, "argv", base + extra), patch.object(self.runner, "execute", return_value=0) as execute:
                with self.assertRaises(SystemExit) as ended:
                    self.runner.main()
                self.assertEqual(ended.exception.code, 0)
                self.assertEqual(execute.call_args.args[0].crown_relax, expected)
        # A legacy Namespace reaches the original output setup using the default.
        legacy = argparse.Namespace(output=Path("unused"), mode="batch2")
        with patch.object(Path, "mkdir", side_effect=RuntimeError("output setup")):
            with self.assertRaisesRegex(RuntimeError, "output setup"):
                self.runner.execute(legacy)


if __name__ == "__main__":
    unittest.main()
