"""Small exact algebra and outward-conversion checks for offline export."""
from fractions import Fraction as Q
import math
import unittest

from experiments.whole_engine_feasibility.export import (
    exact_composition, canonical_models, _outward,
)


class ExportTests(unittest.TestCase):
    def test_shared_symbols_negative_coefficient_and_remainders(self):
        # pre = 5 - 2*z1^2 + 3*t*z2; z1 = 1 + u + [-1/4,1/4],
        # z2 = -2 + v; pre remainder [-1/8,1/8].
        result = exact_composition(
            [[5, -2, 3], [0, 0, 0]],
            [[Q(-1, 8), Q(1, 8)], [0, 0]],
            [(0, 0, 0), (0, 2, 0), (1, 0, 1)],
            [[1, 1, 0], [-2, 0, 1]],
            [[Q(-1, 4), Q(1, 4)], [0, 0]],
            [(0, 0, 0), (0, 1, 0), (0, 0, 1)],
        )[0]
        self.assertEqual(result, {
            (0, 0, 0): (Q(7, 4), Q(4)),
            (0, 1, 0): (Q(-5), Q(-3)),
            (0, 2, 0): (Q(-2), Q(-2)),
            (1, 0, 0): (Q(-6), Q(-6)),
            (1, 0, 1): (Q(3), Q(3)),
        })
        for t in [Q(0), Q(1, 10)]:
            for u in [Q(-1), Q(0), Q(1)]:
                for v in [Q(-1), Q(1)]:
                    lo = hi = Q(0)
                    for degrees, (a, b) in result.items():
                        monomial = t ** degrees[0] * u ** degrees[1] * v ** degrees[2]
                        lo += min(a * monomial, b * monomial)
                        hi += max(a * monomial, b * monomial)
                    for error in [Q(-1, 4), Q(0), Q(1, 4)]:
                        for rem in [Q(-1, 8), Q(1, 8)]:
                            value = 5 - 2 * (1 + u + error) ** 2 + 3 * t * (-2 + v) + rem
                            self.assertLessEqual(lo, value)
                            self.assertLessEqual(value, hi)

    def test_outward_and_reused_model(self):
        for value in [Q(1, 10), Q(-1, 10), Q(1, 3), Q(-1, 3),
                      Q(1, 2 ** 1075), Q(-1, 2 ** 1075)]:
            self.assertLessEqual(Q(_outward(value, True)), value)
            self.assertGreaterEqual(Q(_outward(value, False)), value)
        models = canonical_models([
            {(0, 1, 0): (Q(1, 10), Q(1, 10))},
            {(1, 0, 1): (Q(-1, 3), Q(-1, 3))},
        ], .01)
        self.assertEqual(models["tube"]["domain"][0], [0., .01])
        self.assertEqual(models["endpoint"]["domain"][0], [.01, .01])
        self.assertIs(models["tube"]["components"], models["endpoint"]["components"])
        self.assertEqual(models["tube"]["variables"], ["tau", "ux", "uy"])


if __name__ == "__main__":
    unittest.main()
