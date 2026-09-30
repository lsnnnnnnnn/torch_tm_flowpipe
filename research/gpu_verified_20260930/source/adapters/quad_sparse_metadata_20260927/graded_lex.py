"""Standalone on-demand monomial index prototype, independent of the engine.

IDs match degree-major, then ascending exponent-tuple lexicographic order.
The basis stores only degree prefix counts, not the monomial universe.
"""
from bisect import bisect_right
from dataclasses import dataclass, field
from math import comb


@dataclass(frozen=True)
class GradedLexBasis:
    nvars: int
    maxdegree: int
    prefixes: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        if type(self.nvars) is not int or self.nvars < 1:
            raise ValueError('nvars must be a positive integer')
        if type(self.maxdegree) is not int or self.maxdegree < 0:
            raise ValueError('maxdegree must be a nonnegative integer')
        object.__setattr__(self, 'prefixes', tuple(
            comb(self.nvars + d, d) for d in range(self.maxdegree + 1)))

    @property
    def size(self):
        return self.prefixes[-1]

    def rank(self, exponents):
        exponents = tuple(exponents)
        if len(exponents) != self.nvars or any(type(e) is not int or e < 0 for e in exponents):
            raise ValueError('invalid exponent vector')
        remaining = sum(exponents)
        if remaining > self.maxdegree:
            raise ValueError('degree outside basis')
        index = self.prefixes[remaining - 1] if remaining else 0
        for i, exponent in enumerate(exponents[:-1]):
            slots = self.nvars - i - 1
            # Count preceding fixed-degree tuples at this coordinate. This
            # is sum_{a=0}^{exponent-1} C(remaining-a+slots-1, slots-1).
            index += comb(remaining + slots, slots) - comb(remaining - exponent + slots, slots)
            remaining -= exponent
        return index

    def unrank(self, index):
        if type(index) is not int or not 0 <= index < self.size:
            raise ValueError('index outside basis')
        degree = bisect_right(self.prefixes, index)
        residual = index - (self.prefixes[degree - 1] if degree else 0)
        remaining = degree
        exponents = []
        for i in range(self.nvars - 1):
            slots = self.nvars - i - 1
            for exponent in range(remaining + 1):
                count = comb(remaining - exponent + slots - 1, slots - 1)
                if residual < count:
                    break
                residual -= count
            exponents.append(exponent)
            remaining -= exponent
        exponents.append(remaining)
        assert residual == 0
        return tuple(exponents)
