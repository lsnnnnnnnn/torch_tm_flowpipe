"""Bounded one-generation source ledger for complete-O4 boundary carry.

The contract in this module is intentionally smaller than Flow*'s internal
symbolic-remainder object and different from the historical K16 interval-linear
queue.  At accepted boundary ``n`` it represents a state component as

    X_i = P_i(u) + R_o,i + rho_i z_i,       z_i in [-1, 1].

There is exactly one fresh source slot per state component.  ``rho_i z_i`` is
an affine representation of the complete additive validated-remainder ledger
for that component (an asymmetric ledger interval is recentered first).  The
same ``z_i`` is therefore present in every polynomial path of the *next* dense
Picard solve.  At the next accepted boundary all polynomial terms containing an
old source are merged by exponent and evaluated outward once; that interval is
retired into ``R_o``.  A new generation is then created from the new accepted
validated ledger.  The number of polynomial variables is consequently
``2 * state_dim`` at every boundary and cannot grow with the horizon.

Set obligations
---------------

``affine_lift_interval`` certifies

    [lo, hi] subseteq midpoint + radius * [-1, 1].

``collapse_source_polynomial`` partitions a canonical polynomial into the
source-free terms and all source-bearing terms.  Polynomial construction has
already merged duplicate exponents.  Natural outward interval evaluation of
the latter proves

    P(u, z) subseteq P_without_sources(u) + R_source.

The boundary bridge additionally requires the complete dense validated ledger
to contain the unchanged accepted Picard image.  Substituting its affine lift
for that image and adding the source-collapse interval therefore contains the
accepted endpoint.  Rejected candidates never call the bridge; the immutable
accepted state is returned byte-for-byte by ``commit_or_preserve``.

CPU float64 is the authoritative numerical lane.  CUDA uses the same tensor
operations for implementation-consistency testing, but is not advertised as a
formal directed-rounding implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from .interval import Interval
from .polynomial import Polynomial


BOUNDED_SOURCE_LEDGER_CANDIDATE = "normalized_insertion_bounded_source_ledger_o4_g1"
BOUNDED_SOURCE_LEDGER_SCHEMA = "complete_o4_bounded_source_ledger"
BOUNDED_SOURCE_LEDGER_SCHEMA_VERSION = 1
SOURCE_GENERATIONS_RETAINED = 1


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _tensor_hex(value: torch.Tensor) -> list[list[str]]:
    cpu = value.detach().to(device="cpu", dtype=torch.float64)
    return [[float(item).hex() for item in row] for row in cpu]


@dataclass(frozen=True)
class AffineLiftWitness:
    """Machine-checkable enclosure witness for an interval-to-source lift."""

    midpoint: torch.Tensor
    radius: torch.Tensor
    represented_lo: torch.Tensor
    represented_hi: torch.Tensor
    contains_input: torch.Tensor

    def as_dict(self) -> dict[str, Any]:
        return {
            "midpoint_hex": _tensor_hex(self.midpoint),
            "radius_hex": _tensor_hex(self.radius),
            "represented_lo_hex": _tensor_hex(self.represented_lo),
            "represented_hi_hex": _tensor_hex(self.represented_hi),
            "contains_input": self.contains_input.detach().cpu().tolist(),
        }


def affine_lift_interval(lo: torch.Tensor, hi: torch.Tensor) -> AffineLiftWitness:
    """Lift finite ``[lo, hi]`` tensors to outward affine sources.

    Every arithmetic boundary is expanded with ``nextafter``.  This is a
    binary64 containment construction, not a reliance on a rounded midpoint
    identity.
    """

    if lo.shape != hi.shape or lo.ndim != 2:
        raise ValueError("source intervals must have matching [batch,state] shape")
    if lo.dtype != torch.float64 or hi.dtype != torch.float64:
        raise TypeError("the authoritative source lift requires float64")
    if lo.device != hi.device:
        raise ValueError("source interval bounds must share a device")
    if not bool(torch.all(torch.isfinite(lo)) and torch.all(torch.isfinite(hi))):
        raise ValueError("source interval must be finite")
    if not bool(torch.all(lo <= hi)):
        raise ValueError("source lower bounds must not exceed upper bounds")
    neg_inf = torch.full_like(lo, -torch.inf)
    pos_inf = torch.full_like(lo, torch.inf)
    midpoint = lo + (hi - lo) * 0.5
    lower_distance = torch.nextafter(midpoint - lo, pos_inf)
    upper_distance = torch.nextafter(hi - midpoint, pos_inf)
    radius = torch.nextafter(torch.maximum(lower_distance, upper_distance), pos_inf)
    represented_lo = torch.nextafter(midpoint - radius, neg_inf)
    represented_hi = torch.nextafter(midpoint + radius, pos_inf)
    contains = (represented_lo <= lo) & (represented_hi >= hi)
    # A second deterministic outward ulp is cheap and protects the contract
    # from device-specific intermediate rounding without changing decisions.
    if not bool(torch.all(contains)):
        radius = torch.nextafter(radius, pos_inf)
        represented_lo = torch.nextafter(midpoint - radius, neg_inf)
        represented_hi = torch.nextafter(midpoint + radius, pos_inf)
        contains = (represented_lo <= lo) & (represented_hi >= hi)
    if not bool(torch.all(contains)):
        raise FloatingPointError("affine source lift failed interval containment")
    return AffineLiftWitness(midpoint, radius, represented_lo, represented_hi, contains)


@dataclass(frozen=True)
class SourceCollapseWitness:
    """A source-free polynomial plus an outward interval for retired terms."""

    retained: Polynomial
    collapsed: Interval
    source_term_count: int
    retained_term_count: int
    source_support_sha256: str
    retained_support_sha256: str


def _support_hash(poly: Polynomial) -> str:
    rows = [
        [list(exponent), float(coefficient.detach().cpu()).hex()]
        for exponent, coefficient in sorted(poly.terms.items())
    ]
    return _canonical_hash(rows)


def collapse_source_polynomial(
    polynomial: Polynomial,
    domain: Sequence[Interval],
    source_indices: Sequence[int],
) -> SourceCollapseWitness:
    """Retire every term containing a source variable, after exponent merge."""

    if len(domain) != polynomial.n_vars:
        raise ValueError("collapse domain and polynomial variable count disagree")
    sources = frozenset(int(index) for index in source_indices)
    if any(index < 0 or index >= polynomial.n_vars for index in sources):
        raise IndexError("source variable index is outside the polynomial")
    retained_terms: dict[tuple[int, ...], torch.Tensor] = {}
    source_terms: dict[tuple[int, ...], torch.Tensor] = {}
    for exponent, coefficient in polynomial.terms.items():
        target = (
            source_terms
            if any(int(exponent[index]) != 0 for index in sources)
            else retained_terms
        )
        target[exponent] = coefficient
    retained = Polynomial(retained_terms, polynomial.n_vars)
    source_poly = Polynomial(source_terms, polynomial.n_vars)
    collapsed = source_poly.evaluate_interval(domain)
    return SourceCollapseWitness(
        retained=retained,
        collapsed=collapsed,
        source_term_count=len(source_poly.terms),
        retained_term_count=len(retained.terms),
        source_support_sha256=_support_hash(source_poly),
        retained_support_sha256=_support_hash(retained),
    )


@dataclass(frozen=True)
class BoundedSourceLedgerState:
    """Immutable accepted-boundary state for the one-generation policy."""

    state_dim: int
    base_dim: int
    accepted_boundary_index: int
    generation: int
    source_ids: tuple[str, ...]
    active: tuple[bool, ...]
    radii_hex: tuple[str, ...]
    lineage: tuple[tuple[str, ...], ...]
    collapse_count: int = 0
    retired_source_count: int = 0
    schema: str = BOUNDED_SOURCE_LEDGER_SCHEMA
    schema_version: int = BOUNDED_SOURCE_LEDGER_SCHEMA_VERSION
    generations_retained: int = SOURCE_GENERATIONS_RETAINED

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.base_dim <= 0:
            raise ValueError("source-ledger dimensions must be positive")
        if self.base_dim != self.state_dim:
            raise ValueError("the bounded O4 contract uses one base slot per state")
        expected = self.state_dim
        if any(
            len(value) != expected
            for value in (self.source_ids, self.active, self.radii_hex, self.lineage)
        ):
            raise ValueError("source-ledger slot arrays must match state_dim")
        if self.generations_retained != 1:
            raise ValueError("this contract is exactly one-generation")
        if self.schema != BOUNDED_SOURCE_LEDGER_SCHEMA:
            raise ValueError("source-ledger schema mismatch")
        if self.schema_version != BOUNDED_SOURCE_LEDGER_SCHEMA_VERSION:
            raise ValueError("source-ledger schema version mismatch")

    @staticmethod
    def initial(state_dim: int) -> "BoundedSourceLedgerState":
        dim = int(state_dim)
        return BoundedSourceLedgerState(
            state_dim=dim,
            base_dim=dim,
            accepted_boundary_index=0,
            generation=0,
            source_ids=tuple("" for _ in range(dim)),
            active=tuple(False for _ in range(dim)),
            radii_hex=tuple(float(0.0).hex() for _ in range(dim)),
            lineage=tuple(tuple() for _ in range(dim)),
        )

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(range(self.base_dim, self.base_dim + self.state_dim))

    @property
    def live_source_count(self) -> int:
        return sum(self.active)

    @property
    def fingerprint(self) -> str:
        return _canonical_hash(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "state_dim": self.state_dim,
            "base_dim": self.base_dim,
            "accepted_boundary_index": self.accepted_boundary_index,
            "generation": self.generation,
            "generations_retained": self.generations_retained,
            "source_ids": list(self.source_ids),
            "active": list(self.active),
            "radii_hex": list(self.radii_hex),
            "lineage": [list(row) for row in self.lineage],
            "collapse_count": self.collapse_count,
            "retired_source_count": self.retired_source_count,
            "live_source_count": self.live_source_count,
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value


def accepted_successor(
    previous: BoundedSourceLedgerState,
    radius: torch.Tensor,
    owner_categories: Sequence[str],
) -> BoundedSourceLedgerState:
    """Create the only legal successor of an accepted boundary state."""

    if radius.shape != (1, previous.state_dim):
        raise ValueError("production successor currently requires B1 [1,state]")
    if radius.dtype != torch.float64 or not bool(torch.all(torch.isfinite(radius))):
        raise ValueError("source radii must be finite float64")
    if not bool(torch.all(radius >= 0)):
        raise ValueError("source radii must be nonnegative")
    boundary = previous.accepted_boundary_index + 1
    active = tuple(bool(value > 0.0) for value in radius[0].detach().cpu().tolist())
    source_ids = tuple(
        f"boundary:{boundary}:component:{index}:validated_remainder_aggregate"
        if active[index]
        else ""
        for index in range(previous.state_dim)
    )
    lineage_row = tuple(sorted(str(name) for name in owner_categories))
    return BoundedSourceLedgerState(
        state_dim=previous.state_dim,
        base_dim=previous.base_dim,
        accepted_boundary_index=boundary,
        generation=previous.generation + 1,
        source_ids=source_ids,
        active=active,
        radii_hex=tuple(float(value).hex() for value in radius[0].detach().cpu()),
        lineage=tuple(lineage_row if flag else tuple() for flag in active),
        collapse_count=previous.collapse_count + int(previous.live_source_count > 0),
        retired_source_count=previous.retired_source_count + previous.live_source_count,
    )


def commit_or_preserve(
    previous: BoundedSourceLedgerState,
    proposed: BoundedSourceLedgerState,
    *,
    accepted: bool,
) -> BoundedSourceLedgerState:
    """Commit atomically on acceptance; preserve object identity on rejection."""

    if not accepted:
        return previous
    if proposed.accepted_boundary_index != previous.accepted_boundary_index + 1:
        raise ValueError("accepted source-ledger successor skips a boundary")
    return proposed


def metadata_tamper(state: BoundedSourceLedgerState, label: str) -> BoundedSourceLedgerState:
    """Return a metadata-only variant used by the consumer tamper oracle."""

    lineage = tuple(tuple((*row, f"metadata:{label}")) for row in state.lineage)
    return replace(state, lineage=lineage)


def source_payload_hash(midpoint: torch.Tensor, radius: torch.Tensor) -> str:
    """Hash only fields that alter the actual affine Picard input."""

    return _canonical_hash(
        {"midpoint_hex": _tensor_hex(midpoint), "radius_hex": _tensor_hex(radius)}
    )
