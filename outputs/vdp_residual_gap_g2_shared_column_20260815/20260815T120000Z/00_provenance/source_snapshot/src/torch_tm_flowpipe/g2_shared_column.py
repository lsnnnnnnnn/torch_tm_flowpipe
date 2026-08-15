"""Fixed-3d, two-generation shared-column carry for complete-O4 boundaries.

The G2 contract has exactly three banks of ``state_dim`` variables: normalized
base variables, one retained previous-generation source bank, and one fresh
source bank.  It is deliberately a single frozen candidate, not a configurable
K/generation family.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from .interval import Interval
from .polynomial import Polynomial


G2_SHARED_COLUMN_CANDIDATE = "normalized_insertion_bounded_shared_source_o4_g2"
G2_SHARED_COLUMN_SCHEMA = "complete_o4_g2_shared_column_carry"
G2_SHARED_COLUMN_SCHEMA_VERSION = 1
G2_SOURCE_GENERATIONS = 2


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def polynomial_table(polynomial: Polynomial) -> list[list[Any]]:
    return [
        [list(exponent), float(coefficient.detach().cpu()).hex()]
        for exponent, coefficient in sorted(polynomial.terms.items())
    ]


def polynomial_payload_sha256(polynomials: Sequence[Polynomial]) -> str:
    return canonical_hash([polynomial_table(poly) for poly in polynomials])


def support_sha256(polynomial: Polynomial) -> str:
    return canonical_hash([list(exponent) for exponent in sorted(polynomial.terms)])


@dataclass(frozen=True)
class G2SharedColumnState:
    """Immutable source-bank metadata for one accepted G2 boundary."""

    state_dim: int
    base_dim: int
    accepted_boundary_index: int
    generation: int
    retained_source_ids: tuple[str, ...]
    fresh_source_ids: tuple[str, ...]
    retained_active: tuple[bool, ...]
    fresh_active: tuple[bool, ...]
    fresh_radii_hex: tuple[str, ...]
    retained_lineage: tuple[tuple[str, ...], ...]
    fresh_lineage: tuple[tuple[str, ...], ...]
    retained_payload_sha256: str
    collapse_count: int = 0
    retired_source_count: int = 0
    schema: str = G2_SHARED_COLUMN_SCHEMA
    schema_version: int = G2_SHARED_COLUMN_SCHEMA_VERSION
    generations_retained: int = G2_SOURCE_GENERATIONS

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.base_dim != self.state_dim:
            raise ValueError("G2 uses one normalized base slot per state component")
        arrays = (
            self.retained_source_ids,
            self.fresh_source_ids,
            self.retained_active,
            self.fresh_active,
            self.fresh_radii_hex,
            self.retained_lineage,
            self.fresh_lineage,
        )
        if any(len(value) != self.state_dim for value in arrays):
            raise ValueError("G2 source-bank arrays must match state_dim")
        if self.generations_retained != 2:
            raise ValueError("G2 is frozen to exactly two source generations")
        if self.schema != G2_SHARED_COLUMN_SCHEMA or self.schema_version != 1:
            raise ValueError("G2 source schema mismatch")
        if len(self.retained_payload_sha256) != 64:
            raise ValueError("G2 retained payload must have a canonical SHA256")

    @staticmethod
    def initial(state_dim: int) -> "G2SharedColumnState":
        dim = int(state_dim)
        empty_payload = polynomial_payload_sha256(
            [Polynomial.zero(3 * dim) for _ in range(dim)]
        )
        return G2SharedColumnState(
            state_dim=dim,
            base_dim=dim,
            accepted_boundary_index=0,
            generation=0,
            retained_source_ids=tuple("" for _ in range(dim)),
            fresh_source_ids=tuple("" for _ in range(dim)),
            retained_active=tuple(False for _ in range(dim)),
            fresh_active=tuple(False for _ in range(dim)),
            fresh_radii_hex=tuple(float(0.0).hex() for _ in range(dim)),
            retained_lineage=tuple(tuple() for _ in range(dim)),
            fresh_lineage=tuple(tuple() for _ in range(dim)),
            retained_payload_sha256=empty_payload,
        )

    @property
    def oldest_indices(self) -> tuple[int, ...]:
        return tuple(range(self.base_dim, self.base_dim + self.state_dim))

    @property
    def current_indices(self) -> tuple[int, ...]:
        start = self.base_dim + self.state_dim
        return tuple(range(start, start + self.state_dim))

    @property
    def variable_count(self) -> int:
        return 3 * self.state_dim

    @property
    def live_source_count(self) -> int:
        return sum(self.retained_active) + sum(self.fresh_active)

    @property
    def fingerprint(self) -> str:
        return canonical_hash(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "state_dim": self.state_dim,
            "base_dim": self.base_dim,
            "variable_count": self.variable_count,
            "accepted_boundary_index": self.accepted_boundary_index,
            "generation": self.generation,
            "generations_retained": self.generations_retained,
            "retained_source_ids": list(self.retained_source_ids),
            "fresh_source_ids": list(self.fresh_source_ids),
            "retained_active": list(self.retained_active),
            "fresh_active": list(self.fresh_active),
            "fresh_radii_hex": list(self.fresh_radii_hex),
            "retained_lineage": [list(row) for row in self.retained_lineage],
            "fresh_lineage": [list(row) for row in self.fresh_lineage],
            "retained_payload_sha256": self.retained_payload_sha256,
            "collapse_count": self.collapse_count,
            "retired_source_count": self.retired_source_count,
            "live_source_count": self.live_source_count,
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value


@dataclass(frozen=True)
class SourcePartition:
    source_free: Polynomial
    source_bearing: Polynomial
    source_free_support_sha256: str
    source_bearing_support_sha256: str


def partition_source_terms(
    polynomial: Polynomial,
    source_indices: Sequence[int],
) -> SourcePartition:
    """Partition canonical terms without evaluating either side."""

    indices = frozenset(int(index) for index in source_indices)
    if any(index < 0 or index >= polynomial.n_vars for index in indices):
        raise IndexError("G2 source index lies outside polynomial")
    free: dict[tuple[int, ...], torch.Tensor] = {}
    bearing: dict[tuple[int, ...], torch.Tensor] = {}
    for exponent, coefficient in polynomial.terms.items():
        target = bearing if any(exponent[index] for index in indices) else free
        target[exponent] = coefficient
    free_poly = Polynomial(free, polynomial.n_vars)
    bearing_poly = Polynomial(bearing, polynomial.n_vars)
    return SourcePartition(
        source_free=free_poly,
        source_bearing=bearing_poly,
        source_free_support_sha256=support_sha256(free_poly),
        source_bearing_support_sha256=support_sha256(bearing_poly),
    )


def rotate_current_to_retained(polynomial: Polynomial, state_dim: int) -> Polynomial:
    """Rename the current bank into retained slots and clear fresh slots."""

    dim = int(state_dim)
    if polynomial.n_vars != 3 * dim:
        raise ValueError("G2 rotation requires exactly 3d variables")
    rotated: dict[tuple[int, ...], torch.Tensor] = {}
    for exponent, coefficient in polynomial.terms.items():
        if any(exponent[dim + index] for index in range(dim)):
            raise ValueError("oldest source survived before G2 bank rotation")
        updated = list(exponent)
        for index in range(dim):
            updated[dim + index] = exponent[2 * dim + index]
            updated[2 * dim + index] = 0
        key = tuple(updated)
        rotated[key] = rotated.get(key, torch.zeros_like(coefficient)) + coefficient
    return Polynomial(rotated, 3 * dim)


def accepted_successor(
    previous: G2SharedColumnState,
    fresh_radius: torch.Tensor,
    owner_categories: Sequence[str],
    *,
    retained_payload_sha256: str,
    retained_active: Sequence[bool],
) -> G2SharedColumnState:
    """Rotate banks for one accepted boundary; no rejected path calls this."""

    if fresh_radius.shape != (1, previous.state_dim):
        raise ValueError("G2 production successor requires B1 [1,state]")
    if fresh_radius.dtype != torch.float64 or not bool(torch.all(torch.isfinite(fresh_radius))):
        raise ValueError("G2 fresh radii must be finite float64")
    if not bool(torch.all(fresh_radius >= 0)):
        raise ValueError("G2 fresh radii must be nonnegative")
    retained_flags = tuple(bool(value) for value in retained_active)
    if len(retained_flags) != previous.state_dim:
        raise ValueError("G2 retained activity must match state_dim")
    boundary = previous.accepted_boundary_index + 1
    fresh_flags = tuple(bool(value > 0.0) for value in fresh_radius[0].detach().cpu())
    fresh_ids = tuple(
        f"boundary:{boundary}:component:{index}:complete_validated_ledger"
        if fresh_flags[index]
        else ""
        for index in range(previous.state_dim)
    )
    lineage = tuple(sorted(str(name) for name in owner_categories))
    return G2SharedColumnState(
        state_dim=previous.state_dim,
        base_dim=previous.base_dim,
        accepted_boundary_index=boundary,
        generation=previous.generation + 1,
        retained_source_ids=previous.fresh_source_ids,
        fresh_source_ids=fresh_ids,
        retained_active=retained_flags,
        fresh_active=fresh_flags,
        fresh_radii_hex=tuple(float(value).hex() for value in fresh_radius[0].detach().cpu()),
        retained_lineage=previous.fresh_lineage,
        fresh_lineage=tuple(lineage if flag else tuple() for flag in fresh_flags),
        retained_payload_sha256=str(retained_payload_sha256),
        collapse_count=previous.collapse_count + int(any(previous.retained_active)),
        retired_source_count=previous.retired_source_count + sum(previous.retained_active),
    )


def commit_or_preserve(
    previous: G2SharedColumnState,
    proposed: G2SharedColumnState,
    *,
    accepted: bool,
) -> G2SharedColumnState:
    """Commit exactly one generation on acceptance; preserve identity on retry."""

    if not accepted:
        return previous
    if proposed.accepted_boundary_index != previous.accepted_boundary_index + 1:
        raise ValueError("accepted G2 successor skips a boundary")
    return proposed


def metadata_tamper(state: G2SharedColumnState, label: str) -> G2SharedColumnState:
    fresh = tuple(tuple((*row, f"metadata:{label}")) for row in state.fresh_lineage)
    return replace(state, fresh_lineage=fresh)


def owner_rows(
    polynomial: Polynomial,
    domain: Sequence[Interval],
    *,
    component: int,
    oldest_indices: Sequence[int],
    current_indices: Sequence[int],
    oldest_source_ids: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Read-only interaction-aware accounting of terms retired at a boundary."""

    oldest = tuple(int(index) for index in oldest_indices)
    current = tuple(int(index) for index in current_indices)
    grouped: dict[tuple[Any, ...], dict[tuple[int, ...], torch.Tensor]] = {}
    for exponent, coefficient in polynomial.terms.items():
        owners = tuple(
            str(oldest_source_ids[offset])
            for offset, index in enumerate(oldest)
            if exponent[index] and str(oldest_source_ids[offset])
        )
        if not any(exponent[index] for index in oldest):
            continue
        total_degree = sum(int(value) for value in exponent)
        monomial_class = (
            "linear" if total_degree == 1 else
            "quadratic" if total_degree == 2 else
            "cubic" if total_degree == 3 else
            "quartic"
        )
        mixed_current = any(exponent[index] for index in current)
        key = (owners, monomial_class, mixed_current)
        grouped.setdefault(key, {})[exponent] = coefficient
    rows: list[Mapping[str, Any]] = []
    for (owners, monomial_class, mixed_current), terms in sorted(grouped.items(), key=str):
        grouped_poly = Polynomial(terms, polynomial.n_vars)
        enclosure = grouped_poly.evaluate_interval(domain)
        rows.append(
            {
                "component": int(component),
                "source_generation_ids": list(owners),
                "monomial_class": monomial_class,
                "oldest_current_mixed": bool(mixed_current),
                "vdp_x2y_nonlinear_path": monomial_class != "linear",
                "canonical_support_sha256": support_sha256(grouped_poly),
                "coefficient_payload_sha256": canonical_hash(polynomial_table(grouped_poly)),
                "term_count": len(grouped_poly.terms),
                "outward_lo_hex": float(enclosure.lo.detach().cpu()).hex(),
                "outward_hi_hex": float(enclosure.hi.detach().cpu()).hex(),
                "width": float(enclosure.width().detach().cpu()),
                "containment_witness": "natural_outward_interval_evaluation_on_full_3d_domain",
                "additivity": "interaction_owner_only_not_asserted_additive",
            }
        )
    return rows


__all__ = [
    "G2_SHARED_COLUMN_CANDIDATE",
    "G2_SHARED_COLUMN_SCHEMA",
    "G2_SHARED_COLUMN_SCHEMA_VERSION",
    "G2SharedColumnState",
    "SourcePartition",
    "accepted_successor",
    "canonical_hash",
    "commit_or_preserve",
    "metadata_tamper",
    "owner_rows",
    "partition_source_terms",
    "polynomial_payload_sha256",
    "polynomial_table",
    "rotate_current_to_retained",
    "support_sha256",
]
