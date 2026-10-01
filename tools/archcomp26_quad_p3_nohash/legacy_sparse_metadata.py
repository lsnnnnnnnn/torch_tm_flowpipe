"""On-demand sparse metadata prototype with an explicit limited advance bridge.

No dense degree-2k basis or Cartesian working-basis pair table is built. Global
monomial IDs and local pair reduction order equal the existing dense factory.
SparseEngine's bindings are reused where they do not call dense constructors.
The compact schedule permits explicitly selected scalar-step strict Horner.
CUDA bindings are experimental and unqualified; full_solver_supported=False.
This module neither monkey-patches production code nor pretends to be MonomialTables.
"""
from dataclasses import dataclass
from itertools import islice
from numbers import Integral
from types import SimpleNamespace

import numpy as np
import torch

from . import interval as iv
from .graded_lex import GradedLexBasis
from .monomials import CATEGORY_INTERVALS
from .polynomial import StepTables, build_step_tables, build_step_tables_batched
from .support import Support, SparseEngine, PairBind, _serial


@dataclass(frozen=True)
class SparseBasis:
    n: int
    k: int
    max_rows: int = 100_000
    max_pairs: int = 1_000_000

    def __post_init__(self):
        if type(self.n) is not int or self.n < 1 or type(self.k) is not int or self.k < 2:
            raise ValueError('need integer n >= 1 and order >= 2')
        if type(self.max_rows) is not int or self.max_rows < 1 or type(self.max_pairs) is not int or self.max_pairs < 1:
            raise ValueError('positive integer metadata limits required')
        object.__setattr__(self, 'full', GradedLexBasis(self.n + 1, 2 * self.k))
        object.__setattr__(self, 'spatial', GradedLexBasis(self.n, 2 * self.k))
        if self.T2 >= 2**63:
            raise ValueError('global basis IDs do not fit the existing int64 API')

    device = 'cpu'
    @property
    def T(self): return self.full.prefixes[self.k]
    @property
    def T2(self): return self.full.size
    @property
    def Ts(self): return self.spatial.prefixes[self.k]
    @property
    def Ts2(self): return self.spatial.size
    @property
    def prefix_len_cpu(self): return (0,) + self.full.prefixes
    @property
    def sp_prefix_len_cpu(self): return (0,) + self.spatial.prefixes

    def family(self, spatial):
        if type(spatial) is not bool:
            raise TypeError('spatial must be a boolean')
        return self.spatial if spatial else self.full

    def bounded_ids(self, ids, spatial):
        basis = self.family(spatial)
        if hasattr(ids, '__len__') and len(ids) > self.max_rows:
            raise MemoryError('requested support exceeds max_rows before materialization')
        ids = tuple(islice(iter(ids), self.max_rows + 1))
        if len(ids) > self.max_rows:
            raise MemoryError('requested iterable exceeds max_rows')
        if any(not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < basis.size for i in ids):
            raise ValueError('invalid global monomial ID')
        return tuple(int(i) for i in ids)

    def exponents(self, ids, spatial=False):
        basis = self.family(spatial)
        ids = self.bounded_ids(ids, spatial)
        return np.asarray([basis.unrank(i) for i in ids], dtype=np.int64).reshape(len(ids), basis.nvars)

    def onehot_ids(self, spatial=False):
        basis = self.family(spatial)
        return tuple(basis.rank(tuple(int(j == i) for j in range(basis.nvars))) for i in range(basis.nvars))


@dataclass(frozen=True)
class CompactStepTables:
    delta: float
    pow_iv: torch.Tensor
    end_pow: torch.Tensor
    end_iv: torch.Tensor
    lanes: int = 0
    deltas: torch.Tensor | None = None

    # Reuse the repaired strict endpoint and broadcasting accessors verbatim.
    end_gather = StepTables.end_gather
    end_iv_gather = StepTables.end_iv_gather
    dt = StepTables.dt

    def to(self, device):
        return CompactStepTables(self.delta, self.pow_iv.to(device), self.end_pow.to(device),
                                 self.end_iv.to(device), self.lanes,
                                 None if self.deltas is None else self.deltas.to(device))

    def factor(self, *args, **kwargs):
        raise NotImplementedError('compact steps require engine.factor(support), not a dense global slice')


def compact_step_tables(basis, delta):
    """Reuse existing time-power arithmetic with zero factor rows, then discard
    that private seed. No dummy dense table or factor tensor escapes this API.
    """
    seed = SimpleNamespace(k=basis.k, exponents=torch.empty(0, dtype=torch.int64),
                           cat_iv=torch.empty((0, 2), dtype=torch.float64),
                           t_deg=torch.empty(0, dtype=torch.int64))
    if isinstance(delta, torch.Tensor):
        if delta.device.type != 'cpu' or delta.dtype != torch.float64 or delta.ndim != 1 or not delta.numel():
            raise ValueError('CPU float64 nonempty vector required for per-lane steps')
        if not bool(torch.isfinite(delta).all()):
            raise ValueError('finite steps required')
        original = build_step_tables_batched(seed, delta)
    else:
        delta = float(delta)
        if not np.isfinite(delta):
            raise ValueError('finite step required')
        original = build_step_tables(seed, delta)
    return CompactStepTables(original.delta, original.pow_iv, original.end_pow,
                             original.end_iv, original.lanes, original.deltas)


class SparseMetadataEngine(SparseEngine):
    """Existing binding interface with support-sized CPU metadata only.

    Intentionally separate construction and support factories. Do not pass this
    object to advance_sparse/initial_sparse_state until their global dense
    helpers and direct table consumers have been explicitly adapted/tested.
    """
    def __init__(self, basis, step, device='cpu'):
        device = str(torch.device(device))
        if torch.device(device).type not in ('cpu', 'cuda'):
            raise ValueError('on-demand metadata supports explicit CPU or CUDA only')
        super().__init__(basis, step.to(device), device)
        self._supports = {}
        self._rows = {}

    def support(self, ids=(), spatial=False):
        ids = tuple(sorted({0, *self.tables.bounded_ids(ids, spatial)}))
        if len(ids) > self.tables.max_rows:
            raise MemoryError('support with required constant exceeds max_rows')
        key = spatial, ids
        if key not in self._supports:
            basis = self.tables.family(spatial)
            from bisect import bisect_right
            degs = tuple(bisect_right(basis.prefixes, i) for i in ids)
            prefix = (0,) + tuple(sum(d <= order for d in degs) for order in range(2 * self.tables.k + 1))
            self._supports[key] = Support(self.tables.n, self.tables.k, spatial, ids, degs, prefix)
        return self._supports[key]

    def _check(self, sup, spatial=None):
        if (sup.n, sup.k) != (self.tables.n, self.tables.k) or (spatial is not None and sup.spatial != spatial):
            raise ValueError('support family mismatch')
        if self.support(sup.ids, sup.spatial) != sup:
            raise ValueError('noncanonical support metadata')

    def exponents(self, sup):
        self._check(sup)
        if sup not in self._rows:
            rows = self.tables.exponents(sup.ids, sup.spatial)
            rows.flags.writeable = False
            self._rows[sup] = rows
        return self._rows[sup]

    def ids_t(self, sup):
        self._check(sup)
        return super().ids_t(sup)

    def embed_idx(self, small, big):
        self._check(small); self._check(big, small.spatial)
        return super().embed_idx(small, big)

    def range_idx(self, sup):
        self._check(sup)
        return super().range_idx(sup)

    def full_support(self, order, spatial=False):
        if type(order) is not int or not 0 <= order <= 2 * self.tables.k:
            raise ValueError('invalid full-support degree')
        return self.support(range(self.tables.family(spatial).prefixes[order]), spatial)

    def extract(self, coeffs, spatial, base=None):
        if base is not None:
            self._check(base, spatial)
        mask = (coeffs != 0).reshape(-1, coeffs.shape[-1]).any(dim=0)
        positions = mask.nonzero(as_tuple=True)[0].cpu().numpy()
        ids = np.asarray(base.ids, dtype=np.int64)[positions] if base is not None else positions
        return self.support(ids, spatial)

    def spatial_ids_to_full(self, ids):
        ids = self.tables.bounded_ids(ids, True)
        return tuple(self.tables.full.rank((0,) + self.tables.spatial.unrank(i)) for i in ids)

    def pair(self, a, b):
        self._check(a); self._check(b, a.spatial)
        if max(a.degs) > self.tables.k or max(b.degs) > self.tables.k:
            raise ValueError('pair inputs must have degree <= k')
        if a.size * b.size > self.tables.max_pairs:
            raise MemoryError('support Cartesian product exceeds max_pairs')
        key = a, b
        if key not in self._pair:
            family = self.tables.family(a.spatial)
            ea, eb = self.exponents(a), self.exponents(b)
            # Same a-major enumeration, then stable output sort as pair_tables.
            out = np.fromiter((family.rank(tuple(int(x) + int(y) for x, y in zip(ra, rb)))
                               for ra in ea for rb in eb), dtype=np.int64, count=a.size * b.size)
            sup = self.support(np.unique(out), a.spatial)
            slots = np.searchsorted(np.asarray(sup.ids, dtype=np.int64), out)
            perm = np.argsort(slots, kind='stable')
            lengths = np.bincount(slots[perm], minlength=sup.size)
            offsets = np.concatenate(([0], np.cumsum(lengths)))
            self._pair[key] = PairBind(sup, torch.from_numpy(perm // b.size).to(self.device), torch.from_numpy(perm % b.size).to(self.device),
                                       torch.from_numpy(lengths).to(self.device), torch.from_numpy(offsets).to(self.device),
                                       f'sparse-meta[{_serial(a)}x{_serial(b)}]')
        return self._pair[key]

    def union(self, a, b):
        self._check(a); self._check(b, a.spatial)
        key = a, b
        if key not in self._sub:
            out = self.support(np.union1d(a.ids, b.ids), a.spatial)
            self._sub[key] = (out, self.embed_idx(a, out), self.embed_idx(b, out))
        return self._sub[key]

    def integ(self, sup):
        self._check(sup, False)
        if max(sup.degs) >= 2 * self.tables.k:
            raise ValueError('integration input exceeds degree 2k-1')
        if sup not in self._integ:
            rows = self.exponents(sup)
            targets = [self.tables.full.rank((int(row[0]) + 1,) + tuple(int(x) for x in row[1:])) for row in rows]
            out = self.support(targets)
            tgt = torch.from_numpy(np.searchsorted(out.ids, targets))
            denominators = rows[:, 0] + 1
            scale = 1.0 / denominators.astype(np.float64)
            exact = (denominators & (denominators - 1)) == 0
            scale_iv = np.stack((np.where(exact, scale, np.nextafter(scale, -np.inf)),
                                 np.where(exact, scale, np.nextafter(scale, np.inf))), axis=1)
            self._integ[sup] = out, tgt.to(self.device), torch.from_numpy(scale).to(self.device), torch.from_numpy(scale_iv).to(self.device)
        return self._integ[sup]

    def _time_map(self, sup):
        self._check(sup, False)
        rows = self.exponents(sup)
        targets = [self.tables.full.rank((0,) + tuple(int(x) for x in row[1:])) for row in rows]
        out = self.support(targets)
        return out, torch.from_numpy(np.searchsorted(out.ids, targets)).to(self.device), torch.tensor(rows[:, 0].copy(), device=self.device)

    def evalt(self, sup, ref_rank=3):
        key = sup, ref_rank
        if key not in self._evalt:
            out, tgt, tdeg = self._time_map(sup)
            self._evalt[key] = out, tgt, self.step.end_gather(tdeg, ref_rank)
        return self._evalt[key]

    def evalt_interval(self, sup, ref_rank=3):
        key = sup, ref_rank
        if key not in self._evalt_iv:
            _, _, tdeg = self._time_map(sup)
            self._evalt_iv[key] = self.step.end_iv_gather(tdeg, ref_rank)
        return self._evalt_iv[key]

    def evalt_error(self, sup):
        if sup not in self._evalt_error:
            from .rounding import prepare_dot_error_bound
            out, tgt, _ = self._time_map(sup)
            self._evalt_error[sup] = prepare_dot_error_bound(torch.bincount(tgt.cpu(), minlength=out.size).to(self.device))
        return self._evalt_error[sup]

    def _categories(self, sup):
        rows = self.exponents(sup)
        spatial = rows if sup.spatial else rows[:, 1:]
        category = np.where((spatial % 2 == 1).any(axis=1), 2,
                            np.where((spatial != 0).any(axis=1), 1, 0))
        return torch.from_numpy(CATEGORY_INTERVALS[category]).to(self.device)

    def factor(self, sup):
        self._check(sup, False)
        if sup not in self._factor:
            tdeg = torch.tensor(self.exponents(sup)[:, 0].copy(), device=self.device)
            cat = self._categories(sup)
            powers = self.step.pow_iv[tdeg] if not self.step.lanes else self.step.pow_iv[:, tdeg]
            value = iv.mul(cat, powers)
            self._factor[sup] = value.contiguous() if not self.step.lanes else value.unsqueeze(1).contiguous()
        return self._factor[sup]

    def cat(self, sup):
        self._check(sup, True)
        if sup not in self._cat:
            self._cat[sup] = self._categories(sup).contiguous()
        return self._cat[sup]

    def spatial_to_full(self, sup):
        self._check(sup, True)
        return self.support([self.tables.full.rank((0,) + tuple(int(x) for x in row)) for row in self.exponents(sup)])

    def full_to_spatial(self, sup):
        self._check(sup, False)
        rows = self.exponents(sup)
        if rows[:, 0].any():
            raise ValueError('input is not time-free')
        return self.support([self.tables.spatial.rank(tuple(int(x) for x in row[1:])) for row in rows], True)


def build_sparse_metadata(n, order, delta, *, device="cpu", max_rows=100_000, max_pairs=1_000_000):
    basis = SparseBasis(n, order, max_rows, max_pairs)
    return SparseMetadataEngine(basis, compact_step_tables(basis, delta), device)


@dataclass(frozen=True)
class SparseHornerSchedule:
    n: int
    k: int
    Ts: int
    var_image: torch.Tensor
    horner_only = True

    def _unsupported(self):
        raise NotImplementedError('compact schedule has no monomial-composition DAG; select Horner explicitly')
    parent = property(_unsupported)
    lastvar = property(_unsupported)
    levels = property(_unsupported)


def sparse_horner_schedule(basis):
    """Only canonical one-hot IDs; full composition schedules stay unchanged."""
    return SparseHornerSchedule(basis.n, basis.k, basis.Ts,
                               torch.tensor(basis.onehot_ids(True), dtype=torch.int64))
