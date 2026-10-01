"""No-digest copy of the pinned P3 metadata migration.

Call install(identity) once in a fresh process, then engine(n,k,h,device).
The global replacements are integer metadata factories, plus the metadata-only
weighted plan/factory. Active pairs keep the original stable a-major ordering.
No dense table or fake Tensor facade is supplied. Dense-only clients fail closed.
Only the source identity gate is replaced by source-path checks here.
"""
from pathlib import Path
from functools import lru_cache
from math import comb
from types import SimpleNamespace
import importlib.util, sys
import numpy as np
import torch
from flowstar_gpu import support as sp, sparse_exec as se, weighted_validation as wv
from flowstar_gpu import interval as iv, elementary as elem, injective_index as ii


def _load(name, filename, digest):
    path = Path(__file__).with_name(filename)
    assert path.is_file(), filename
    if name in sys.modules:
        assert Path(sys.modules[name].__file__).resolve() == path.resolve()
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


graded = _load('flowstar_gpu.graded_lex', 'graded_lex.py',
               '5181161bf78c16273b6b5725e1e6a449d7d3617c59b05a9db7656fe3eb551cd3')
legacy = _load('flowstar_gpu.quad_metadata_legacy', 'legacy_sparse_metadata.py',
               '67ebe12629e12617cfe56bbaafae5337ba41a9f54802d7cfa86bc2abafd822f2')
_original_validation = se.validation_engine_for_order
_original_weighted = wv._plan
_installed = None


@lru_cache(maxsize=64)
def basis(n, k):
    # Explicit limits: no claim about arbitrary n/k or higher validation order.
    if type(n) is not int or not 1 <= n <= 16 or type(k) is not int or not 2 <= k <= 5:
        raise ValueError('isolated metadata supports 1<=n<=16 and 2<=k<=5 only')
    return legacy.SparseBasis(n, k, max_rows=100_000, max_pairs=4_000_000)


@lru_cache(maxsize=4096)
def make_support(n, k, spatial, ids):
    b = basis(n, k)
    values = tuple(sorted({0, *b.bounded_ids(ids, spatial)}))
    if len(values) > b.max_rows:
        raise MemoryError('support with constant exceeds max_rows')
    from bisect import bisect_right
    degrees = tuple(bisect_right(b.family(spatial).prefixes, i) for i in values)
    prefix = (0,) + tuple(sum(d <= degree for d in degrees) for degree in range(2*k+1))
    return sp.Support(n, k, spatial, values, degrees, prefix)


@lru_cache(maxsize=4096)
def exponents(sup):
    assert sup == make_support(sup.n, sup.k, sup.spatial, sup.ids)
    rows = basis(sup.n, sup.k).exponents(sup.ids, sup.spatial)
    rows.flags.writeable = False
    return rows


def spatial_to_full(n, k, sup):
    assert sup.n == n and sup.k == k and sup.spatial
    b = basis(n, k)
    return make_support(n, k, False, tuple(b.full.rank((0,)+tuple(map(int, row))) for row in exponents(sup)))


def full_to_spatial(n, k, sup):
    assert sup.n == n and sup.k == k and not sup.spatial
    rows = exponents(sup)
    if rows[:, 0].any(): raise ValueError('support is not time free')
    b = basis(n, k)
    return make_support(n, k, True, tuple(b.spatial.rank(tuple(map(int, row[1:]))) for row in rows))


def pair_global_ids(ea, eb, family):
    """Integer rank of each a-major exponent sum, O(pair_count) temporary space."""
    remaining = (ea.sum(1)[:, None] + eb.sum(1)[None, :]).reshape(-1)
    prefixes = np.asarray((0,)+family.prefixes, dtype=np.int64)
    result = prefixes[remaining].copy()
    for j in range(family.nvars-1):
        slots = family.nvars-j-1
        values = (ea[:, j, None] + eb[None, :, j]).reshape(-1)
        lut = np.asarray([comb(d+slots, slots) for d in range(family.maxdegree+1)], dtype=np.int64)
        result += lut[remaining]-lut[remaining-values]
        remaining = remaining-values
    return result


class MetadataEngine(legacy.SparseMetadataEngine):
    def __init__(self, n, k, h, device='cpu', budget=None):
        b = basis(n, k)
        # Use the original power arithmetic on the target device. CPU powers
        # copied to CUDA are not a bitwise substitute for CUDA directed powers.
        from flowstar_gpu.polynomial import build_step_tables
        seed = SimpleNamespace(k=k, exponents=torch.empty(0,dtype=torch.long,device=device),
            cat_iv=torch.empty((0,2),dtype=torch.float64,device=device),
            t_deg=torch.empty(0,dtype=torch.long,device=device))
        raw = build_step_tables(seed,h)
        step = legacy.CompactStepTables(raw.delta,raw.pow_iv,raw.end_pow,raw.end_iv)
        super().__init__(b, step, device)
        # Shared between working/validation engines; bounded pair cache, no eviction
        # while CUDA graphs may retain plans. This is not total process-memory cap.
        self.pair_budget = budget if budget is not None else dict(bytes=0, limit=2*1024**3, bindings=0)

    def support(self, ids=(), spatial=False):
        values = tuple(self.tables.bounded_ids(ids, spatial))
        return make_support(self.tables.n, self.tables.k, spatial, values)

    def exponents(self, sup):
        self._check(sup)
        return exponents(sup)

    def pair(self, a, b):
        self._check(a); self._check(b, a.spatial)
        if max(a.degs) > self.tables.k or max(b.degs) > self.tables.k:
            raise ValueError('pair inputs must have degree <=k')
        count = a.size*b.size
        if count > self.tables.max_pairs:
            raise MemoryError(f'active pair count {count} exceeds {self.tables.max_pairs}')
        key = a, b
        if key not in self._pair:
            # Bound retained index bytes above before allocating this binding.
            worst_bytes = 32*count+24
            if self.pair_budget['bytes']+worst_bytes > self.pair_budget['limit']:
                raise MemoryError('cumulative live pair metadata budget exhausted')
            out = pair_global_ids(self.exponents(a), self.exponents(b), self.tables.family(a.spatial))
            sup = self.support(np.unique(out), a.spatial)
            local = np.searchsorted(sup.ids, out)
            perm = np.argsort(local, kind='stable')
            lengths = np.bincount(local[perm], minlength=sup.size)
            offsets = np.concatenate(([0], np.cumsum(lengths)))
            arrays = [perm//b.size, perm % b.size, lengths, offsets]
            values = [torch.from_numpy(v).to(self.device) for v in arrays]
            self._pair[key] = sp.PairBind(sup, *values, f'quad-meta[{sp._serial(a)}x{sp._serial(b)}]')
            self.pair_budget['bytes'] += sum(v.nbytes for v in arrays)
            self.pair_budget['bindings'] += 1
        return self._pair[key]

    def integ_plan(self, sup):
        self._check(sup, False)
        key = 'integ', sup
        if key not in self._injective:
            out, _, _, _ = self.integ(sup)
            targets = [self.tables.full.rank((int(row[0])+1,)+tuple(map(int,row[1:]))) for row in self.exponents(sup)]
            # No D2H access during capture: bind exact integer host metadata.
            self._injective[key] = ii.bind(np.searchsorted(out.ids, targets), out.size, self.device)
        return self._injective[key]


def validation_engine(eng, order):
    if not isinstance(eng, MetadataEngine): return _original_validation(eng, order)
    if order <= eng.tables.k: return eng
    if eng.step.lanes: raise ValueError('higher validation requires fixed steps')
    cache = se._eng_cache(eng, '_validation_engines')
    key = order, eng.step.delta
    if key not in cache:
        value = MetadataEngine(eng.tables.n, order, eng.step.delta, eng.device, eng.pair_budget)
        # Graded-lex IDs depend on n and the exponent only, not maxdegree.
        assert value.tables.full.prefixes[:2*eng.tables.k+1] == eng.tables.full.prefixes
        cache[key] = value
    return cache[key]


def weighted_plan(code, sup, initial_sup, var_sups, eng, cutoff):
    """a3fb _plan with only exponent lookup/ranking changed; _map is untouched."""
    if not isinstance(eng, MetadataEngine):
        return _original_weighted(code, sup, initial_sup, var_sups, eng, cutoff)
    cache = se._eng_cache(eng, '_weighted_plans')
    key = (se._code_serial(code), sup, initial_sup, var_sups, cutoff)
    if key in cache: return cache[key]
    if eng.step.lanes != 0 or not eng.step.delta > 0:
        raise ValueError('weighted validation requires a positive common step')
    n, k = eng.tables.n, eng.tables.k
    family = eng.tables.full
    rows = dict(zip(sup.ids, eng.exponents(sup)))
    time_id = family.rank((1,)+(0,)*n)
    full_sup = make_support(n, k, False, tuple(sorted(set(sup.ids)|{time_id})))
    weighted_vars = tuple(make_support(n,k,False,tuple(sorted(set(v.ids)|{time_id}))) for v in var_sups)
    spec = se.specialize(code, weighted_vars, code.order, full_sup, eng)
    # Match the dense factory's actual Tensor device ("cuda:0", not the
    # unresolved input spelling "cuda"); elem_tables also keys by this value.
    device = eng._cutoff_zero.device
    index = lambda v: torch.tensor(v, dtype=torch.long, device=device)
    input_positions = index([full_sup.ids.index(i) for i in sup.ids])
    zero_ids = [i for i in sup.ids if rows[i][0] == 0]
    if any(i not in zero_ids for i in initial_sup.ids):
        raise ValueError('initial polynomial must have only supported time-zero terms')
    zero_positions = index([sup.ids.index(i) for i in zero_ids])
    initial_positions = index([zero_ids.index(i) for i in initial_sup.ids])
    shifted_ids, shifted_positions = [], []
    for j, i in enumerate(sup.ids):
        powers = list(map(int, rows[i]))
        if powers[0]:
            powers[0] -= 1
            shifted_ids.append(family.rank(tuple(powers)))
            shifted_positions.append(j)
    field_ids = spec.sup_out_union.ids
    residual_sup = make_support(n,k,False,tuple(sorted(set(field_ids)|set(shifted_ids))))
    h = torch.tensor(eng.step.delta, dtype=torch.float64, device=device)
    h_iv = iv.from_point(h)
    divisors = torch.tensor([family.unrank(i)[0]+1 for i in field_ids], dtype=torch.float64, device=device)
    weights, bad_div = iv.div(h_iv, iv.from_point(divisors))
    if bool(bad_div.any()): raise ValueError('invalid positive integration divisor')
    weights = torch.where((divisors == 1)[:,None], h_iv, weights)
    cache[key] = dict(spec=spec,sup=full_sup,input_positions=input_positions,
        time_position=full_sup.ids.index(time_id),zero_positions=zero_positions,
        initial_positions=initial_positions,shifted_positions=index(shifted_positions),
        shifted_targets=index([residual_sup.ids.index(i) for i in shifted_ids]),
        field_targets=index([residual_sup.ids.index(i) for i in field_ids]),
        residual_size=residual_sup.size,factors=eng.factor(residual_sup),weights=weights,h_iv=h_iv,
        tabs=elem.elem_tables(code.order,str(device)) if code.has_elem else None)
    return cache[key]


def install(identity=None):
    """Explicit process-local changes without content-digest calls."""
    global _installed
    if _installed is not None:
        return
    assert se.VALIDATION_POLICY == 'solution_plus_one' and se.COMPOSITION_MODE == 'horner'
    assert se.SUPPORT_POLICY == 'structural' and se.INJECTIVE_GLUE and ii.ENABLED
    sp.make_support = make_support
    sp._exps_for = exponents
    sp.spatial_to_full_ids = spatial_to_full
    sp.full_to_spatial_ids = full_to_spatial
    se.validation_engine_for_order = validation_engine
    wv._plan = weighted_plan
    _installed = True


def engine(n, k, h, device='cpu'):
    if _installed is None: raise RuntimeError('call install before creating engine')
    value = MetadataEngine(n,k,h,device)
    # Only the working spatial prefix is stored (n16 P5: 20349 integers),
    # needed by unchanged a3fb affine initialization/normalization.
    b = value.tables
    if not hasattr(b, 'spatial_index'):
        indices = [b.full.rank((0,)+b.spatial.unrank(i)) for i in range(b.Ts)]
        object.__setattr__(b, 'spatial_index', torch.tensor(indices,dtype=torch.long))
    return value, legacy.sparse_horner_schedule(b)


def endpoint(st, eng):
    """Pinned strict endpoint contract with explicit sparse metadata queries."""
    if (st.pre.dtype != torch.float64 or st.pre_rem.dtype != torch.float64
            or st.pre.ndim != 3 or tuple(st.pre_rem.shape) != (*st.pre.shape[:2],2)
            or st.pre.shape[1] != eng.tables.n or st.pre.shape[0] == 0
            or tuple(st.status.shape) != (st.pre.shape[0],)
            or st.pre.device != eng._cutoff_zero.device or st.pre_rem.device != st.pre.device
            or st.status.device != st.pre.device or st.pre_sup.spatial
            or st.pre_sup.n != eng.tables.n or st.pre_sup.k != eng.tables.k
            or st.pre.shape[-1] != st.pre_sup.size):
        raise ValueError('strict endpoint requires matching FULL float64 state')
    if (not bool((st.status == 0).all()) or not bool(torch.isfinite(st.pre).all())
            or not bool(torch.isfinite(st.pre_rem).all())
            or not bool((st.pre_rem[...,0] <= st.pre_rem[...,1]).all())):
        raise ValueError('strict endpoint requires active finite ordered leaves')
    point, support, error = sp.evaluate_time_end_s_with_roundoff(st.pre,eng,st.pre_sup)
    remainder = iv.add(st.pre_rem,error)
    if (support.spatial or support.n != eng.tables.n or support.k != eng.tables.k
            or tuple(point.shape) != (*st.pre.shape[:2],support.size)
            or error.shape != st.pre_rem.shape or (eng.exponents(support)[:,0] != 0).any()
            or not bool(torch.isfinite(point).all() and torch.isfinite(error).all() and torch.isfinite(remainder).all())
            or not bool((error[...,0] <= error[...,1]).all() and (remainder[...,0] <= remainder[...,1]).all())):
        raise ValueError('strict endpoint produced invalid enclosure')
    point, remainder, error = point.clone(), remainder.clone(), error.clone()
    st.pre, st.pre_sup, st.pre_rem = point, support, remainder
    return error
