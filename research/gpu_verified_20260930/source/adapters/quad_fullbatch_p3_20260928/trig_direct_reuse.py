"""Candidate: call-local reuse of identical direct-variable sin/cos series.

No compiler/engine source edits. Original instruction/cache/tail layouts stay
intact. Install only with fresh graph owners; discard them before restore.
Plan keys use CPU Support values and shared device-table OBJECT identities,
never device values. No candidate CUDA or trajectory qualification is implied.
"""
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
import hashlib
import inspect

PINS = {
    'se': '56e590796dbae42685b0331c7f41b53b62a3f73ed9be74e22738493820353ddf',
    'elem': '3e9d7766761c7fe5188da53269346b009072553c7e8fb8480dfa29c87ca4a9b1',
    'sp': '414c274973c585894e545d7d9e70f6a6f6365d7085ae18bd51d6ffbc4bfa4c50',
}
POLICY = 'one_exec_valid_s_direct_var_same_op_shared_plan_owned_clones_original_cache_layout'


def _support(s):
    return (s.n, s.k, s.spatial, s.ids, s.degs, s.prefix_by_deg)


def _pair(p):
    return (id(p), _support(p.sup_out), id(p.pair_ia), id(p.pair_jb),
            id(p.seg_len), id(p.seg_offsets), p.tag)


def direct_plan(spec, code, elem):
    """Conservative metadata-only plan; differing bindings simply do not hit."""
    assert code.order == spec.k and len(code.instrs) == len(spec.instrs)
    by_dst = {s.dst: s for s in spec.instrs}
    assert len(by_dst) == len(spec.instrs)
    # Compiler allocates every original cache block contiguously. Check the
    # actual compiler lengths, including non-trig blocks, before copying any.
    cursor = 0
    for c in code.instrs:
        if c.cache_len:
            assert c.cache_base == cursor and c.cache_len > 0
            cursor += c.cache_len
    assert cursor == code.n_cache
    rows = []
    for s, c in zip(spec.instrs, code.instrs):
        assert (s.op, s.dst, s.a, s.b, s.var, s.cache_base, s.strict_slot) == (
            c.op, c.dst, c.a, c.b, c.var, c.cache_base, c.strict_slot)
        if s.op not in ('sin', 'cos'):
            continue
        length = elem.cache_len(s.op, spec.k)
        assert c.cache_len == length and 0 <= s.cache_base <= code.n_cache - length
        leaf = by_dst[s.a]
        key = None
        if leaf.op == 'var':
            assert len(s.chain) == spec.k and len(s.acc) == spec.k
            key = (s.op, leaf.var, _support(leaf.sup_in), _support(leaf.sup_out),
                   leaf.keep_len, id(leaf.gather), spec.k, length,
                   _support(s.sup_f), _support(s.sup_out),
                   tuple((_pair(p), keep, _support(sup)) for p, keep, sup in s.chain),
                   tuple((_support(sup), id(a), id(b)) for sup, a, b in s.acc))
        rows.append((s.cache_base, s.op, length, key))
    return rows


def _bind(se):
    """Binding core also exercised by CPU checks; production uses install."""
    elem = se.elem
    raw_exec, raw_trig = se.exec_valid_s, elem._trig_series_valid_g
    active = ContextVar('direct_trig_reuse_call', default=None)
    counters = dict(exec_calls=0, completed_execs=0, failed_execs=0,
                    trig_calls=0, original_bodies=0, reuse_hits=0,
                    ineligible_calls=0, outside_exec_forwarded=0)
    last_plan = {}

    def trig(coeffs, rem, k, cache, base, tabs, make_ctx, cos_cycle):
        call = active.get()
        if call is None:
            counters['outside_exec_forwarded'] += 1
            return raw_trig(coeffs, rem, k, cache, base, tabs, make_ctx, cos_cycle)
        # Frozen elementary has no recursive _trig_series_valid_g calls;
        # frozen exec invokes it once per sin/cos in original tape order.
        assert call['at'] < len(call['rows']), 'Unexpected nested/extra trig call'
        expected_base, op, length, key = call['rows'][call['at']]
        assert (base, k, cos_cycle) == (expected_base, call['order'], op == 'cos')
        assert tabs is call['tabs'] and cache.shape[1] == call['n_cache']
        if call['cache'] is None:
            call['cache'] = cache
        assert cache is call['cache'], 'Unexpected foreign executor cache'
        call['at'] += 1
        counters['trig_calls'] += 1
        if key is not None and key in call['memo']:
            saved_c, saved_r, saved_cache = call['memo'][key]
            cache[:, base:base + length].copy_(saved_cache)
            counters['reuse_hits'] += 1
            return saved_c.clone(), saved_r.clone()
        counters['original_bodies'] += 1
        if key is None:
            counters['ineligible_calls'] += 1
        out_c, out_r = raw_trig(coeffs, rem, k, cache, base, tabs, make_ctx, cos_cycle)
        if key is not None:
            # Own snapshots even if a downstream consumer later mutates the
            # first result. Every reused output owns storage as well.
            call['memo'][key] = (out_c.clone(), out_r.clone(),
                                 cache[:, base:base + length].clone())
        return out_c, out_r

    def execute(spec, code, x, x_rem, eng, cutoff_threshold, tabs,
                bad_out=None, *, interval_coefficients=False):
        rows = direct_plan(spec, code, elem)
        keys = [key for _, _, _, key in rows if key is not None]
        last_plan.clear()
        last_plan.update(trig_calls=len(rows), eligible_direct_calls=len(keys),
                         unique_direct_plans=len(set(keys)),
                         potential_reuse_hits=len(keys)-len(set(keys)),
                         interval_coefficients=bool(interval_coefficients), order=spec.k)
        call = dict(rows=rows, at=0, memo={}, cache=None, order=spec.k,
                    tabs=tabs, n_cache=code.n_cache)
        token = active.set(call)
        counters['exec_calls'] += 1
        try:
            result = raw_exec(spec, code, x, x_rem, eng, cutoff_threshold, tabs,
                              bad_out=bad_out, interval_coefficients=interval_coefficients)
            assert call['at'] == len(rows), 'Missing original trig call'
            counters['completed_execs'] += 1
            return result
        except BaseException:
            counters['failed_execs'] += 1
            raise
        finally:
            active.reset(token)

    se.exec_valid_s, elem._trig_series_valid_g = execute, trig

    def restore():
        assert active.get() is None, 'Cannot restore during an active executor'
        assert se.exec_valid_s is execute and elem._trig_series_valid_g is trig
        se.exec_valid_s, elem._trig_series_valid_g = raw_exec, raw_trig

    return SimpleNamespace(policy=POLICY, counters=counters, last_plan=last_plan,
                           original_exec=raw_exec, original_trig=raw_trig,
                           candidate_exec=execute, candidate_trig=trig, restore=restore)


def install(torch, se):
    for name, module in [('se', se), ('elem', se.elem), ('sp', se.sp)]:
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == PINS[name]
    for module, name in [(se, 'exec_valid_s'), (se.elem, '_trig_series_valid_g')]:
        function = getattr(module, name)
        assert function.__qualname__ == name and function.__globals__ is module.__dict__
        assert Path(inspect.getfile(function)).resolve() == Path(module.__file__).resolve()
    result = _bind(se)
    result.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result.source_pins = dict(PINS)
    return result
