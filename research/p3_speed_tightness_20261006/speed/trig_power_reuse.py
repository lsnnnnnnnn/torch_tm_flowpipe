"""Call-local sin/cos power reuse over the existing same-op trig binding.

The original trig body still computes each derivative wheel, accumulation,
and strict Lagrange tail. Only SparseValidCtx.mul_valid calls of the opposite
trig operation may replay owned power/cache snapshots from the same input
variable and identical metadata bindings within this one executor invocation.
"""

from contextvars import ContextVar
import inspect
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


DIRECT_POLICY = "one_exec_valid_s_direct_var_same_op_shared_plan_owned_clones_original_cache_layout"


def install(torch, se, direct_binding):
    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError("production trig-power adapter requires PyTorch 2.5.1")
    if (not isinstance(se, ModuleType) or se.__name__ != "flowstar_gpu.sparse_exec"
            or sys.modules.get(se.__name__) is not se or not Path(se.__file__).is_file()):
        raise RuntimeError("expected the preloaded sparse executor module")
    elem = se.elem
    raw_exec, raw_trig = se.exec_valid_s, elem._trig_series_valid_g
    if (direct_binding.policy != DIRECT_POLICY or raw_exec is not direct_binding.candidate_exec
            or raw_trig is not direct_binding.candidate_trig):
        raise RuntimeError("install directly over the existing qualified same-op trig binding")
    original_trig = direct_binding.original_trig
    if (original_trig.__name__ != "_trig_series_valid_g"
            or original_trig.__globals__ is not elem.__dict__
            or inspect.getfile(original_trig) != str(Path(elem.__file__))):
        raise RuntimeError("same-op binding did not retain the original trig body")
    plan_builder = raw_exec.__globals__.get("direct_plan")
    if (not inspect.isfunction(plan_builder)
            or inspect.getfile(plan_builder) != inspect.getfile(raw_exec)):
        raise RuntimeError("same-op metadata plan is unavailable")
    ctx_type = se.SparseValidCtx
    raw_mul, raw_acc = ctx_type.mul_valid, ctx_type.acc_add
    for fn, name in ((raw_mul, "mul_valid"), (raw_acc, "acc_add")):
        if fn.__name__ != name or inspect.getfile(fn) != str(Path(se.__file__)):
            raise RuntimeError("requires original sparse trig context methods")
    active = ContextVar("trig_power_reuse_call", default=None)
    counters = dict(exec_calls=0, completed_execs=0, failed_execs=0, trig_calls=0,
                    saved_power_chains=0, reused_power_chains=0, computed_powers=0,
                    reused_powers=0, ineligible_calls=0, outside_exec_forwarded=0)
    last_plan = {}

    def trig(coeffs, rem, k, cache, base, tabs, make_ctx, cos_cycle):
        call = active.get()
        if call is None:
            counters["outside_exec_forwarded"] += 1
            return raw_trig(coeffs, rem, k, cache, base, tabs, make_ctx, cos_cycle)
        if call["at"] >= len(call["rows"]):
            raise RuntimeError("unexpected extra trig call")
        expected_base, op, length, original_key = call["rows"][call["at"]]
        call["at"] += 1
        counters["trig_calls"] += 1
        if (base, k, cos_cycle) != (expected_base, call["order"], op == "cos"):
            raise RuntimeError("trig call differs from the original metadata plan")
        if tabs is not call["tabs"] or cache.shape[1] != call["n_cache"]:
            raise RuntimeError("trig table/cache owner changed inside execution")
        if call["cache"] is None:
            call["cache"] = cache
        if cache is not call["cache"]:
            raise RuntimeError("foreign executor cache")
        if original_key is None:
            counters["ineligible_calls"] += 1
            return raw_trig(coeffs, rem, k, cache, base, tabs, make_ctx, cos_cycle)
        # Existing plan includes operation first; every other component,
        # including variable id, pair identities, supports and acc binds stays.
        key = original_key[1:]
        entry = call["memo"].get(key)
        save = entry is None
        if save:
            entry = dict(op=op, powers=[], complete=False)
            call["memo"][key] = entry
        reuse = not save and entry["op"] != op
        if reuse and (not entry["complete"] or len(entry["powers"]) != k):
            raise RuntimeError("opposite trig power chain is incomplete")
        contexts = []

        def make_power_ctx(tm_f, f_rem):
            ctx = make_ctx(tm_f, f_rem)
            if (contexts or type(ctx) is not ctx_type or ctx._i != 0 or ctx._j != 0
                    or ctx.eng is not call["eng"] or ctx.cutoff != call["cutoff"]
                    or ctx.preserve_intervals != call["interval_coefficients"]
                    or ctx_type.mul_valid is not raw_mul or ctx_type.acc_add is not raw_acc):
                raise RuntimeError("unexpected sparse trig context or method mutation")
            contexts.append(ctx)

            def mul_valid(a_c, a_r, target_cache, at):
                i = ctx._i
                if (not 0 <= i < k or target_cache is not cache
                        or at != base + 1 + 4 * i or at + 3 > base + length):
                    raise RuntimeError("trig multiplication/cache cursor changed")
                if reuse:
                    value_c, value_r, triple = entry["powers"][i]
                    target_cache[:, at:at + 3].copy_(triple)
                    ctx._i += 1  # The original method's only context mutation.
                    counters["reused_powers"] += 1
                    return value_c.clone(), value_r.clone()
                value_c, value_r = raw_mul(ctx, a_c, a_r, target_cache, at)
                counters["computed_powers"] += 1
                if save:
                    entry["powers"].append((value_c.clone(), value_r.clone(),
                                             target_cache[:, at:at + 3].clone()))
                return value_c, value_r

            # Forward the untouched accumulator, range and initialization.
            # No instance monkeypatch or ctx/self closure cycle is created.
            return SimpleNamespace(mul_valid=mul_valid, acc_add=ctx.acc_add,
                                   range_f=ctx.range_f, fresh_one=ctx.fresh_one,
                                   fresh_zero=ctx.fresh_zero)

        result = raw_trig(coeffs, rem, k, cache, base, tabs, make_power_ctx, cos_cycle)
        if contexts:
            if contexts[0]._i != k or contexts[0]._j != k:
                raise RuntimeError("original trig body did not finish all multiplications/accumulations")
            if save:
                if len(entry["powers"]) != k:
                    raise RuntimeError("new trig power snapshots are incomplete")
                entry["complete"] = True
                counters["saved_power_chains"] += 1
            elif reuse:
                counters["reused_power_chains"] += 1
        elif save:
            raise RuntimeError("existing same-op cache skipped an unseen power key")
        return result

    def execute(spec, code, x, x_rem, eng, cutoff_threshold, tabs,
                bad_out=None, *, interval_coefficients=False):
        if active.get() is not None:
            raise RuntimeError("trig-power executor is not reentrant")
        rows = plan_builder(spec, code, elem)
        groups = {}
        for _, op, _, key in rows:
            if key is not None:
                groups.setdefault(key[1:], set()).add(op)
        last_plan.clear()
        last_plan.update(trig_calls=len(rows), unique_power_plans=len(groups),
                         opposite_op_pairs=sum(ops == {"sin", "cos"} for ops in groups.values()),
                         order=spec.k, interval_coefficients=bool(interval_coefficients))
        call = dict(rows=rows, at=0, memo={}, cache=None, order=spec.k, tabs=tabs,
                    n_cache=code.n_cache, eng=eng, cutoff=cutoff_threshold,
                    interval_coefficients=bool(interval_coefficients))
        token = active.set(call)
        counters["exec_calls"] += 1
        try:
            output = raw_exec(spec, code, x, x_rem, eng, cutoff_threshold, tabs,
                              bad_out=bad_out, interval_coefficients=interval_coefficients)
            if call["at"] != len(rows):
                raise RuntimeError("missing original trig call")
            counters["completed_execs"] += 1
            return output
        except BaseException:
            counters["failed_execs"] += 1
            raise
        finally:
            active.reset(token)

    def restore():
        if active.get() is not None or se.exec_valid_s is not execute or elem._trig_series_valid_g is not trig:
            raise RuntimeError("trig-power binding no longer owns the original slots")
        se.exec_valid_s, elem._trig_series_valid_g = raw_exec, raw_trig

    receipt = dict(policy="one_exec_opposite_trig_owned_power_chain_same_direct_variable_metadata",
                   source_path=str(Path(__file__)), sparse_exec_path=str(Path(se.__file__)),
                   elementary_path=str(Path(elem.__file__)), inherited_direct_reuse_path=inspect.getfile(raw_exec),
                   original_accumulation_and_tail=True, original_cache_layout=True,
                   cross_executor_or_round_reuse=False, same_op_reuse_retained=True,
                   counters_scope="Python eager/warmup/capture construction, not CUDA graph replay counts",
                   qualification="new candidate; full tensor/cache/tail and saved-output gates required",
                   no_jit=True, no_digest_operations=True)
    se.exec_valid_s, elem._trig_series_valid_g = execute, trig
    return SimpleNamespace(receipt=receipt, counters=counters, last_plan=last_plan,
                           original_exec=raw_exec, candidate_exec=execute, restore=restore)
