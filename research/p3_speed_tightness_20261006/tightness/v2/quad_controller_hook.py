"""Extra affine envelopes for the unchanged strict same-slope QUAD control TM.

Install inside the wrapped driver.main, after the canonical runner installed
its strict injection and NN counter. Never install during prepare itself.
No frozen function, model file, ODE setting, state polynomial or SR is changed.
"""
import sys
import time
from types import SimpleNamespace

from anchored_control import contract


def install(torch, driver, iv, sp):
    required = {"--engine": "sparse", "--crown-domain": "box",
                "--crown-relax": "same-slope", "--crown-transport": "native-f64",
                "--crown-input-layout": "native", "--nn-mode": "crown", "--order": "3"}
    argv = sys.argv[1:]
    for flag, value in required.items():
        if argv.count(flag) != 1 or argv[argv.index(flag)+1:argv.index(flag)+2] != [value]:
            raise ValueError("anchored QUAD requires the original strict native-f64 box route")
    if "--strict" not in argv or any(v.startswith("--crown-alpha") for v in argv):
        raise ValueError("strict original CROWN route required")
    names = ("build_crown", "crown_bounds", "inject_controls_s", "initial_sparse_state")
    original = {name: getattr(driver, name) for name in names}
    if original["inject_controls_s"].__name__ != "strict_inject":
        raise ValueError("install after the canonical strict injection wrapper")
    extra_bounds = driver.crown_bounds_two_slope
    state = dict(engine=None, base_model=None, extra_model=None, config=None, pending=None)
    counters = dict(base_nn_calls=0, extra_nn_calls=0, total_nn_calls=0,
                    injected_refreshes=0, contracted_rows=0, total_control_rows=0,
                    extra_model_build_wall_s=0.0, extra_crown_wall_s=0.0,
                    contraction_wall_s=0.0)
    records = []
    receipt = dict(implementation_revision="v2_real_support_exponent_lookup",
                   policy="same-slope polynomial with independent two-slope affine remainder intersection",
                   scope="paper QUAD, strict box/native-f64, order 3",
                   baseline_nn_counter="canonical nn_calls counts original same-slope calls only",
                   actual_nn_counter="base_nn_calls + extra_nn_calls count attempted calls, including a call that raises; completed attempts require every pair to reach injection",
                   timing="extra model construction precedes driver timed loop; extra CROWN and contraction are inside it; component wall timers are unsynchronized host timings, not device profiles; outer wall includes all",
                   certificate_assumption="both floating affine enclosures concern the same unmodified controller over the exact recorded hull",
                   end_to_end_strict_certificate=False, gpu_qualified=False)

    def build(config, device, relax="same-slope", input_layout="native"):
        if (state["base_model"] is not None or relax != "same-slope" or input_layout != "native"
                or (config["num_vars"], config["num_nn_input"], config["num_nn_output"]) != (16, 12, 3)
                or float(config["output_scale"]) <= 0):
            raise ValueError("unexpected QUAD model build")
        base = original["build_crown"](config, device, relax=relax, input_layout=input_layout)
        start = time.perf_counter()
        extra = original["build_crown"](config, device, relax="two-slope", input_layout=input_layout)
        counters["extra_model_build_wall_s"] += time.perf_counter()-start
        if extra is base:
            raise RuntimeError("independent CROWN module required")
        state.update(base_model=base, extra_model=extra, config=config)
        return base

    def initial(cells, eng, sched):
        if state["engine"] is not None or (eng.tables.n, eng.tables.k) != (16, 3):
            raise ValueError("one paper QUAD P3 engine required")
        result = original["initial_sparse_state"](cells, eng, sched)
        state["engine"] = eng
        return result

    def bounds(model, config, lower, upper, **kw):
        if (state["pending"] is not None or model is not state["base_model"]
                or config is not state["config"] or kw != {"input_layout": "native"}
                or lower.ndim != 2 or lower.shape[1] != 12 or upper.shape != lower.shape):
            raise ValueError("controller certificate sequence or domain mismatch")
        counters["base_nn_calls"] += 1
        counters["total_nn_calls"] += 1
        values = original["crown_bounds"](model, config, lower, upper, **kw)
        start = time.perf_counter()
        counters["extra_nn_calls"] += 1
        counters["total_nn_calls"] += 1
        affine = extra_bounds(state["extra_model"], config, lower, upper, **kw)
        counters["extra_crown_wall_s"] += time.perf_counter()-start
        if len(affine) != 4:
            raise ValueError("four affine envelope tensors required")
        state["pending"] = (values, tuple(v.detach().clone() for v in affine),
                            torch.stack((lower, upper), -1).detach().clone())
        return values

    def inject(st, T, L, U, u_ids, nn_in):
        pending = state["pending"]
        if (pending is None or state["engine"] is None or tuple(u_ids) != (13, 14, 15)
                or nn_in != 12 or any(a is not b for a, b in zip((T, L, U), pending[0]))):
            raise ValueError("strict injection must consume its exact pending native-f64 certificate")
        # The canonical wrapper checks holder/state ownership and performs the
        # original strict arithmetic first. A later failure aborts this attempt;
        # no candidate remainder is written until contract has completed.
        result = original["inject_controls_s"](st, T, L, U, u_ids, nn_in)
        before = st.pre_rem[:, 13:16]
        lA, uA, lb, ub = pending[1]
        start = time.perf_counter()
        after, check = contract(torch, iv, sp, eng=state["engine"], sup=st.pre_sup,
            input_rows=st.pre[:, :12], input_rem=st.pre_rem[:, :12],
            control_rows=st.pre[:, 13:16], control_rem=before,
            lower_A=lA, upper_A=uA, lower_b=lb, upper_b=ub, certified_hull=pending[2])
        old_width = before[..., 1]-before[..., 0]
        new_width = after[..., 1]-after[..., 0]
        records.append(dict(refresh=len(records)+1, contracted_rows=check["contracted_rows"],
                            total_rows=check["total_rows"],
                            old_remainder_width_mean=old_width.mean(0).detach().cpu().tolist(),
                            new_remainder_width_mean=new_width.mean(0).detach().cpu().tolist(),
                            old_remainder_width_max=old_width.max(0).values.detach().cpu().tolist(),
                            new_remainder_width_max=new_width.max(0).values.detach().cpu().tolist()))
        st.pre_rem[:, 13:16] = after
        counters["contraction_wall_s"] += time.perf_counter()-start
        counters["injected_refreshes"] += 1
        counters["contracted_rows"] += check["contracted_rows"]
        counters["total_control_rows"] += check["total_rows"]
        state["pending"] = None
        return result

    installed = dict(build_crown=build, crown_bounds=bounds,
                     inject_controls_s=inject, initial_sparse_state=initial)
    for name, value in installed.items():
        setattr(driver, name, value)

    def restore():
        if any(getattr(driver, name) is not value for name, value in installed.items()):
            raise RuntimeError("anchored controller hook ownership changed")
        for name, value in original.items():
            setattr(driver, name, value)

    return SimpleNamespace(receipt=receipt, counters=counters, records=records, restore=restore)
