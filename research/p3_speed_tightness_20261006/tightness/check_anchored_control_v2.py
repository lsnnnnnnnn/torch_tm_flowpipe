#!/usr/bin/env python3
"""New CPU algebra/rounding check, not a historical numerical checker."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def prohibited(*args, **kwargs):
    raise RuntimeError("content digest prohibited")


for name in (*hashlib.algorithms_guaranteed, "new", "file_digest"):
    if hasattr(hashlib, name):
        setattr(hashlib, name, prohibited)
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/gpu_verified_20260930/source/engine/src"))
import torch
from flowstar_gpu import interval as iv, support as sp, cuda_kernels as ck, monomials, polynomial
sys.path.insert(0, str(Path(__file__).with_name("v2")))
from anchored_control import contract

ck.available = prohibited  # Any accidental CUDA extension discovery fails.
iv.USE_KERNELS = False
torch.set_default_dtype(torch.float64)
T = lambda value: torch.tensor(value, dtype=torch.float64)
tables = monomials.build_tables(2,2)
sup = sp.make_support(2,2,False,(0,1,2))
eng = sp.SparseEngine(tables,polynomial.build_step_tables(tables,.01),"cpu")
assert not hasattr(eng,"exponents")  # The exact missing API in failed GPU v1.


def run(rows, poly, old, low_a, up_a, low_b, up_b, hull=None):
    rows, poly = T([rows]), T([poly])
    rem = torch.zeros(1, rows.shape[1], 2)
    values = dict(input_rows=rows, input_rem=rem, control_rows=poly, control_rem=T([old]),
        lower_A=T([low_a]), upper_A=T([up_a]), lower_b=T([low_b]), upper_b=T([up_b]),
        certified_hull=T([hull]) if hull is not None else iv.add(sp.range_normal_s(rows,eng,sup),rem))
    before = {k:v.clone() for k,v in values.items()}
    result, receipt = contract(torch,iv,sp,eng=eng,sup=sup,**values)
    assert all(torch.equal(v,before[k]) for k,v in values.items())
    assert result.untyped_storage().data_ptr() != values["control_rem"].untyped_storage().data_ptr()
    assert bool((result[...,0] >= values["control_rem"][...,0]).all())
    assert bool((result[...,1] <= values["control_rem"][...,1]).all())
    return result, receipt


def main():
    checks = []
    # Exact correlated linear states x1=x2=r: the NN x1-x2 is identically zero.
    result, receipt = run([[0,1,0],[0,1,0]], [[0,0,0]], [[-.25,.25]],
                          [[1,-1]], [[1,-1]], [0], [0])
    assert abs(float(result[0,0,0])) < 1e-12 and abs(float(result[0,0,1])) < 1e-12
    assert receipt["contracted_rows"] == 1
    checks.append("correlated state cancellation contracts the unchanged zero polynomial")
    # ReLU envelopes on [-2,2]: 0 <= relu(x) <= .5*x+1. Exact extrema
    # of relu(x)-(.25+.5*x) on represented [-1,1] occur at -1,0,1.
    result, _ = run([[0,1,0]], [[.25,.5,0]], [[-1,1]], [[0]], [[.5]], [0], [1], [[-2,2]])
    for x in (F(-1),F(0),F(1)):
        residual = max(x,F(0))-(F(1,4)+F(1,2)*x)
        assert F(float(result[0,0,0])) <= residual <= F(float(result[0,0,1]))
    checks.append("two-slope ReLU exact breakpoints contained with unchanged anchor")
    result, receipt = run([[0,1,0]], [[0,0,0]], [[-.25,.25]], [[0]], [[0]], [-10], [10])
    assert torch.equal(result,T([[[-.25,.25]]])) and receipt["contracted_rows"] == 0
    checks.append("uninformative envelope leaves original remainder unchanged")
    # Large constant plus .75 is not representable as that same RN coefficient.
    result, _ = run([[2**53,1,0]], [[2**53,1,0]], [[-4,4]], [[1]], [[1]], [.75], [.75])
    assert F(float(result[0,0,0])) <= F(3,4) <= F(float(result[0,0,1]))
    checks.append("coefficient addition/subtraction roundoff is retained")
    failures = [
        ([[0,1,0]], [[0,0,0]], [[-.25,.25]], [[0]], [[0]], [1], [1], None),
        ([[0,1,0]], [[0,0,0]], [[-.25,.25]], [[0]], [[0]], [0], [0], [[-.1,.1]]),
        ([[0,1,0]], [[0,0,0]], [[-.25,.25]], [[float('nan')]], [[0]], [0], [0], None),
    ]
    for args in failures:
        try:
            run(*args)
        except (ValueError,FloatingPointError):
            continue
        raise AssertionError("invalid/disjoint/domain input was accepted")
    checks.append("disjoint, insufficient-domain and nonfinite cases rejected")
    checks.append("all successful calls preserve input tensors and own the returned subset")
    checks.extend(check_hook())
    checks.append("all helper and hook cases use real frozen SparseEngine, which has no exponents method")
    time_id = int(torch.nonzero((tables.exponents[:,0] == 1) & (tables.exponents.sum(-1) == 1)).item())
    time_sup = sp.make_support(2,2,False,(0,1,time_id))
    try:
        contract(torch,iv,sp,eng=eng,sup=time_sup,input_rows=torch.zeros(1,1,3),
            input_rem=torch.zeros(1,1,2),control_rows=torch.zeros(1,1,3),control_rem=T([[[-1,1]]]),
            lower_A=torch.zeros(1,1,1),upper_A=torch.zeros(1,1,1),
            lower_b=torch.zeros(1,1),upper_b=torch.zeros(1,1),certified_hull=T([[[-1,1]]]))
    except ValueError as error:
        assert "time-free" in str(error)
    else:
        raise AssertionError("time-dependent FULL support accepted")
    checks.append("real time-dependent FULL support is rejected before contraction")
    receipt=dict(status="PASS",checks=checks,torch_version=torch.__version__,device="cpu",
        frozen_interval_backend="research/gpu_verified_20260930/source/engine/src/flowstar_gpu/interval.py",
        exact_reference="Fraction arithmetic for affine cancellation, ReLU breakpoints and a large-constant rounding case",
        implementation_revision="v2_real_support_exponent_lookup",real_sparse_engine=True,
        gpu_tested=False,real_nn_tested=False,new_solver_runs=0,old_checkers_executed=0,digest_operations=0,
        independent_end_to_end_floating_nncs_certificate=False)
    Path(__file__).with_name("CPU_CHECK_V2.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


def check_hook():
    """Real interval/strict injection plus mock NN calls; no real NN or ODE."""
    import importlib.util
    from quad_controller_hook import install
    path = ROOT / "research/gpu_verified_20260930/source/adapters/quad_control_transfer_20260927/strict_injection.py"
    spec = importlib.util.spec_from_file_location("strict_injection_for_new_algebra_check", path)
    injection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(injection)
    tables = monomials.build_tables(16,3)
    terms = sp.make_support(16,3,False,(0,1,2))
    engine = sp.SparseEngine(tables,polynomial.build_step_tables(tables,.005),"cpu")
    assert not hasattr(engine,"exponents")
    state = SimpleNamespace(pre=torch.zeros(1,16,3), pre_rem=torch.zeros(1,16,2),
        pre_sup=terms, status=torch.zeros(1,dtype=torch.int8),
        tmv=torch.zeros(1,16,3),tmv_rem=torch.zeros(1,16,2))
    state.pre[:,0,1] = state.pre[:,1,1] = 1  # x1=x2=r, NN=x1-x2=0.
    matrix = torch.zeros(1,3,12)
    matrix[:,:,0], matrix[:,:,1] = 1, -1
    calls = []
    def build(config, device, relax="same-slope", input_layout="native"):
        calls.append(relax)
        return object()
    def baseline(model, config, lower, upper, **kw):
        return matrix.clone(), torch.full((1,3),-.25), torch.full((1,3),.25)
    def extra(model, config, lower, upper, **kw):
        return matrix.clone(),matrix.clone(),torch.zeros(1,3),torch.zeros(1,3)
    def strict_inject(st, A, L, U, u_ids, nn_in):
        assert st is state
        injection.inject_controls_s(st,A,L,U,u_ids,nn_in,eng=engine)
    driver = SimpleNamespace(build_crown=build,crown_bounds=baseline,
        crown_bounds_two_slope=extra,inject_controls_s=strict_inject,
        initial_sparse_state=lambda cells,eng,sched:state)
    originals = {name:getattr(driver,name) for name in
                 ("build_crown","crown_bounds","inject_controls_s","initial_sparse_state")}
    previous = sys.argv
    sys.argv = ["new_check", "config", "--strict", "--engine", "sparse", "--crown-domain", "box",
        "--crown-relax", "same-slope", "--crown-transport", "native-f64",
        "--crown-input-layout", "native", "--nn-mode", "crown", "--order", "3"]
    try:
        hook = install(torch,driver,iv,sp)
        config = dict(num_vars=16,num_nn_input=12,num_nn_output=3,output_scale=1)
        model = driver.build_crown(config,"cpu")
        assert calls == ["same-slope","two-slope"]
        driver.initial_sparse_state(None,engine,None)
        hull = iv.add(sp.range_normal_s(state.pre[:,:12],engine,terms),state.pre_rem[:,:12])
        A,L,U = driver.crown_bounds(model,config,hull[...,0],hull[...,1],input_layout="native")
        unchanged = state.pre[:,:13].clone()
        try:
            driver.inject_controls_s(state,A.clone(),L,U,(13,14,15),12)
        except ValueError:
            pass
        else:
            raise AssertionError("altered pending certificate accepted")
        driver.inject_controls_s(state,A,L,U,(13,14,15),12)
        assert torch.equal(state.pre[:,:13],unchanged)
        assert torch.equal(state.pre[:,13:16],torch.zeros(1,3,3))
        assert float(state.pre_rem[:,13:16].abs().max()) < 1e-12
        assert hook.counters["base_nn_calls"] == hook.counters["extra_nn_calls"] == 1
        assert hook.counters["total_nn_calls"] == 2 and hook.counters["contracted_rows"] == 3
        assert hook.records[0]["total_rows"] == 3
        try:
            driver.inject_controls_s(state,A,L,U,(13,14,15),12)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate certificate consumption accepted")
        hook.restore()
        assert all(getattr(driver,k) is v for k,v in originals.items())
        try:
            hook.restore()
        except RuntimeError:
            pass
        else:
            raise AssertionError("double restore accepted")
        sys.argv[sys.argv.index("native-f64")] = "rpc-float32"
        try:
            install(torch,driver,iv,sp)
        except ValueError:
            pass
        else:
            raise AssertionError("unqualified transport accepted")
    finally:
        sys.argv = previous
    return ["hook uses real strict injection, distinct NN modules and the unchanged polynomial",
            "hook rejects altered/duplicate certificates and RPC transport; counts extra NN and restores owned hooks"]


if __name__ == "__main__":
    main()
