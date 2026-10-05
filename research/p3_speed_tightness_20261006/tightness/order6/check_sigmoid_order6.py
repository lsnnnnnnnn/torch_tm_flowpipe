#!/usr/bin/env python3
"""New staging and order6 strict-injection algebra check; no ODE/NN run."""
import ast
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def prohibited(*a, **kw):
    raise RuntimeError("digest or extension discovery prohibited")


for name in (*hashlib.algorithms_guaranteed,"new","file_digest"):
    if hasattr(hashlib,name):
        setattr(hashlib,name,prohibited)

from run_sigmoid_order6_candidate import EDITS, revised
ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "docs/evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_u11_fourway_campaign_20261003_002/later05_ours_p3/payload"


def main():
    for name in EDITS:
        before = (SOURCE/name).read_text()
        after = revised(name,before)
        assert after != before
        try:
            revised(name,after)
        except RuntimeError:
            pass
        else:
            raise AssertionError("double numerical-source patch accepted")
    assert '"--strict", "--order", "6",' in revised("run_full.py",(SOURCE/"run_full.py").read_text())
    # Extract only the real existing function definition. Never execute a
    # saved campaign, runtime preparation, or historical self-check.
    source = SOURCE/"archcomp26_dp_p3_nohash.py"
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == "strict_injection")
    namespace = {"json":json}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),"exec"),namespace)
    sys.path.insert(0,str(ROOT/"research/gpu_verified_20260930/source/engine/src"))
    import torch
    torch.jit.script = lambda function=None,**kw: function if function is not None else lambda value:value
    from flowstar_gpu import interval as iv, support as sp, cuda_kernels as ck, monomials, polynomial
    ck.available = prohibited
    iv.USE_KERNELS = False
    torch.set_default_dtype(torch.float64)
    tables = monomials.build_tables(6,6)
    # Verify validation7 basis compatibility using the real integer enumerator;
    # no complete order7 multiplication table or ODE solve is needed here.
    higher = monomials._enumerate_exponents(7,14)
    assert torch.equal(tables.exponents,torch.from_numpy(higher[:tables.T2]))
    ids = []
    for degree in (0,1,6):
        exponent = torch.zeros(7,dtype=torch.int64)
        exponent[1] = degree
        ids.append(int(torch.nonzero((tables.exponents == exponent).all(-1)).item()))
    sup = sp.make_support(6,6,False,tuple(ids))
    engine = sp.SparseEngine(tables,polynomial.build_step_tables(tables,.01),"cpu")
    state = SimpleNamespace(pre=torch.zeros(1,6,3),pre_rem=torch.zeros(1,6,2),pre_sup=sup,
        status=torch.zeros(1,dtype=torch.int8),tmv=torch.zeros(1,6,3),tmv_rem=torch.zeros(1,6,2))
    state.pre[0,:4] = torch.tensor([[.1,.5,1e-8],[.2,-.3,0],[.3,.2,-2e-8],[.4,-.1,0]])
    original = state.pre.clone()
    matrix = torch.tensor([[[.1,-.25,.5,1.0]]])
    lower,upper = torch.tensor([[-.001]]),torch.tensor([[.002]])
    namespace["strict_injection"](state,matrix,lower,upper,[5],4,engine)
    assert torch.equal(original[:,:5],state.pre[:,:5])
    assert torch.isfinite(state.pre_rem).all() and (state.pre_rem[...,0] <= state.pre_rem[...,1]).all()
    for r in (F(-1),F(-1,2),F(0),F(1,2),F(1)):
        basis = (F(1),r,r**6)
        inputs = [sum(F(float(c))*monomial for c,monomial in zip(row,basis)) for row in original[0,:4]]
        point = sum(F(float(c))*monomial for c,monomial in zip(state.pre[0,5],basis))
        affine = sum(F(float(a))*x for a,x in zip(matrix[0,0],inputs))
        for bias in (lower.item(),upper.item()):
            remainder = affine+F(bias)-point
            assert F(float(state.pre_rem[0,5,0])) <= remainder <= F(float(state.pre_rem[0,5,1]))
    degree7 = torch.zeros(7,dtype=torch.int64)
    degree7[1] = 7
    id7 = int(torch.nonzero((tables.exponents == degree7).all(-1)).item())
    state.pre_sup = sp.make_support(6,6,False,(ids[0],ids[1],id7))
    before_pre,before_rem = state.pre.clone(),state.pre_rem.clone()
    try:
        namespace["strict_injection"](state,matrix,lower,upper,[5],4,engine)
    except ValueError as error:
        assert "approved time-free FULL layout" in str(error)
        pass
    else:
        raise AssertionError("degree7 support accepted by order6 injection")
    assert torch.equal(state.pre,before_pre) and torch.equal(state.pre_rem,before_rem)
    receipt = dict(status="PASS",torch_version=torch.__version__,device="cpu",
        checks=["exact declared staged edits parse and reject repeated patching",
                "real order6 support with a degree6 term accepted by unchanged strict injection",
                "exact-rational affine residual values enclosed at five support points and both biases",
                "physical rows unchanged; order6 extended monomial basis is a prefix of order7",
                "degree7 support rejected by unchanged strict injection before state mutation"],
        strict_injection_source=str(source),working_order=6,point_order=5,validation_order=7,
        real_nn_tested=False,gpu_tested=False,new_solver_runs=0,old_checkers_executed=0,digest_operations=0,
        independent_end_to_end_floating_nncs_certificate=False)
    Path(__file__).with_name("SIGMOID_ORDER6_CPU_CHECK.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
