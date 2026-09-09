"""The same native sparse inputs for scalar, grouped CPU and CUDA measurements."""
from collections import defaultdict
from dataclasses import dataclass
import torch
from torch_tm_flowpipe import Interval,Polynomial
from torch_tm_flowpipe.packed_boundary_range import evaluate_polynomial,evaluate_interval_coefficients
from torch_tm_flowpipe.range_requests import RangeRequest
from .common import RUN,read,read_records,request_from_record,digest


@dataclass
class SparseInput:
    record: dict
    polynomial: object
    domain: list

    def request(self):
        r=self.record["request"]
        exponents=tuple(tuple(e) for e in r["exponents"])
        if r["kind"]=="interval-coefficient":
            cl=torch.stack([self.polynomial[e].lo for e in exponents]) if exponents else torch.empty(0,dtype=torch.float64)
            ch=torch.stack([self.polynomial[e].hi for e in exponents]) if exponents else torch.empty(0,dtype=torch.float64)
        else:
            cl=torch.stack(list(self.polynomial.terms.values())) if exponents else torch.empty(0,dtype=torch.float64)
            ch=cl
        dl=torch.stack([d.lo for d in self.domain]) if self.domain else torch.empty(0,dtype=torch.float64)
        dh=torch.stack([d.hi for d in self.domain]) if self.domain else torch.empty(0,dtype=torch.float64)
        assert r["step_powers"] is None
        return RangeRequest(r["request_id"],exponents,cl[None],ch[None],dl,dh,r["kind"],tuple(r["state_variables"]),r["time_variable"])

    def scalar(self):
        r=self.record["request"]
        if r["kind"]=="interval-coefficient":
            result=evaluate_interval_coefficients(self.polynomial,self.domain,reference=self.domain[0])
        else:
            result=evaluate_polynomial(self.polynomial,self.domain,normal=r["kind"]=="normal",
                state_variables=tuple(r["state_variables"]),time_variable=r["time_variable"])
        if not result.is_finite():raise FloatingPointError("scalar baseline not finite")
        return result


def sparse(record):
    r=request_from_record(record["request"])
    assert r.coefficients_lo.shape[0]==1
    if r.kind=="interval-coefficient":
        poly={e:Interval(r.coefficients_lo[0,i],r.coefficients_hi[0,i]) for i,e in enumerate(r.exponents)}
    else:
        poly=Polynomial({},r.domain_lo.numel())
        poly.terms.update({e:r.coefficients_lo[0,i].clone() for i,e in enumerate(r.exponents)})
    return SparseInput(record,poly,[Interval(a,b) for a,b in zip(r.domain_lo,r.domain_hi)])


def stages(records,plant,batch):
    subset=set(read(RUN/"PARTITION_PLAN.json")["subsets"][str(batch)])
    grouped=defaultdict(list)
    for record in records:
        if record["plant"]==plant and record["task"] in subset:
            grouped[record["step"],record["ordinal"]].append(record)
    return [(key,[sparse(r) for r in sorted(rows,key=lambda r:r["task"])]) for key,rows in sorted(grouped.items())]


def request_identity(workload):
    return digest([x.record["request_numeric_sha256"] for _,rows in workload for x in rows])
