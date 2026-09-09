from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch_tm_flowpipe import Interval
from torch_tm_flowpipe.range_requests import RangeRequest, RangeResult

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/"artifacts/runs/range_batch_device_20260909T030609Z"
PARENT=ROOT/"artifacts/runs/boundary_execution_20260908T172756Z"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def read(path): return json.loads(Path(path).read_text())
def head(): return subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
def numerical_sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/"src/torch_tm_flowpipe").glob("*")) if p.suffix in {".py",".cu"}}
def hex_tensor(t): return {"shape":list(t.shape),"values":[float(x).hex() for x in t.detach().cpu().flatten().tolist()]}
def tensor(record):
    return torch.tensor([float.fromhex(x) for x in record["values"]],dtype=torch.float64).reshape(record["shape"])
def request_record(r):
    return dict(request_id=r.request_id,exponents=r.exponents,coefficients_lo=hex_tensor(r.coefficients_lo),
        coefficients_hi=hex_tensor(r.coefficients_hi),domain_lo=hex_tensor(r.domain_lo),domain_hi=hex_tensor(r.domain_hi),
        kind=r.kind,state_variables=r.state_variables,time_variable=r.time_variable,
        step_powers={str(p):[float(iv.lo).hex(),float(iv.hi).hex()] for p,iv in r.step_powers.items()} if r.step_powers is not None else None,
        enabled=r.enabled,cancelled=r.cancelled,dtype="float64",device="cpu")
def request_from_record(r):
    assert r["dtype"]=="float64" and r["device"]=="cpu"
    return RangeRequest(r["request_id"],tuple(tuple(e) for e in r["exponents"]),
        *(tensor(r[f]) for f in ("coefficients_lo","coefficients_hi","domain_lo","domain_hi")),r["kind"],
        tuple(r["state_variables"]),r["time_variable"],
        {int(p):Interval(*(float.fromhex(x) for x in v)) for p,v in r["step_powers"].items()} if r["step_powers"] is not None else None,
        r["enabled"],r["cancelled"])
def result_record(r):
    return dict(request_id=r.request_id,status=r.status,lo=hex_tensor(r.lo) if r.lo is not None else None,
        hi=hex_tensor(r.hi) if r.hi is not None else None,
        terms_lo=hex_tensor(r.terms_lo) if r.terms_lo is not None else None,
        terms_hi=hex_tensor(r.terms_hi) if r.terms_hi is not None else None,
        powers=[[v,p,a.hex(),b.hex(),fix] for v,p,a,b,fix in r.powers])
def result_from_record(r):
    return RangeResult(r["request_id"],r["status"],tensor(r["lo"]) if r["lo"] else None,
        tensor(r["hi"]) if r["hi"] else None,terms_lo=tensor(r["terms_lo"]) if r["terms_lo"] else None,
        terms_hi=tensor(r["terms_hi"]) if r["terms_hi"] else None,
        powers=tuple((v,p,float.fromhex(a),float.fromhex(b),fix) for v,p,a,b,fix in r["powers"]))
def read_records(path):
    with gzip.open(path,"rt") as f:
        for line in f:
            assert len(line)<2_000_000
            yield json.loads(line)

