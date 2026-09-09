"""Raw CPU/CUDA term and power outputs, exact oracle checks and grouping counts."""
from collections import Counter,defaultdict
import csv
import gzip
import json
import time

import torch
from torch_tm_flowpipe.range_requests import evaluate_range_requests,prepare_range_requests,structure_key
from torch_tm_flowpipe.range_cuda import get_module
from torch_tm_flowpipe.packed_boundary_range import packed_boundary_execution
from torch_tm_flowpipe.prepared_remainder_replay import prepared_remainder_replay
from .common import RUN,save,read,read_records,request_from_record,result_record,digest
from .oracle import check


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    records=list(read_records(RUN/"raw/corpus/independent.jsonl.gz"))+list(read_records(RUN/"raw/corpus/offline.jsonl.gz"))
    by_stage=defaultdict(list)
    for record in records:
        # Offline calls may only be batched under an explicit replay label.
        key=(record["scope"],record["plant"],record["step"],record["ordinal"])
        by_stage[key].append(record)
    stats=[];totals={};numeric={};invocations={}
    output=RUN/"raw/operator_outputs";output.mkdir(parents=True,exist_ok=False)
    device_events=[]
    for backend in ("cpu","cuda"):
        counts=Counter();width_differences=[];calls=0
        with gzip.open(output/f"{backend}.jsonl.gz","wt") as f:
            for key,rows in by_stage.items():
                requests=[request_from_record(r["request"]) for r in rows]
                timing={}
                import torch_tm_flowpipe.range_cuda as cuda
                original=cuda.run_resident
                def observed(resident):
                    result=original(resident)
                    receipt=result[-1].cpu().tolist()
                    device_events.append(dict(stage=list(key),receipt=receipt,rows=resident.coefficients_lo.shape[0]))
                    return result
                if backend=="cuda":cuda.run_resident=observed
                try:
                    with prepared_remainder_replay(True),packed_boundary_execution(True):
                        results=evaluate_range_requests(requests,backend=backend,diagnostics=True,timings=timing)
                finally:
                    cuda.run_resident=original
                calls+=timing.get("actual_kernel_invocations",0)
                if backend=="cpu":
                    groups,_,_=prepare_range_requests(requests)
                    for g in groups:
                        stats.append(dict(scope=key[0],plant=key[1],step=key[2],ordinal=key[3],
                            structure_sha256=digest(g.key),group_size=len(g.requests),terms=len(g.plan.exponents),
                            request_ids="|".join(r.request_id for r in g.requests)))
                for record,r in zip(rows,requests):
                    result=results[r.request_id]
                    exact_counts=check(r,result)
                    counts.update(exact_counts);counts["requests"]+=1;counts[result.status]+=1
                    bounds=[float(result.lo[0]).hex(),float(result.hi[0]).hex()]
                    same=bounds==record["scalar_cpu_bounds"]
                    counts["legacy_bitwise_equal" if same else "legacy_bitwise_different"]+=1
                    if not same:
                        width_differences.append(dict(request_id=r.request_id,legacy=record["scalar_cpu_bounds"],actual=bounds))
                    f.write(json.dumps(dict(request_id=r.request_id,input_sha256=digest(record["request"]),
                        result=result_record(result),exact_counts=exact_counts,legacy_bitwise_equal=same),separators=(",",":"))+"\n")
                if counts["requests"]%1000 < len(rows):print(backend,counts["requests"],flush=True)
        totals[backend]=dict(counts)
        save(output/f"{backend}_width_differences.json",width_differences)
        invocations[backend]=calls
    with (RUN/"grouping_statistics.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    save(RUN/"cpu_batch_equivalence.json",dict(**totals["cpu"],all_requests_finite=True,
        correction_policy="exact endpoint certification; original scalar path unchanged"))
    save(RUN/"cuda_arithmetic_checks.json",dict(**totals["cuda"],all_requests_finite=True,
        arithmetic="directed binary64 add/multiply and finite integer powers",build=get_module().build))
    save(RUN/"cuda_actual_invocations.json",dict(audit_device_executed_kernel_invocations=invocations["cuda"],
        raw_device_events=device_events,
        per_group_receipt=[1,1,1,1],receipt_origin="four distinct device kernels each atomicAdd its own receipt cell",
        cpu_invocations=invocations["cpu"],mock=False))


if __name__=="__main__":main()
