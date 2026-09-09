"""Serial diagnostic survey of real sparse calls outside the B1 adapter."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import torch

from experiments.range_batch_device.corpus import capture_ranges
from experiments.range_batch_device.common import save, sha
from .runner import run_case


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summaries=[]
    for plant in ("van_der_pol","brusselator"):
        # The inherited wrapper observes one serial worker only. No concurrent
        # global patching, no recorded input is used by this actual solve.
        observed=[]
        with capture_ranges(observed,dict(case=f"live-coverage-{plant}",step=0)):
            run,events,_=run_case(plant,[0],2,"S",diagnostic=True,run_id=f"coverage-{plant}")
        submitted=[event for event in events if event["event"]=="submit"]
        nonempty=[event for event in observed if event["request"]["exponents"]]
        empty=[event for event in observed if not event["request"]["exponents"]]
        # setup and the identical common observer also call the sparse methods.
        # Match the exact live coefficients/domains, preserving multiplicities.
        def key(request):
            request=dict(request)
            request.pop("request_id")
            return json.dumps(request,sort_keys=True,separators=(",",":"))
        dispatched=Counter(key(event["request"]) for event in submitted)
        actual=Counter(key(event["request"]) for event in observed)
        assert not (dispatched-actual), "adapter request did not originate in an actual sparse call"
        remaining=actual-dispatched
        unmatched=[]
        for event in observed:
            identity=key(event["request"])
            if remaining[identity]:
                unmatched.append(event)
                remaining[identity]-=1
        summary=dict(plant=plant,real_steps=2,strict_adapter_requests=len(submitted),
            observed_sparse_calls=len(observed),observed_empty_support=len(empty),
            uncovered_or_observer_calls=len(unmatched),
            scope="serial diagnostic including initialization/common observer; unmatched records retain call stack",
            all_ranges_on_gpu=False,dense_range_math="remains CPU and outside this sparse-method survey")
        payload=dict(summary=summary,observed=observed,submitted=submitted,unmatched=unmatched)
        filename=args.output/f"{plant}.json.gz"
        with gzip.open(filename,"wt") as stream:
            json.dump(payload,stream,separators=(",",":"),allow_nan=False)
        summaries.append(dict(**summary,raw_file=filename.name,sha256=sha(filename)))
    save(args.output/"coverage_summary.json",summaries)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=="__main__":
    main()
