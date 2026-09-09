"""Execute an exact argv with source/environment/exit/JUnit receipts."""
import argparse
import os
import subprocess
import sys
import time
from .common import ROOT,RUN,save,head,numerical_sources


def main():
    p=argparse.ArgumentParser();p.add_argument("name");p.add_argument("argv",nargs=argparse.REMAINDER);a=p.parse_args()
    argv=a.argv[1:] if a.argv and a.argv[0]=="--" else a.argv
    output=RUN/"tests";output.mkdir(exist_ok=True)
    record=dict(name=a.name,argv=argv,cwd=str(ROOT),source_sha=head(),numerical_sources=numerical_sources(),
        environment={k:os.environ.get(k) for k in ("PYTHONPATH","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","CUDA_VISIBLE_DEVICES","DIFFREACH_ROOT","JAX_PLATFORMS")},
        started_utc=time.time(),affinity=sorted(os.sched_getaffinity(0)),
        log=f"tests/{a.name}.log",exit=f"tests/{a.name}.exit",xml=f"tests/{a.name}.xml")
    with (output/f"{a.name}.log").open("w") as log:
        child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        record.update(pid=child.pid,proc_start_ticks=open(f"/proc/{child.pid}/stat").read().split()[21])
        save(output/f"{a.name}.command.json",record)
        code=child.wait()
    record.update(exit_code=code,finished_utc=time.time())
    (output/f"{a.name}.exit").write_text(str(code)+"\n")
    save(output/f"{a.name}.command.json",record)
    print(record,flush=True);sys.exit(code)


if __name__=="__main__":main()
