"""Monitor only a launched process group; limits are measured RSS/GPU usage."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--timeout',type=float,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
r=Path(a.output);r.mkdir(parents=True,exist_ok=True)
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
start=time.perf_counter();peak_rss=peak_gpu=0;why=None
with (r/'stdout.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=r)
 root=psutil.Process(proc.pid)
 try:
  while proc.poll() is None:
   try:procs=[root]+root.children(recursive=True)
   except psutil.NoSuchProcess:break
   pids={p.pid for p in procs};rss=0
   for item in procs:
    try:rss+=item.memory_info().rss
    except psutil.Error:pass
   peak_rss=max(peak_rss,rss)
   gpu=0
   try:
    report=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
    for line in report.splitlines():
     pid,mem=[s.strip() for s in line.split(',')]
     if int(pid) in pids and mem.isdigit():gpu+=int(mem)*2**20
   except (subprocess.SubprocessError,ValueError):pass
   peak_gpu=max(peak_gpu,gpu)
   if rss>11.5*2**30:why='rss_guard_11.5GiB'
   elif gpu>14*2**30:why='gpu_guard_14GiB'
   elif time.perf_counter()-start>a.timeout:why='timeout'
   if why:
    # The process is a new session leader owned by this supervisor.
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
    break
   time.sleep(.1)
  rc=proc.wait()
 finally:
  result={'argv':cmd,'pid':proc.pid,'returncode':proc.poll(),'status':why or ('completed' if proc.returncode==0 else 'error'),'timeout_s':a.timeout,'process_wall_s':time.perf_counter()-start,'peak_tree_rss_bytes':peak_rss,'peak_owned_gpu_bytes':peak_gpu,'rss_guard_bytes':11.5*2**30,'gpu_guard_bytes':14*2**30,'resource_variant':'user_requested_gpu14_rss_unchanged','poll_s':.1}
  (r/'process.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
