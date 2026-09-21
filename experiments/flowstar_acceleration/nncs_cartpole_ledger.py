"""Observe the original CartPole failure; change no numerical settings."""
import json,sys,runpy,dataclasses
from pathlib import Path
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');sys.path.insert(0,str(N/'repo/experiments/flowstar_acceleration'));import nncs_screen
original=nncs_screen.load_driver;out=Path(sys.argv[sys.argv.index('--output')+1]);step=0

def load(engine):
 d=original(engine);advance=d.advance_sparse
 def observed(*args,**kw):
  global step
  step+=1
  if step>=155:
   def record(event):
    with (out/'refinement_events.jsonl').open('a') as f:f.write(json.dumps({'ode_step':step,**event})+'\n')
   args=list(args);args[4]=dataclasses.replace(args[4],refinement_callback=record)
  return advance(*args,**kw)
 d.advance_sparse=observed;return d
nncs_screen.load_driver=load
runpy.run_path(str(N/'repo/experiments/flowstar_acceleration/nncs_candidate.py'),run_name='__main__')
