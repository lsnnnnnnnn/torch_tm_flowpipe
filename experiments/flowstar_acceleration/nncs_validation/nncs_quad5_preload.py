from pathlib import Path
import sys,time,json,torch,traceback,hashlib,subprocess,ast,os
N=Path('/srv/local/shengenli/flowstar_acceleration_20260921T153643Z');E=N/'engine_endpoint_graph';sys.path.insert(0,str(E/'src'))
expected='5d1bac094639cd42e428e65744654fad275b4c2e';actual=subprocess.check_output(['git','-C',str(E),'rev-parse','HEAD'],text=True).strip();assert actual==expected
assert not subprocess.check_output(['git','-C',str(E),'status','--short'],text=True)
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
from torch.utils import cpp_extension
original=cpp_extension.load_inline
def logged(*a,**kw):
 try:return original(*a,**kw)
 except Exception:traceback.print_exc();raise
cpp_extension.load_inline=logged
from flowstar_gpu import cuda_kernels as ck,tape_kernels as tk,sr_kernels as sk,sr_sum_kernels as sj
r={'engine_sha':actual,'torch':torch.__version__,'cuda_compiler_home':os.environ.get('CUDA_HOME'),'cache':os.environ.get('TORCH_EXTENSIONS_DIR'),'extensions':{}}
for name,mod,fn,attr in [('interval',ck,ck.available,'_ext'),('tape',tk,tk.available,'_ext'),('validation',tk,tk.valid_available,'_vext'),('sr_phi',sk,sk.available,'_ext'),('sr_j_sum',sj,sj.available,'_ext')]:
 t=time.perf_counter();ok=fn();torch.cuda.synchronize();ext=getattr(mod,attr)
 row={'available':ok,'wall_s':time.perf_counter()-t,'module_source':mod.__file__,'module_source_sha256':hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),'source_string_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in vars(mod).items() if isinstance(v,str) and len(v)>100 and ('CUDA' in k or 'CPP' in k)}}
 if ext is not None:row.update(binary=ext.__file__,binary_sha256=hashlib.sha256(Path(ext.__file__).read_bytes()).hexdigest())
 r['extensions'][name]=row;print(name,row,flush=True)
def strings(p):
 out={}
 for node in ast.parse(p.read_text()).body:
  if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
   for t in node.targets:
    if isinstance(t,ast.Name):out[t.id]=node.value.value
 return out
old=strings(N/'repo/experiments/flowstar_acceleration/sr_sum_kernels.py');new=strings(Path(sj.__file__))
r['prototype_to_integrated']={'cpp_source_identical':old['_CPP']==new['_CPP'],'cuda_source_identical':old['_CUDA']==new['_CUDA'],'prototype_sha256':hashlib.sha256((N/'repo/experiments/flowstar_acceleration/sr_sum_kernels.py').read_bytes()).hexdigest(),'integration_module_sha256':hashlib.sha256(Path(sj.__file__).read_bytes()).hexdigest(),'scope':'relative import and extension name adjusted; CUDA/CPP numeric source unchanged'}
assert all(x['available'] for x in r['extensions'].values()) and all(r['prototype_to_integrated'][x] for x in ['cpp_source_identical','cuda_source_identical'])
Path('preload.json').write_text(json.dumps(r,indent=2)+'\n')
