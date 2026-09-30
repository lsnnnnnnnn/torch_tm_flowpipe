"""Explicitly authorized build of two private candidate extensions only."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, time
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main(a):
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();r=dict(status='exception');h=None
    try:
        assert sha(a.common)=='1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985'
        sys.path.insert(0,str(a.engine_root/'src'));common=load('reciprocal_build_common',a.common)
        assert common.E.resolve()==a.engine_root.resolve();common.bootstrap();torch=common.torch
        os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
        inst=load('reciprocal_build_installer',Path(__file__).with_name('install.py'))
        h=inst.install(a.cache,allow_build=True);inst.emit(h,a.output/'sources');builds=[]
        for name,fn in [('replay',h.tape.available),('valid',h.tape.valid_available)]:
            t=time.perf_counter();assert fn();torch.cuda.synchronize();builds.append(dict(name=name,wall_s=time.perf_counter()-t))
        assert len(h.loads)==2 and all(Path(v['path']).is_relative_to(a.cache.resolve()) for v in h.loads.values())
        r=dict(status='passed',candidate=h.metadata,extensions=h.loads,build=builds,
            baseline_extensions=common.ORIGINAL['extensions'],common_sha256=sha(a.common),engine_root=str(a.engine_root.resolve()),
            torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),device='cuda',
            environment={k:os.environ.get(k) for k in ['CXX','CC','CUDA_HOME','TORCH_CUDA_ARCH_LIST','MAX_JOBS','CUDA_VISIBLE_DEVICES','PATH']},
            scope='Build/load preflight only, not a numerical or dispatch qualification.')
    except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        if h is not None:h.restore()
        r.update(script_sha256=sha(__file__),process_body_s=time.perf_counter()-start)
        (a.output/'RESULT.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(r),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['engine-root','common','cache','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
