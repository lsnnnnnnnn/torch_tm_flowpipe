"""Small CPU-only pinned-a3fb SR storage/chunk tests; no QUAD/NN/CUDA."""
from pathlib import Path
from types import ModuleType
import argparse,copy,hashlib,importlib.util,json,os,shutil,sys,time,traceback

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def main(args):
    start=time.perf_counter();args.output.mkdir(parents=True,exist_ok=False)
    result={'status':'exception'};installed=None
    try:
        here=Path(__file__).parent;expected=json.loads(args.identity.read_text())['engine']['python_sha256']
        for rel,value in expected.items():assert sha(args.engine_root/rel)==value,rel
        env=load('host_check_env',here.parent/'quad_sparse_metadata_20260927/check_backend.py');os.environ.update(env.ENV)
        sys.path.insert(0,str(args.engine_root/'src'))
        import torch
        from flowstar_gpu import sparse_exec as se,interval as iv,symbolic_remainder as sym
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.set_num_interop_threads(1)
        h=load('host_candidate',here/'host_ledger.py');old=load('frozen_k20_reference',here/'periodic_sr.py')
        assert sha(here/'periodic_sr.py')=='17ac3a69f934fb6943df0f2dab8072d84cb327b2ce00bd58573bfdae64fcfe86'
        rawprop=sym.propagate;rawmake=sym.make_symbolic_remainder
        c=ModuleType('host_check_common');c.se=se;c.make_symbolic_remainder=rawmake
        c.fingerprint=lambda st,sr: None;c.save_state=lambda *args,**kwargs:None
        def same(a,b):return tuple(a.shape)==tuple(b.shape) and a.dtype==b.dtype and h.digest(a)==h.digest(b)
        def original(sr):
            return {key:None if getattr(sr,key) is None else getattr(sr,key)[:sr.qlen].clone() if key in ['phi_buf','phi_iv_buf','j_buf'] else getattr(sr,key).clone() for key in h.SR_FIELDS}|dict(qlen=sr.qlen,jlen=sr.jlen)
        def identical(x,y):
            return all(same(x[k],y[k]) if isinstance(x[k],torch.Tensor) else x[k]==y[k] for k in x)
        def values(step,b):
            p=torch.eye(16).expand(b,16,16).clone()
            for lane in range(b):
                p[lane,0,0]+=((step%3)+1)/128;p[lane,0,1]=((-1)**step)*(lane+1)/32
                p[lane,1,0]=(step%5+lane+1)/128;p[lane,1,1]-=(lane+1)/256
            f=iv.from_point(p);f[:,:2,:2,0]-=2.**-24;f[:,:2,:2,1]+=2.**-24
            J=torch.tensor([-2.**-24,2.**-23]).expand(b,16,2).clone();J[:,13:16]*=(step%4+1)
            return p,f,J
        def append(sr,J):sr.scalars_iv=iv.from_point(sr.scalars);sr.append_j(J)
        rows=[];negative=0;step_checks=0
        for enabled in [False,True]:
            for chunk in [1,2,16,32]:
                sr=rawmake(2,16,1000,'cpu');base=rawmake(2,16,1000,'cpu')
                installed=old.install(c,enabled=enabled);reference=c.make_symbolic_remainder(2,16,1000,'cpu')
                ledger=h.HostFactorLedger(2,lane_chunk=chunk,enabled=enabled)
                for step in range(1,41):
                    p,f,J=values(step,2)
                    if enabled and chunk==1 and step==20:
                        parent=h.signature(ledger.payload(sr));trial=copy.deepcopy(ledger);tsr=copy.deepcopy(sr)
                        original_chunk=h.balanced_chunk
                        def interrupted(*unused):raise RuntimeError('synthetic interruption after raw propagation')
                        h.balanced_chunk=interrupted
                        try:
                            try:trial.propagate(tsr,p,f,status_before=trial.current_status,raw_propagate=rawprop,sym=sym)
                            except RuntimeError:negative+=1
                            else:raise AssertionError('interruption not raised')
                        finally:h.balanced_chunk=original_chunk
                        assert trial.poisoned and h.signature(ledger.payload(sr))==parent
                        try:trial.save_components(args.output/'poisoned_components',tsr)
                        except AssertionError:negative+=1
                        else:raise AssertionError('poisoned checkpoint accepted')
                    out=ledger.propagate(sr,p,f,status_before=ledger.current_status,raw_propagate=rawprop,sym=sym)
                    ref=se.propagate(reference,p,strict=True,phi_i_iv=f)
                    raw=rawprop(base,p,strict=True,phi_i_iv=f)
                    if step==20:
                        try:ledger.save_checkpoint(args.output/'pending.pt',sr)
                        except AssertionError:negative+=1
                        else:raise AssertionError('pending save accepted')
                    append(sr,J);append(reference,J);append(base,J)
                    ledger.finish_step(sr,torch.zeros(2,dtype=torch.int8),token=step)
                    assert same(out,ref) and identical(original(sr),original(reference))
                    assert all(same(getattr(sr,k),getattr(base,k)) for k in ['scalars','scalars_iv'])
                    assert same(sr.phi_buf[:step],base.phi_buf[:step]) and same(sr.j_buf[:step],base.j_buf[:step])
                    if not enabled or step<20:assert same(out,raw) and identical(original(sr),original(base))
                    assert same(ledger.factors[step-1],f)
                    if step==7:
                        recorded=h.digest(ledger.factors[:step]);f.fill_(123.)
                        assert h.digest(ledger.factors[:step])==recorded
                    step_checks+=1
                    if step==20:
                        path=args.output/f'normal_{enabled}_{chunk}_20.pt';ledger.save_checkpoint(path,sr)
                        saved20=h.signature(ledger.payload(sr))
                        restored,other=h.restore_checkpoint(path,rawmake,'cpu')
                        assert h.signature(other.payload(restored))==h.signature(ledger.payload(sr))
                        assert other.factors.data_ptr()!=ledger.factors.data_ptr()
                        component_path=args.output/f'normal_{enabled}_{chunk}_components'
                        ledger.save_components(component_path,sr)
                        streamed,stream_ledger=h.restore_components(component_path,rawmake,'cpu')
                        assert h.signature(stream_ledger.payload(streamed))==h.signature(ledger.payload(sr))
                        left=copy.deepcopy(sr);lh=copy.deepcopy(ledger);right=restored;rh=other
                        p21,f21,J21=values(21,2)
                        lo=lh.propagate(left,p21,f21,status_before=lh.current_status,raw_propagate=rawprop,sym=sym)
                        ro=rh.propagate(right,p21,f21,status_before=rh.current_status,raw_propagate=rawprop,sym=sym)
                        so=stream_ledger.propagate(streamed,p21,f21,status_before=stream_ledger.current_status,raw_propagate=rawprop,sym=sym)
                        append(left,J21);append(right,J21)
                        append(streamed,J21);stream_ledger.finish_step(streamed,stream_ledger.current_status,token=21)
                        lh.finish_step(left,lh.current_status,token=21);rh.finish_step(right,rh.current_status,token=21)
                        assert same(lo,ro) and h.signature(lh.payload(left))==h.signature(rh.payload(right))
                        assert same(lo,so) and h.signature(lh.payload(left))==h.signature(stream_ledger.payload(streamed))
                        assert h.signature(ledger.payload(sr))==saved20
                        assert ledger.length==20 and lh.length==21 and ledger.factors.data_ptr()!=lh.factors.data_ptr()
                rows.append(dict(batch=2,lane_chunk=chunk,enabled=enabled,steps=40,
                    frozen_K20_all_original_SR_and_image_bytes_equal=True,original_point_J_scalars_bytes_equal=True,
                    own_cold20_to21_all_SR_and_ledger_bytes_equal=True,owned_input_and_deepcopy_storage=True,
                    streaming_components_cold21_all_bytes_equal=True))
                installed.uninstall();installed=None
        # Mixed original statuses and bad input stay per lane. Lane0 alone is
        # eligible and must match a separately propagated all-finite B2 lane0.
        for chunk in [1,2,3]:
            b=5;sr=rawmake(b,16,1000,'cpu');base=rawmake(b,16,1000,'cpu')
            status=torch.tensor([0,1,0,0,0],dtype=torch.int8)
            ledger=h.HostFactorLedger(b,lane_chunk=chunk,status=status)
            installed=old.install(c,enabled=True);reference=c.make_symbolic_remainder(2,16,1000,'cpu')
            for step in range(1,41):
                p,f,J=values(step,b);pg,fg,Jg=values(step,2)
                f[1,0,0]=float('nan');p[1,0,0]=float('nan')
                if step==5:f[2,0,0]=float('nan')
                if step==7:f[3,0,0]=torch.tensor([1.,-1.])
                if step in [10,11]:p[4,0,0]=1e200;f[4,0,0]=1e200
                out=ledger.propagate(sr,p,f,status_before=status,raw_propagate=rawprop,sym=sym)
                raw=rawprop(base,p,strict=True,phi_i_iv=f);ref=se.propagate(reference,pg,strict=True,phi_i_iv=fg)
                append(sr,J);append(base,J);append(reference,Jg)
                after=status.clone()
                if step==5:after[2]=3
                if step==7:after[3]=3
                if step==20:after[4]=3
                ledger.finish_step(sr,after,token=step);status=after
                assert same(out[0],ref[0]) and same(sr.phi_iv_buf[:step,0],reference.phi_iv_buf[:step,0])
                assert same(out[1:],raw[1:]) and same(sr.phi_iv_buf[:step,1:],base.phi_iv_buf[:step,1:])
                assert same(sr.phi_buf[:step],base.phi_buf[:step]) and same(sr.j_buf[:step],base.j_buf[:step])
                if step in [20,40]:
                    assert ledger.replaced[step-1].tolist()==[True,False,False,False,False]
                    assert ledger.fallback[step-1].tolist()==([False,False,False,False,True] if step==20 else [False]*5)
                    assert ledger.reset_if_full(sr) is False
                step_checks+=1
            path=args.output/f'mixed_{chunk}_40.pt';ledger.save_checkpoint(path,sr)
            restored,other=h.restore_checkpoint(path,rawmake,'cpu')
            assert h.signature(ledger.payload(sr))==h.signature(other.payload(restored))
            component_path=args.output/f'mixed_{chunk}_components';ledger.save_components(component_path,sr)
            streamed,stream_ledger=h.restore_components(component_path,rawmake,'cpu')
            assert h.signature(ledger.payload(sr))==h.signature(stream_ledger.payload(streamed))
            bad=status.clone();bad[1]=0
            try:other.propagate(restored,*values(41,b)[:2],status_before=bad,raw_propagate=rawprop,sym=sym)
            except AssertionError:negative+=1
            else:raise AssertionError('inactive revival accepted')
            rows.append(dict(batch=b,lane_chunk=chunk,steps=40,mixed_status_and_nonfinite_raw_lanes_bytes_preserved=True,
                lane0_frozen_K20_bytes_equal=True,one_lane_overflow_fallback=True,mixed_checkpoint_all_bytes_equal=True,
                streaming_mixed_components_bytes_equal=True))
            installed.uninstall();installed=None
        # Original1000 reset/transport only: synthetic buffers, no claim to have
        # executed a1000-step reachability or matrix multiplication trajectory.
        ledger=h.HostFactorLedger(2);sr=rawmake(2,16,1000,'cpu');sr.reserve(1000)
        p,f,J=values(1,2);sr.phi_buf.copy_(p);sr.phi_iv_buf=f.expand(1000,*f.shape).clone();sr.j_buf.copy_(J)
        sr.scalars_iv=iv.from_point(sr.scalars);sr.qlen=sr.jlen=ledger.length=1000
        ledger.factors.copy_(f);ledger.point_finite.fill_(True);ledger.factor_valid.fill_(True);ledger.status_before.zero_()
        ledger.eligible[19::20]=True;ledger.replaced[19::20]=True;ledger.last_rebuilt_lane.fill_(1000)
        ledger.save_checkpoint(args.output/'synthetic_full1000.pt',sr)
        assert ledger.reset_if_full(sr) and ledger.epoch==1000 and ledger.length==sr.qlen==sr.jlen==0
        ledger.save_checkpoint(args.output/'reset1000.pt',sr);restored,other=h.restore_checkpoint(args.output/'reset1000.pt',rawmake,'cpu')
        assert h.signature(ledger.payload(sr))==h.signature(other.payload(restored))
        ledger.save_components(args.output/'reset_components',sr)
        streamed,stream_ledger=h.restore_components(args.output/'reset_components',rawmake,'cpu')
        assert h.signature(ledger.payload(sr))==h.signature(stream_ledger.payload(streamed))
        p,f,J=values(1001,2)
        x=ledger.propagate(sr,p,f,status_before=ledger.current_status,raw_propagate=rawprop,sym=sym)
        y=other.propagate(restored,p,f,status_before=other.current_status,raw_propagate=rawprop,sym=sym)
        append(sr,J);append(restored,J);ledger.finish_step(sr,ledger.current_status,token=1001);other.finish_step(restored,other.current_status,token=1001)
        assert same(x,y) and h.signature(ledger.payload(sr))==h.signature(other.payload(restored))
        saved=ledger.payload(sr)
        for key in ['factors','sr_phi_buf','sr_phi_iv_buf','sr_j_buf','sr_scalars','sr_scalars_iv','status_before','factor_valid']:
            bad=copy.deepcopy(saved);bad.pop(key)
            try:h.validate_payload(bad)
            except (AssertionError,KeyError):negative+=1
            else:raise AssertionError('missing history accepted')
        for key in ['factor_valid','eligible','replaced','fallback','last_rebuilt_lane']:
            bad=copy.deepcopy(saved);bad[key].fill_(True if bad[key].dtype==torch.bool else 999)
            if h.signature(bad)==h.signature(saved):bad[key].zero_()
            try:h.validate_payload(bad)
            except AssertionError:negative+=1
            else:raise AssertionError('corrupt ledger masks accepted')
        for key in ['sr_scalars_iv','sr_phi_iv_buf','sr_j_buf']:
            bad=copy.deepcopy(saved);bad[key].reshape(-1,2)[0]=torch.tensor([1.,-1.])
            try:h.validate_payload(bad)
            except AssertionError:negative+=1
            else:raise AssertionError('active reversed interval accepted')
        reset=torch.load(args.output/'reset1000.pt',weights_only=True)
        for key,value in [('sr_scalars',0.),('sr_scalars_iv',0.),('sr_phi_iv_buf',None)]:
            bad=copy.deepcopy(reset)
            if value is None:bad[key]=None
            else:bad[key].fill_(value)
            try:h.validate_payload(bad)
            except AssertionError:negative+=1
            else:raise AssertionError('bad reset accepted')
        ledger.save_components(args.output/'current_components',sr)
        for source,key in [('current_components','sr_phi_iv_buf'),('reset_components','sr_scalars')]:
            corrupted=args.output/('corrupt_'+key);shutil.copytree(args.output/source,corrupted)
            manifest=json.loads((corrupted/'MANIFEST.json').read_text());entry=manifest['tensors'][key]
            assert len(entry['blocks'])==1
            record=entry['blocks'][0];file=corrupted/record['file'];value=torch.load(file,weights_only=True)
            if key=='sr_scalars':value.zero_()
            else:value.reshape(-1,2)[0]=torch.tensor([1.,-1.])
            torch.save(value,file);record.update(sha256=sha(file),size=file.stat().st_size);entry['raw_sha256']=h.digest(value)
            (corrupted/'MANIFEST.json').write_text(json.dumps(manifest)+'\n')
            try:h.restore_components(corrupted,rawmake,'cpu')
            except AssertionError:negative+=1
            else:raise AssertionError('hash-updated invalid component checkpoint accepted')
        result=dict(status='passed',device='cpu',script_sha256=sha(__file__),adapter_sha256=sha(here/'host_ledger.py'),
            frozen_K20_sha256=sha(here/'periodic_sr.py'),engine_python_sha256=expected,rows=rows,
            step_comparisons=step_checks,negative_cases_rejected=negative,original1000_reset_and_nextstep_cold_bytes_equal=True,
            streamed_original_SR_and_host_ledger_roundtrips_passed=True,stream_block_bytes=h.BLOCK_BYTES,
            poisoned_owned_trial_does_not_mutate_parent=True,hash_updated_invalid_component_checkpoints_rejected=True,
            scope='Pinned actual CPU SR arithmetic, small B2/B5 only; mixed nonfinite/status and isolated overflow fallback; full SR/host-ledger checkpoint. No plant/support/cap checkpoint, QUAD/NN/CUDA/fullB1024/resource gate, or new soundness claim for inactive/bad lanes.')
    except BaseException as exc:result.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());raise
    finally:
        if installed is not None:installed.uninstall()
        result['process_s']=time.perf_counter()-start
        (args.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['engine-root','identity','output']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
