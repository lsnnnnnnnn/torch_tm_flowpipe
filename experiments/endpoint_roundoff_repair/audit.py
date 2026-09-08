"""Independent exact-rational audit of saved repaired endpoints and clocks."""
import argparse,csv,gzip,json,math
from fractions import Fraction as F
from pathlib import Path
from experiments.xiangru_adoption.common import f,power,product,measure
from torch_tm_flowpipe import load_terminal_checkpoint,accepted_boundary_sr_queue_sha256
from torch_tm_flowpipe import Interval,Polynomial,TaylorModel,TMVector
from torch_tm_flowpipe.batched_dense_tm import sparse_tmvector_to_dense
from torch_tm_flowpipe.brusselator_canonical_exchange import _append_tmv
from experiments.xiangru_adoption.common import from_existing_canonical
from experiments.endpoint_roundoff_repair.local_oracles import exact_coefficients,exact_error_range


def decode(model,order):
    domain=[Interval(*map(float.fromhex,d)) for d in model['domain']]
    return TMVector(TaylorModel(Polynomial({tuple(t['degrees']):float.fromhex(t['coefficient'][0])
                    for t in c['terms']},len(domain)),Interval(*map(float.fromhex,c['remainder'])),domain,order=order)
                    for c in model['components'])


def encode(tm,name):
    values=[]
    _append_tmv(values,name,tm,variable_order=('ux','uy','tau') if tm.n_vars==3 else ('ux','uy'))
    return from_existing_canonical(dict(values),name)


def require(value,message):
    if not value: raise ValueError(message)


def prove_run(root,*,partial=False):
    rows=list(csv.DictReader((root/'bounds.csv').open()))
    audits=[json.loads(line) for line in (root/'endpoint_audit.jsonl').read_text().splitlines()]
    source=json.loads((root/'source.json').read_text())
    contract=json.loads((root/'execution_contract.json').read_text())
    require(source['data_origin']=='FRESH_ENDPOINT_REPAIRED','fresh run role')
    require((source['device'],source['dtype'],source['threads'],source['interop_threads'])==('cpu','float64',1,1),'CPU binary64 single-thread identity')
    require(len(rows)==len(audits),'bounds/audit rows')
    total=F()
    pointwise=0
    coefficients_checked=0
    replayed=0
    capacity=contract['config']['accepted_boundary_sr_capacity']
    maxima=[(0.,None),(0.,None)]
    first=[None,None]
    checkpoint_rows=[]
    for i,(row,audit) in enumerate(zip(rows,audits),1):
        require(int(row['step'])==audit['step']==i,'step sequence')
        h=float.fromhex(row['h_hex'])
        require(math.isfinite(h) and h>0,'finite positive actual h')
        if not contract['adaptive']: require(h.hex()==contract['fixed_h_hex'],'fixed binary64 h changed')
        require(row['t_start_exact']==str(total),'exact interval start')
        total+=F(h)
        require(row['t_end_exact']==audit['t_end_exact']==str(total),'exact binary64 accumulated time')
        require(float(row['t_end'])==float(total),'display time is not the recorded actual time')
        require(int(row['queue_size'])==audit['queue_size']==i%capacity,'queue capacity transition')
        require(audit['queue_reset_count']==i//capacity,'queue reset count')
        if i%capacity: require(audit['owner_boundary']==i,'current owner generation')
        for component,bounds in enumerate(audit['substitution_error_hex']):
            values=list(map(float.fromhex,bounds))
            require(all(map(math.isfinite,values)) and values[0]<=values[1],'finite ordered endpoint correction')
            magnitude=max(map(abs,values))
            if magnitude and first[component] is None:first[component]=i
            if magnitude>maxima[component][0]:maxima[component]=(magnitude,i)
        checkpoint=root/f'checkpoint_{i:04d}'
        if checkpoint.exists():
            restored=load_terminal_checkpoint(checkpoint,expected_contract=contract,
                       expected_order=contract['config']['order'],expected_dtype='float64')
            require(restored.scheduler['time_exact']==str(total),'checkpoint actual clock')
            require(restored.normal_state.step_index==i,'checkpoint accepted boundary')
            require(accepted_boundary_sr_queue_sha256(restored.normal_state.symbolic_queue)==audit['queue_sha256'],'saved queue versus consumed queue')
            if i%capacity:
                require([[float(v.lo).hex(),float(v.hi).hex()] for v in restored.normal_state.symbolic_queue.J[-1]]==audit['new_owner_hex'],
                        'saved current owner versus actual consumed owner')
            checkpoint_rows.append({'step':i,'queue_size':i%capacity,'reset_count':i//capacity,
                                    'queue_sha256':audit['queue_sha256'],'path':checkpoint.name})
    if not partial:
      with gzip.open(root/'models.jsonl.gz','rt') as stream:
        for row,audit,line in zip(rows,audits,stream,strict=True):
          record=json.loads(line)
          require(record['step']==int(row['step']) and record['t_end_exact']==row['t_end_exact'],'model identity/time')
          tube=record['models']['tube'];endpoint=record['models']['endpoint']
          h=F(float.fromhex(row['h_hex']))
          source_tm=decode(tube,contract['config']['order'])
          replay,errors=source_tm.substitute_const_with_roundoff(source_tm.n_vars-1,float(h))
          for j,(source_model,point,error) in enumerate(zip(source_tm,replay,errors)):
            _,_,enclosures=source_model.polynomial.substitute_const_with_roundoff(source_model.n_vars-1,float(h),source_model.domain)
            exact=exact_coefficients(source_model.polynomial,source_model.n_vars-1,float(h))
            for beta,value in exact.items():
                bound=enclosures[beta+(0,)];require(F(float(bound.lo))<=value<=F(float(bound.hi)),'coefficient enclosure')
                coefficients_checked+=1
            elo,ehi=exact_error_range(exact,point.polynomial.drop_variable(point.n_vars-1),source_model.domain[:-1])
            saved_lo,saved_hi=map(lambda v:F(float.fromhex(v)),audit['substitution_error_hex'][j])
            require(saved_lo<=elo<=ehi<=saved_hi,'saved endpoint correction loses exact coefficient error')
          correction=[[float(e.lo).hex(),float(e.hi).hex()] for e in errors]
          require(correction==audit['substitution_error_hex'],'published endpoint correction differs from production replay')
          dense=sparse_tmvector_to_dense(source_tm,order=contract['config']['order'])
          _,dlo,dhi,_,_=dense.poly.substitute_const_and_drop_with_roundoff(source_tm.n_vars-1,float(h),dense.domain_lo,dense.domain_hi)
          require([[float(dlo[0,j]).hex(),float(dhi[0,j]).hex()] for j in range(len(source_tm))]==audit['internal_substitution_error_hex'],
                  'internal endpoint correction differs from production replay')
          replay=replay.drop_variable(replay.n_vars-1).apply_cutoff(contract['config']['cutoff'])
          require(encode(replay,'endpoint')==endpoint,'saved published endpoint differs from repaired CPU binary64 replay')
          replayed+=1
          for old,new in zip(tube['components'],endpoint['components'],strict=True):
            exact={}
            for term in old['terms']:
                exp=tuple(term['degrees']);beta=exp[:-1]
                exact[beta]=exact.get(beta,F())+f(term['coefficient'][0])*h**exp[-1]
            point={tuple(t['degrees']):f(t['coefficient'][0]) for t in new['terms']}
            lo,hi=map(f,old['remainder'])
            for exp in exact.keys()|point.keys():
                delta=exact.get(exp,F())-point.get(exp,F());interval=(delta,delta)
                for e,domain in zip(exp,endpoint['domain']):interval=product(interval,power(tuple(map(f,domain)),e))
                lo+=interval[0];hi+=interval[1]
            require(f(new['remainder'][0])<=lo and f(new['remainder'][1])>=hi,
                    f"full endpoint model loses exact substitution + old R at step {row['step']}")
            pointwise+=1
          for name,model in record['models'].items():
            for dim,bounds in zip(['x','y'],measure(model)):
                require(float(row[f'common_{name}_{dim}_lo'])==bounds[0] and float(row[f'common_{name}_{dim}_hi'])==bounds[1],
                        'independent common complete-object measurement')
            for dim,bounds in zip(['x','y'],decode(model,contract['config']['order']).range_box()):
                require(float(row[f'published_{name}_{dim}_lo'])==float(bounds.lo) and float(row[f'published_{name}_{dim}_hi'])==float(bounds.hi),
                        'published complete-object measurement')
    summary_path=root/'summary.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        require(summary['accepted_steps']==len(rows),'summary accepted count')
        require(summary['accepted_horizon_exact']==str(total) and summary['accepted_horizon']==float(total),'summary actual completion time')
        if not contract['adaptive']:
            require(summary['completed']==(len(rows)==contract['requested_steps'] and summary['failure'] is None),'summary completion status')
        else:
            scheduler=0.
            next_h=.1
            for row in rows:
                remaining=summary['requested_horizon']-scheduler
                attempted=min(next_h,.1,remaining)
                if 0.<remaining-attempted<.002:attempted=remaining
                for _ in range(audits[int(row['step'])-1]['step_rejections']):attempted*=.5
                require(attempted.hex()==row['h_hex'],'native adaptive scheduling policy changed')
                require(.002<=attempted<=.1,'native adaptive legal h')
                next_h=min(attempted*1.1,.1)
                scheduler+=float.fromhex(row['h_hex'])
                require(scheduler.hex()==row['scheduler_time_hex'],'native float scheduler clock')
            require(summary['scheduler_time_hex']==scheduler.hex(),'terminal scheduler clock')
            require(summary['completed']==(scheduler>=summary['requested_horizon']-1e-12 and summary['failure'] is None),'adaptive completion status')
        require(summary['rejected_attempts']==sum(a['step_rejections'] for a in audits),'rejected attempts')
        for key in audits[0]['refinement_counters']:
            require(summary['refinement_totals'][key]==sum(a['refinement_counters'][key] for a in audits),'refinement count')
        require(summary['scientific_sha']==source['scientific_sha'],'summary scientific identity')
    else:require(partial,'terminal run summary missing')
    return {'run':root.name,'scientific_sha':source['scientific_sha'],'steps':len(rows),
            'exact_horizon':str(total),'full_function_containment_checks':pointwise,
            'exact_coefficient_containment_checks':coefficients_checked,'production_endpoint_replays':replayed,
            'first_nonzero_substitution_error_step':first,'max_abs_substitution_error_by_component':maxima,
            'checkpoint_checks':checkpoint_rows,'partial_read':partial,'passed':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    result=prove_run(args.run,partial=args.partial)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2);handle.write('\n')
    print(json.dumps(result,indent=2))
