"""One original QUAD root, optionally bisected in x5 at t=0; full live SR.

Bounded plant/controller diagnostic, not a qualified end-to-end NNCS proof.
No dynamic splitting, failed-leaf dropping, changed budget, or full-batch claim.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse, copy, hashlib, importlib.util, json, math, os, sys, time

BOXES_SHA = '8dd18155d7d63b311fece5732251758a4984c0abb428377c135e68c244cedd30'
ROOT_ID = 1
BATCH = 1024
SUBSTEPS = 20


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def outward(value, upper):
    answer = float(value)
    if (F(answer) < value) if upper else (F(answer) > value):
        answer = math.nextafter(answer, math.inf if upper else -math.inf)
    return answer


def root_leaves(boxes, mode):
    assert len(boxes) == BATCH and len(boxes[ROOT_ID]) == 16
    parent = copy.deepcopy(boxes[ROOT_ID])
    assert all(math.isfinite(x) for pair in parent for x in pair)
    assert all(lo <= hi for lo, hi in parent)
    if mode == 'baseline':
        return [parent], dict(axis=None, exact_parent=parent, leaves=[parent], covered=True)
    lo, hi = map(F, parent[4]); mid = (lo + hi) / 2
    children = [copy.deepcopy(parent), copy.deepcopy(parent)]
    children[0][4][1] = outward(mid, True)
    children[1][4][0] = outward(mid, False)
    assert lo < hi and F(children[0][4][1]) >= F(children[1][4][0])
    assert children[0][4][0] == parent[4][0] and children[1][4][1] == parent[4][1]
    assert all(child[i] == parent[i] for child in children for i in range(16) if i != 4)
    return children, dict(axis='original physical x5 at t=0', exact_parent=parent,
                          exact_midpoint=str(mid), leaves=children, covered=True)


def initial_coverage(st, boxes, eng):
    exps = [tuple(e) for e in eng.tables.exponents[list(st.pre_sup.ids)].cpu().tolist()]
    receipts = []
    assert not bool(st.pre_rem.any() or st.tmv_rem.any())
    for lane, box in enumerate(boxes):
        rows = []
        for i, (row, (lo, hi)) in enumerate(zip(st.pre[lane].cpu().tolist(), box)):
            unit = tuple(1 if j == i + 1 else 0 for j in range(17))
            nonzero = {e: F(v) for e, v in zip(exps, row) if v}
            assert set(nonzero) <= {tuple([0] * 17), unit}
            center = nonzero.get(tuple([0] * 17), F(0)); radius = nonzero.get(unit, F(0))
            assert radius >= 0 and center - radius <= F(lo) and center + radius >= F(hi)
            rows.append(dict(center=float(center), radius=float(radius),
                             low_excess=str(F(lo) - (center - radius)),
                             high_excess=str(center + radius - F(hi))))
        receipts.append(dict(leaf=lane, affine_rows=rows))
    return receipts


def byte_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and c.torch.equal(
        a.detach().cpu().contiguous().view(c.torch.uint8),
        b.detach().cpu().contiguous().view(c.torch.uint8))


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(c.torch.uint8).numpy().tobytes()).hexdigest()


def controller_refresh(st, eng, driver, model, cfg, completed, out):
    """One union-hull certificate applied symbolically to every required leaf."""
    torch = c.torch
    torch.cuda.synchronize(); started = time.perf_counter()
    leaf_hulls = driver.hull_ranges_s(st, eng, 12)
    assert bool(torch.isfinite(leaf_hulls).all())
    root_hull = torch.stack((leaf_hulls[..., 0].amin(0), leaf_hulls[..., 1].amax(0)), -1)
    assert bool(c.iv.contains(root_hull.unsqueeze(0), leaf_hulls).all())
    repeat = root_hull.unsqueeze(0).expand(BATCH, -1, -1).clone()
    zero = torch.zeros_like(repeat); zero[ROOT_ID] = root_hull
    assert not byte_equal(repeat, zero), 'padding schemes must actually differ'
    results = []
    for padded in [zero, repeat]:
        result = driver.crown_bounds(model, cfg, padded[..., 0].contiguous(),
                                     padded[..., 1].contiguous(), input_layout='native')
        result = driver.apply_crown_transport(*result, mode='native-f64')
        target = tuple(x[ROOT_ID:ROOT_ID + 1].detach().clone() for x in result)
        assert all(bool(torch.isfinite(x).all()) for x in target)
        results.append(target)
    equal = [byte_equal(a, b) for a, b in zip(*results)]
    payload = dict(completed_step=completed, next_step=completed + 1, root_id=ROOT_ID,
                   input_root_hull=root_hull.cpu(), input_leaf_hulls=leaf_hulls.cpu(),
                   zero_target=[x.cpu() for x in results[0]],
                   repeat_target=[x.cpu() for x in results[1]],
                   all_padding_outputs_byte_equal=all(equal), output_byte_equal=equal,
                   phase='before_injection', input_layout='native', transport='native-f64')
    path = out / ('controller_at_' + str(completed) + '.pt'); torch.save(payload, path)
    assert all(equal), 'B1024 row1 controller output depends on padding; entire root stopped'
    T, lower, upper = results[0]
    assert bool((lower <= upper).all())
    held = dict(path=str(path), sha256=sha(path), input_root_hull=root_hull.cpu().tolist(),
                T_sha256=tensor_sha(T), lower_sha256=tensor_sha(lower), upper_sha256=tensor_sha(upper),
                completed_step=completed, target_row=ROOT_ID, padding_schemes=['zero', 'repeat_root_hull'],
                output_byte_equal=equal, all_leaf_domains_inside_root_hull=True)
    # Detach only prevents autograd retaining NN graphs; numerical tensors are unchanged.
    with torch.no_grad():
        driver.inject_controls_s(st, T.expand(len(st.pre), -1, -1),
                                 lower.expand(len(st.pre), -1), upper.expand(len(st.pre), -1),
                                 [13, 14, 15], 12)
    torch.cuda.synchronize(); held['controller_and_injection_s'] = time.perf_counter() - started
    return held


def checkpoint(path, st, sr, eng, cap, step, held, metadata, phase):
    c.save_state(path, st, sr, eng, cap, step, phase)
    # All live numerical arrays are in the PT; these fields reconstruct scheduling and identity.
    write(path.with_suffix('.json'), dict(metadata=metadata, completed_step=step,
          next_step=step + 1, completed_control_periods=step // SUBSTEPS,
          next_control_refresh_step=step if step % SUBSTEPS == 0 and phase == 'committed_before_endpoint_handoff' else (step // SUBSTEPS + 1) * SUBSTEPS,
          phase=phase, held_controller=held, sr_history_origin_step=0, qlen=sr.qlen, jlen=sr.jlen,
          state_sha256=sha(path), state_fingerprint=c.fingerprint(st, sr)))


def restore(path, eng):
    """Restore a committed snapshot without checkpointing graphs or unused SR storage."""
    torch = c.torch; saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['n'] == eng.tables.n and saved['k'] == eng.tables.k and saved['h'] == eng.step.delta
    pre_sup = c.sp.make_support(saved['n'], saved['k'], False, tuple(saved['pre_ids']))
    tmv_sup = c.sp.make_support(saved['n'], saved['k'], True, tuple(saved['tmv_ids']))
    assert torch.equal(saved['pre_exponents'], eng.tables.exponents[list(pre_sup.ids)].cpu())
    assert torch.equal(saved['tmv_exponents'], eng.tables.exponents[eng.tables.spatial_index[list(tmv_sup.ids)]].cpu())
    st = c.se.SparseState(**{name: saved[name].to(eng.device) for name in ['pre','pre_rem','tmv','tmv_rem','status']},
                          pre_sup=pre_sup, tmv_sup=tmv_sup)
    sr = c.make_symbolic_remainder(len(st.pre), saved['n'], saved['sr_max_size'], eng.device)
    sr.reserve(saved['sr_capacity'])
    for name in ['scalars','scalars_iv','phi_buf','phi_iv_buf','j_buf']:
        value = saved.get('sr_' + name)
        if value is None: continue
        if name.startswith('phi') or name == 'j_buf':
            target = getattr(sr, name)
            if target is None:
                target = torch.empty((saved['sr_capacity'], *value.shape[1:]), dtype=value.dtype, device=eng.device)
                setattr(sr, name, target)
            target[:len(value)].copy_(value)
        else:
            setattr(sr, name, value.to(eng.device))
    sr.qlen, sr.jlen = saved['qlen'], saved['jlen']
    expected = json.loads(path.with_suffix('.json').read_text())
    assert sha(path) == expected['state_sha256']
    # JSON normalizes tuples into lists.
    assert json.loads(json.dumps(c.fingerprint(st, sr))) == expected['state_fingerprint']
    return st, sr, saved['cap'].to(eng.device)


def main(args):
    global c
    c = load_module('quad_continuation_common', Path(__file__).with_name('continuation.py'))
    edge = c.bootstrap(); torch = c.torch
    assert all(os.environ[key] == value for key, value in {
        'FLOWSTAR_EARLY_WEIGHTED':'1', 'FLOWSTAR_RECENTER_VALIDATION':'1',
        'FLOWSTAR_CENTER_NORMALIZATION':'1', 'FLOWSTAR_VALIDATION_POLICY':'solution_plus_one',
        'FLOWSTAR_WEIGHTED_VALIDATION':'0', 'FLOWSTAR_SELF_MAP_RETRIES':'8'}.items())
    original = c.ORIGINAL
    cfg_path = Path(original['config']); assert sha(cfg_path) == original['config_sha256']
    cfg = c.yaml.safe_load(cfg_path.read_text()); cfg['_config_dir'] = cfg_path.parent
    model_path = (cfg_path.parent.parent / cfg['model_dir']).resolve()
    assert str(model_path) == original['model'] and sha(model_path) == original['model_sha256']
    boxes_path = args.boxes or c.N / 'runs/archcomp_failure_20260923/huan_box_parity_preload/boxes.json'
    assert sha(boxes_path) == BOXES_SHA == original['boxes_sha256']
    assert float(cfg['ode_step_size']) == .005 and int(cfg['ode_order']) == 2 and float(cfg['cut_off_threshold']) == 1e-6
    assert cfg['remainder_estimation'] == [-.1, .1] and cfg['sr_queue'] == 1000
    assert cfg['step_size'] == .1 and round(cfg['step_size']/cfg['ode_step_size']) == SUBSTEPS
    driver_path = Path(original['driver_path']); assert sha(driver_path) == original['driver_sha256']
    endpoint_path = Path(original['endpoint_driver']['path']); assert sha(endpoint_path) == original['endpoint_driver']['sha256']
    driver = load_module('quad_original_controller_driver', driver_path)
    endpoint_driver = load_module('quad_repaired_endpoint_driver', endpoint_path)
    driver.end_of_time_s = endpoint_driver.end_of_time_s
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    # Exact official builder used by run_center.py --official-controller.
    from auto_LiRPA import BoundedModule
    setup_started = time.perf_counter()
    model = BoundedModule(driver.build_raw_net(cfg, experimental=False),
                         torch.zeros(1, *cfg['input_shape'][1:], dtype=torch.float64),
                         device='cuda:0', bound_opts=dict(cfg['bound_opts']))
    torch.cuda.synchronize(); setup_s = time.perf_counter() - setup_started
    boxes = json.loads(boxes_path.read_text()); leaves, coverage = root_leaves(boxes, args.mode)
    eng, sched = c.setup(16, .005, 'cuda', edge)
    st = c.se.initial_sparse_state(torch.tensor(leaves, dtype=torch.float64, device='cuda'), eng, sched)
    sr = c.make_symbolic_remainder(len(leaves), 16, 1000, 'cuda'); sr.reserve(1000)
    cap = torch.tensor([-.1, .1], dtype=torch.float64, device='cuda').expand(len(leaves),16,2).clone()
    settings = c.config.Settings(step=.005, order=2, cutoff=1e-6, remainder_estimation=.1, mode='strict', device='cuda')
    code = c.compile_ode(cfg['dynamics_expressions'], [x['name'] for x in cfg['initial_set']], order=1)
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    metadata = dict(mode=args.mode, root_id=ROOT_ID, original_root_count=BATCH, leaf_count=len(leaves),
        requested_periods=args.periods, target_step=args.periods * SUBSTEPS, coverage=coverage,
        initial_affine_coverage=initial_coverage(st, leaves, eng), boxes_path=str(boxes_path), boxes_sha256=sha(boxes_path),
        config_path=str(cfg_path), config_sha256=sha(cfg_path), model_path=str(model_path), model_sha256=sha(model_path),
        driver_path=str(driver_path), driver_sha256=sha(driver_path), endpoint_driver=original['endpoint_driver'],
        engine=original['engine'], extensions=original['extensions'], script_sha256=sha(__file__),
        common_script_sha256=sha(Path(__file__).with_name('continuation.py')),
        settings=dict(step=.005, order=2, cutoff=1e-6, remainder_estimation=.1, mode='strict', device='cuda'),
        code_order=1, dynamics=cfg['dynamics_expressions'], variable_names=[x['name'] for x in cfg['initial_set']],
        environment={k:v for k,v in os.environ.items() if k.startswith('FLOWSTAR_') or k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','TORCH_EXTENSIONS_DIR']},
        controller='official box/same-slope/native-f64; original physical NN inputs; fixed B1024 row1; zero/repeat padding qualification every refresh',
        end_to_end_strict_certificate=False,
        limitation='Inherited CROWN floating-point and general RN control injection are unqualified; plant strict validation is not a full NNCS proof. No safety/target verdict is computed.',
        timing_scope='Instrumented one-root diagnostic including two controller calls per refresh; not full benchmark speed',
        observer='Local-domain pre+R hulls, not composed-tmv; endpoint time fixed to h', controller_setup_s=setup_s)
    write(out / 'INPUT.json', metadata)
    rows = []; controllers = []; completed = 0; held = None; status = 'running'; started = time.perf_counter()
    try:
        for period in range(args.periods):
            if completed:
                driver.end_of_time_s(st, eng)
            held = controller_refresh(st, eng, driver, model, cfg, completed, out); controllers.append(held)
            for substep in range(SUBSTEPS):
                step = completed + 1
                trial, ok, trial_sr, elapsed = c.attempt(st, sr, code, eng, sched, settings, cap)
                row = dict(step=step, root_id=ROOT_ID, leaf_ids=list(range(len(leaves))), accepted=ok.cpu().tolist(),
                           root_covered=bool(ok.all()), advance_s=elapsed, parent_state_sr_unchanged=True,
                           trial_sr_length=trial_sr.jlen, held_controller_step=held['completed_step'])
                if not bool(ok.all()):
                    status = 'numerical_rejection'
                    checkpoint(out / 'last_committed.pt', st, sr, eng, cap, completed, held, metadata,
                               'committed_after_controller_injection' if substep == 0 else 'committed_before_endpoint_handoff')
                    c.save_state(out / ('failed_trial_' + str(step) + '.pt'), trial, trial_sr, eng, cap, step, 'speculative_no_commit')
                    rows.append(row); print(json.dumps(row), flush=True)
                    with (out/'steps.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
                    break
                st, sr = c.se.prune_state(trial, eng), trial_sr
                completed = step; row['sr_reset_if_full'] = sr.reset_if_full()
                assert sr.qlen == sr.jlen == completed  # Bound <=700; no permitted capacity reset yet.
                observe_started = time.perf_counter()
                tube = driver.hull_ranges_s(st, eng, 12)
                endpoint = driver.rows_range_over_time_sparse(st, eng, torch.full((len(leaves),2),.005,dtype=torch.float64,device='cuda'),12)
                assert bool(torch.isfinite(tube).all() and torch.isfinite(endpoint).all())
                row.update(tube=tube.cpu().tolist(), endpoint=endpoint.cpu().tolist(),
                           union_tube=torch.stack((tube[...,0].amin(0),tube[...,1].amax(0)),-1).cpu().tolist(),
                           union_endpoint=torch.stack((endpoint[...,0].amin(0),endpoint[...,1].amax(0)),-1).cpu().tolist(),
                           observer_s=time.perf_counter()-observe_started, committed_sr_length=sr.jlen)
                rows.append(row)
                with (out/'steps.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
                if completed in [20, 21] or completed == args.periods * SUBSTEPS:
                    path = out / ('committed_' + str(completed) + '.pt')
                    checkpoint(path,st,sr,eng,cap,completed,held,metadata,'committed_before_endpoint_handoff')
                    restored, restored_sr, restored_cap = restore(path,eng)
                    assert c.fingerprint(restored,restored_sr) == c.fingerprint(st,sr) and byte_equal(restored_cap,cap)
                    del restored, restored_sr, restored_cap
                if completed % SUBSTEPS == 0:
                    progress=dict(completed_step=completed,root_covered=True,leaf_count=len(leaves),elapsed_s=time.perf_counter()-started,sr_length=sr.jlen)
                    write(out/'progress.json',progress); print(json.dumps(progress),flush=True)
            if status == 'numerical_rejection': break
        else:
            status = 'target_completed'
    except BaseException as exc:
        status = 'exception'; write(out/'EXCEPTION.json',dict(type=type(exc).__name__,error=str(exc),completed_step=completed))
        raise
    finally:
        write(out/'RESULT.json',dict(status=status,completed_step=completed,root_covered_to_step=completed,
            original_all1024_common_prefix_unchanged=677,elapsed_s=time.perf_counter()-started,
            controllers=controllers,attempted_steps=len(rows),all_required_leaves=len(leaves),
            last_attempt=({k:v for k,v in rows[-1].items() if k not in ['tube','endpoint','union_tube','union_endpoint']} if rows else None),
            total_advance_s=sum(r['advance_s'] for r in rows),input_sha256=sha(out/'INPUT.json'),
            scope=metadata['limitation'],stop_phase='before_endpoint_handoff/controller_refresh' if status=='target_completed' else 'see_failure_checkpoint'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path); p.add_argument('--mode',choices=['baseline','split'],default='split')
    p.add_argument('--periods',type=int,choices=range(1,36),default=2); p.add_argument('--boxes',type=Path)
    p.add_argument('--check-local',action='store_true'); args=p.parse_args()
    if args.check_local:
        assert args.boxes and sha(args.boxes)==BOXES_SHA
        boxes=json.loads(args.boxes.read_text())
        for mode in ['baseline','split']:
            leaves,receipt=root_leaves(boxes,mode); assert receipt['covered']
        assert outward(F(1,3),False)<=F(1,3)<=outward(F(1,3),True)
        print('PASS: archived root1 identity; exact covering physical x5 split; outward midpoint conversions')
    else:
        assert args.output, '--output required for experiment'
        main(args)
