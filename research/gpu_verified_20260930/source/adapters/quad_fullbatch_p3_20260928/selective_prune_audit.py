"""Read-only plan for the frozen working-graph eviction; no solver changes.

Plans contain process-local integer identities only. Recheck immediately before
the original finish_step; compare its receipt afterward. A no-op plan does not
itself establish state/SR equality: the caller must retain its storage/version
guard. No tensor, key, cache, engine or closure references escape in a plan.
"""
from collections import Counter
from pathlib import Path
import hashlib,inspect

WORKING_PRUNE_SHA='3a174e02724ee3166eff9f1b330c137e0c1a4b51f64a6162ea2a6ed96814e21f'
SCHEMA='frozen_working_prune_readonly_plan_v1'

def peek(binding,eng):
    finish=binding.finish_step
    assert finish.__qualname__=='install.<locals>.finish_step'
    path=Path(inspect.getfile(finish))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==WORKING_PRUNE_SHA
    assert binding.source_sha256==WORKING_PRUNE_SHA
    closure=inspect.getclosurevars(finish).nonlocals
    active,target,used,initial=(closure[k] for k in ('active','target','used','initial'))
    category,torch=closure['category'],closure['torch']
    assert category.__qualname__=='install.<locals>.category' and Path(inspect.getfile(category))==path
    assert active is eng and target is getattr(eng,'_graphs',None) and target is not None
    assert target.enabled and not torch.cuda.is_current_stream_capturing()
    assert type(used) is set and type(initial) is set and used
    current=set(target._segs)
    assert used<=current,'Working graph entries disappeared during the advance'
    assert (current-initial)<=used,'Untracked direct working capture is unsupported'
    stale=current-used
    assert type(target.captures) is int and type(target.hits) is int
    return dict(schema=SCHEMA,working_owner_id=id(eng),graph_owner_id=id(target),
        capture_stream_id=id(target._capture_stream),captures=target.captures,hits=target.hits,
        initial_key_ids=sorted(map(id,initial)),current_key_ids=sorted(map(id,current)),
        used_key_ids=sorted(map(id,used)),stale_key_ids=sorted(map(id,stale)),
        stale_count=len(stale),stale_categories=dict(sorted(Counter(category(k) for k in stale).items())))

def assert_same_plan(binding,eng,expected):
    actual=peek(binding,eng)
    assert actual==expected,'Working graph plan changed before original finish_step'
    return actual

def assert_finished(plan,row):
    assert plan['schema']==SCHEMA
    for key in ('working_owner_id','graph_owner_id','capture_stream_id','captures','hits'):
        assert row[key]==plan[key],key
    assert row['nkeys_before']==len(plan['current_key_ids'])
    assert row['used_keys']==row['nkeys_after']==len(plan['used_key_ids'])
    assert row['evicted_entries']==plan['stale_count']
    assert row['evicted_categories']==plan['stale_categories']
    assert row['synchronized_and_empty_cache'] is bool(plan['stale_count'])
    assert row['validation_and_weighted_untouched'] is True
    return True
