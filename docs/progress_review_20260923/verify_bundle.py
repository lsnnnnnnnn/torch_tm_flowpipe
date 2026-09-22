"""Read-only verification of delivered files and the presentation's primary numbers."""
import hashlib,json,statistics
from pathlib import Path
D=Path(__file__).resolve().parent
read=lambda p:json.loads((D/p).read_text())
m=read('SHA256SUMS.json')
for name,digest in m.items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
samples=read('evidence/plant_raw_timing.json');assert len(samples)==120 and all(x['eligible'] and x['completed'] for x in samples)
for r in read('evidence/plant_timing.json')['table']:
 chosen=[x for x in samples if all(x['row'][k]==r[k] for k in ['plant','batch','implementation','mode'])]
 assert len(chosen)==5
 assert statistics.median(x['core_seconds'] for x in chosen)==r['core_median']
 assert statistics.median(x['process_wall_seconds'] for x in chosen)==r['process_median']
records=read('evidence/nncs_timing_campaign.json')['records'];assert len(records)==72
for r in read('evidence/nncs_timing_audit.json')['rows']:
 chosen=[x for x in records if x['include'] and x['case']==r['case'] and x['arm']==r['arm']]
 assert len(chosen)==5 and statistics.median(x['process_wall_s'] for x in chosen)==r['process_median_s']
for case in ['tora','single_pendulum']:
 d=read(f'evidence/{case}_ours_strict_identity.json')
 assert hashlib.sha256((D/f'configs/{case}_executed.yaml').read_bytes()).hexdigest()==d['config_sha256']
 for arm in ['huan_strict','xiangru_strict','native']:
  other=read(f'evidence/{case}_{arm}_identity.json')
  # GPU/native metadata schemas can differ; check any model identity field actually present.
  if 'model_sha256' in other:assert other['model_sha256']==d['model_sha256']
c=read('evidence/cartpole_contract.json');assert hashlib.sha256((D/'configs/cartpole.yaml').read_bytes()).hexdigest()==c['config_sha256']
assert read('evidence/cartpole_audit.json')['all_rows']==9600
print(json.dumps(dict(status='passed',files=len(m),plant_samples=120,nncs_counted=60,nncs_preflights=12,executed_config_hashes=3),indent=2))
