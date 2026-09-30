"""Verify this publication without importing the numerical implementation."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'PUBLICATION_MANIFEST.json').read_text())
source_map = json.loads((root / 'source/SOURCE_MAP.json').read_text())
checked = 0
for item in manifest['files']:
    path = root / item['path']
    data = path.read_bytes()
    assert len(data) == item['bytes'], path
    assert hashlib.sha256(data).hexdigest() == item['sha256'], path
    if path.suffix == '.py':
        ast.parse(data, filename=str(path))
    elif path.suffix == '.json':
        json.loads(data)
    checked += 1
for item in source_map['files']:
    data = (root / 'source' / item['path']).read_bytes()
    assert len(data) == item['bytes'], item['path']
    assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
model = root / 'source/benchmark/quad_controller_3_64_torch.onnx'
assert hashlib.sha256(model.read_bytes()).hexdigest() == 'fabd84e411f4b0ebe0d6b996be6e4bd2adc48cf1118ab99c82180563b95532dd'
periods = [json.loads(line) for line in (root / 'evidence/native_terminal_20260930/periods.jsonl').read_text().splitlines()]
assert [row['period'] for row in periods] == list(range(30))
assert all(row['all_lanes_completed'] and len(row['lanes']) == 1024 and all(lane['completed'] and lane['accepted_this_period'] == 20 for lane in row['lanes']) for row in periods)
print(f'PASS: {checked} payload files; {len(source_map["files"])} frozen sources; model identity; native 30 complete periods. No numerical run executed.')
