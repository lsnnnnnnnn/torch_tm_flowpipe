# S0 TORA Static Input-Partition Sweep

Status: `PARTIAL_VALIDATION`.

All cells use fixed initial grids, K2 polynomial construction,
DR-RP validation, and the same PyTorch/CUDA implementation.

## Validity

- Source hashes/statuses: `True`.
- Grid covers: `True`.
- B12 source classification: `NOT_EVALUATED`.
- CPU/CUDA and outward replays: `True`.

## Cells

| Policy | Method | B | Result | Certified horizon | Failure | Leaf attempts | Solver s | Peak CUDA MiB |
|---|---|---:|---|---:|---:|---:|---:|---:|
| b48_static | complete_q3 | 48 | VERIFIED | 20.0 | none | 9600 | 2.69881 | 91.490 |

## Decision

This is a partial validation run, not the formal static-sweep decision.

CROWN-Reach commit: `27d29050a5f214b56f211ca9cb411e734ed80230` (dirty: `False`); DiffReach commit: `dd628eb443b517d6415de93e7035b4baef73963e`.

Command: `/srv/local/shengenli/native_envs/crownreach28/bin/python /home/xiangru4/CROWN-Reach_Development/experiments/remainder_ablation/run_s0_tora_static_partition_sweep.py --device cuda --policies b48_static --methods complete_q3 --cap 200 --q3-engine dynamic --controller-backend autolirpa --controller-platform cuda --controller-mode eager --controller-composition outward --diffreach-root /home/xiangru4/DiffReach --diffreach-python /srv/local/shengenli/native_envs/diffreach083/bin/python --output-json /tmp/s3c_27d2905/rep1_q3/s3r_q3_b48_rep1.json --output-markdown /tmp/s3c_27d2905/rep1_q3/s3r_q3_b48_rep1.md`
