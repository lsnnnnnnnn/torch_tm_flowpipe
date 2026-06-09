# Batched Dense Range/Remainder Engineering Report

## Scope

This is a B-line representation-engineering sweep over the experimental dense batched Taylor-model path. It does not use Flow*, does not rerun h10, and does not claim Flow* parity.

## Configuration

- Batches: 128,512,2048,8192
- Many-box: steps=50, order=4, h=0.01
- NNCS: num_control_steps=10, plant_substeps=5, order=3, h=0.01
- Devices: cpu,cuda
- Range modes: interval, blocked_interval
- Dropped modes: merged, grouped

## Direct Answers

- Does blocked_interval reduce memory or runtime? yes; runtime min 0.873x, median 1.01x interval elapsed; memory min 1x, median 1x interval CUDA peak allocated.
- Does grouped dropped-term reduce width vs merged? no clear width reduction; grouped/merged width min 1x, max 1x merged max width.
- Does containment still pass? yes.
- Does NNCS CUDA crossover move below 2048? no; first NNCS CUDA win batch: 2048.
- Affine NNCS first CUDA win batch: 2048.
- ReLU-IBP NNCS first CUDA win batch: 8192.
- Is relu_ibp still slower than CPU at batch 2048? yes; best relu_ibp CUDA speedup at 2048: 0.826x.
- Recommendation: CONTINUE_BLOCKED_DENSE.

## Notes

`blocked_interval` chunks interval term accumulation to reduce range-bound intermediates. `grouped` keeps the tensorized duplicate-exponent scatter path for dropped products. Both modes preserve sampled containment in this sweep.

## Output

- `outputs/batched_dense_range_remainder_engineering/range_remainder_summary.csv`
