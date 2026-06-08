# Dense Batched Taylor Model Demo Report

## Scope

This is an experimental dense tensor prototype. It does not add a Flow* mechanism, does not modify the C++ probe, and does not replace production Polynomial/TaylorModel classes.

## Configuration

- Batches: 1,8,32,128,512,2048
- Steps: 10
- Order: 4
- Step size h: 0.01
- Dtype: float64
- Scalar baseline cap: 32
- CUDA available: yes

## Answers

- Contains sampled trajectories: yes
- Contains scalar sampled trajectories: yes
- CPU beats scalar: yes
- CUDA beats CPU: yes
- Dominant operation: rhs_mul_trunc
- Blockers: remainder bounds are still interval-only and need tighter validation before larger claims
- Next step: GPU_PATH_CONTINUE

## Timing Rows

| batch | device | implementation | elapsed ms | sample containment | speedup vs scalar | speedup vs CPU | dominant |
| ---: | --- | --- | ---: | --- | ---: | ---: | --- |
| 1 | cpu | scalar_loop | 148.1 | yes |  |  |  |
| 8 | cpu | scalar_loop | 1256 | yes |  |  |  |
| 32 | cpu | scalar_loop | 4802 | yes |  |  |  |
| 128 | cpu | scalar_loop |  |  |  |  |  |
| 512 | cpu | scalar_loop |  |  |  |  |  |
| 2048 | cpu | scalar_loop |  |  |  |  |  |
| 1 | cpu | torch_dense | 50.02 | yes | 2.962 |  | rhs_mul_trunc |
| 8 | cpu | torch_dense | 55 | yes | 22.84 |  | rhs_mul_trunc |
| 32 | cpu | torch_dense | 63.6 | yes | 75.51 |  | rhs_mul_trunc |
| 128 | cpu | torch_dense | 111.4 | yes |  |  | rhs_mul_trunc |
| 512 | cpu | torch_dense | 307.6 | yes |  |  | rhs_mul_trunc |
| 2048 | cpu | torch_dense | 1275 | yes |  |  | rhs_mul_trunc |
| 1 | cuda | torch_dense | 302.5 | yes |  | 0.1653 | rhs_mul_trunc |
| 8 | cuda | torch_dense | 123.4 | yes |  | 0.4458 | rhs_mul_trunc |
| 32 | cuda | torch_dense | 122.6 | yes |  | 0.5187 | rhs_mul_trunc |
| 128 | cuda | torch_dense | 123.5 | yes |  | 0.9024 | rhs_mul_trunc |
| 512 | cuda | torch_dense | 124.7 | yes |  | 2.467 | rhs_mul_trunc |
| 2048 | cuda | torch_dense | 126.6 | yes |  | 10.07 | rhs_mul_trunc |
