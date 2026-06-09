# Batched Dense Many-Box Plant Demo Report

## Scope

This is a plant-only, explicit Euler-style dense Taylor-model workload. It does not use Flow*, adaptive rejection, or Flow* Picard integration.

## Direct Answers

- Does dense CPU beat scalar loop? yes
- Does CUDA beat dense CPU? yes
- First CUDA win batch: 512
- Merged dropped-term bounding reduces width: yes
- Split range bound cost/tightness: situational: width improves, median cost 3.19x interval
- Sampled trajectory containment: pass
- Dominant operation: rhs construction
- Recommendation: GPU_PATH_CONTINUE

## Timing Rows

| batch | device | impl | steps | mode | dropped | elapsed ms | containment | speedup scalar | speedup CPU | max width | dominant |
| ---: | --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| 1 | cpu | scalar_loop | 50 | n/a | n/a | 832.091284 | yes |  |  | 1.20801163 | scalar_loop |
| 8 | cpu | scalar_loop | 50 | n/a | n/a | 6691.22671 | yes |  |  | 1.62599998 | scalar_loop |
| 32 | cpu | scalar_loop | 50 | n/a | n/a |  |  |  |  |  |  |
| 128 | cpu | scalar_loop | 50 | n/a | n/a |  |  |  |  |  |  |
| 512 | cpu | scalar_loop | 50 | n/a | n/a |  |  |  |  |  |  |
| 2048 | cpu | scalar_loop | 50 | n/a | n/a |  |  |  |  |  |  |
| 8192 | cpu | scalar_loop | 50 | n/a | n/a |  |  |  |  |  |  |
| 1 | cpu | torch_dense | 50 | interval | termwise | 228.930833 | yes | 3.63468421 |  | 1.32160688 | rhs construction |
| 1 | cpu | torch_dense | 50 | interval | merged | 223.997667 | yes | 3.71473193 |  | 1.20801163 | rhs construction |
| 1 | cpu | torch_dense | 50 | split2 | termwise | 796.931431 | yes | 1.04411904 |  | 0.972009553 | rhs construction |
| 1 | cpu | torch_dense | 50 | split2 | merged | 782.94782 | yes | 1.06276723 |  | 0.915180352 | rhs construction |
| 8 | cpu | torch_dense | 50 | interval | termwise | 254.881899 | yes | 26.2522633 |  | 1.78563938 | rhs construction |
| 8 | cpu | torch_dense | 50 | interval | merged | 241.455746 | yes | 27.712021 |  | 1.62599998 | rhs construction |
| 8 | cpu | torch_dense | 50 | split2 | termwise | 872.926875 | yes | 7.66527747 |  | 1.29333122 | rhs construction |
| 8 | cpu | torch_dense | 50 | split2 | merged | 843.434631 | yes | 7.93330801 |  | 1.21345962 | rhs construction |
| 32 | cpu | torch_dense | 50 | interval | termwise | 294.357054 | yes |  |  | 1.78563938 | rhs construction |
| 32 | cpu | torch_dense | 50 | interval | merged | 264.984983 | yes |  |  | 1.62599998 | rhs construction |
| 32 | cpu | torch_dense | 50 | split2 | termwise | 1029.40467 | yes |  |  | 1.29333122 | rhs construction |
| 32 | cpu | torch_dense | 50 | split2 | merged | 944.720617 | yes |  |  | 1.21345962 | rhs construction |
| 128 | cpu | torch_dense | 50 | interval | termwise | 524.731636 | yes |  |  | 1.7892595 | rhs construction |
| 128 | cpu | torch_dense | 50 | interval | merged | 336.69539 | yes |  |  | 1.63046955 | rhs construction |
| 128 | cpu | torch_dense | 50 | split2 | termwise | 1942.22454 | yes |  |  | 1.29629652 | rhs construction |
| 128 | cpu | torch_dense | 50 | split2 | merged | 1183.70404 | yes |  |  | 1.21684986 | rhs construction |
| 512 | cpu | torch_dense | 50 | interval | termwise | 1432.30816 | yes |  |  | 1.83459475 | rhs construction |
| 512 | cpu | torch_dense | 50 | interval | merged | 619.20105 | yes |  |  | 1.6705168 | rhs construction |
| 512 | cpu | torch_dense | 50 | split2 | termwise | 4972.7621 | yes |  |  | 1.32709301 | rhs construction |
| 512 | cpu | torch_dense | 50 | split2 | merged | 2088.50256 | yes |  |  | 1.24499957 | rhs construction |
| 2048 | cpu | torch_dense | 50 | interval | termwise | 5697.62004 | yes |  |  | 1.83459475 | rhs construction |
| 2048 | cpu | torch_dense | 50 | interval | merged | 1716.80087 | yes |  |  | 1.6705168 | rhs construction |
| 2048 | cpu | torch_dense | 50 | split2 | termwise | 19841.8685 | yes |  |  | 1.32709301 | rhs construction |
| 2048 | cpu | torch_dense | 50 | split2 | merged | 5670.86362 | yes |  |  | 1.24499957 | rhs construction |
| 8192 | cpu | torch_dense | 50 | interval | termwise | 29592.871 | yes |  |  | 1.83459475 | rhs construction |
| 8192 | cpu | torch_dense | 50 | interval | merged | 10536.6528 | yes |  |  | 1.6705168 | rhs construction |
| 8192 | cpu | torch_dense | 50 | split2 | termwise | 105123.177 | yes |  |  | 1.32709301 | rhs construction |
| 8192 | cpu | torch_dense | 50 | split2 | merged | 27964.2847 | yes |  |  | 1.24499957 | rhs construction |
| 1 | cuda | torch_dense | 50 | interval | termwise | 861.12642 | yes |  | 0.265850434 | 1.32160688 | rhs construction |
| 1 | cuda | torch_dense | 50 | interval | merged | 793.121277 | yes |  | 0.282425491 | 1.20801163 | rhs construction |
| 1 | cuda | torch_dense | 50 | split2 | termwise | 2203.17303 | yes |  | 0.361719856 | 0.972009553 | rhs construction |
| 1 | cuda | torch_dense | 50 | split2 | merged | 2200.63027 | yes |  | 0.355783446 | 0.915180352 | rhs construction |
| 8 | cuda | torch_dense | 50 | interval | termwise | 749.758834 | yes |  | 0.339951845 | 1.78563938 | rhs construction |
| 8 | cuda | torch_dense | 50 | interval | merged | 747.595359 | yes |  | 0.322976518 | 1.62599998 | rhs construction |
| 8 | cuda | torch_dense | 50 | split2 | termwise | 2236.76938 | yes |  | 0.39026235 | 1.29333122 | rhs construction |
| 8 | cuda | torch_dense | 50 | split2 | merged | 2231.51629 | yes |  | 0.37796481 | 1.21345962 | rhs construction |
| 32 | cuda | torch_dense | 50 | interval | termwise | 749.812821 | yes |  | 0.392574047 | 1.78563938 | rhs construction |
| 32 | cuda | torch_dense | 50 | interval | merged | 756.241856 | yes |  | 0.350397139 | 1.62599998 | rhs construction |
| 32 | cuda | torch_dense | 50 | split2 | termwise | 2229.05925 | yes |  | 0.461811263 | 1.29333122 | rhs construction |
| 32 | cuda | torch_dense | 50 | split2 | merged | 2235.5863 | yes |  | 0.422582934 | 1.21345962 | rhs construction |
| 128 | cuda | torch_dense | 50 | interval | termwise | 750.023063 | yes |  | 0.699620667 | 1.7892595 | rhs construction |
| 128 | cuda | torch_dense | 50 | interval | merged | 760.678576 | yes |  | 0.442625046 | 1.63046955 | rhs construction |
| 128 | cuda | torch_dense | 50 | split2 | termwise | 2251.91928 | yes |  | 0.862475204 | 1.29629652 | rhs construction |
| 128 | cuda | torch_dense | 50 | split2 | merged | 2259.15217 | yes |  | 0.52395941 | 1.21684986 | rhs construction |
| 512 | cuda | torch_dense | 50 | interval | termwise | 755.436498 | yes |  | 1.89600075 | 1.83459475 | rhs construction |
| 512 | cuda | torch_dense | 50 | interval | merged | 760.224024 | yes |  | 0.814498136 | 1.6705168 | rhs construction |
| 512 | cuda | torch_dense | 50 | split2 | termwise | 2270.00353 | yes |  | 2.19064069 | 1.32709301 | rhs construction |
| 512 | cuda | torch_dense | 50 | split2 | merged | 2263.36294 | yes |  | 0.92274311 | 1.24499957 | rhs construction |
| 2048 | cuda | torch_dense | 50 | interval | termwise | 768.838332 | yes |  | 7.41068675 | 1.83459475 | rhs construction |
| 2048 | cuda | torch_dense | 50 | interval | merged | 761.931645 | yes |  | 2.25322164 | 1.6705168 | rhs construction |
| 2048 | cuda | torch_dense | 50 | split2 | termwise | 2291.22766 | yes |  | 8.65992884 | 1.32709301 | rhs construction |
| 2048 | cuda | torch_dense | 50 | split2 | merged | 2298.01785 | yes |  | 2.46771957 | 1.24499957 | rhs construction |
| 8192 | cuda | torch_dense | 50 | interval | termwise | 985.753643 | yes |  | 30.0205546 | 1.83459475 | rhs construction |
| 8192 | cuda | torch_dense | 50 | interval | merged | 764.532828 | yes |  | 13.7818187 | 1.6705168 | rhs construction |
| 8192 | cuda | torch_dense | 50 | split2 | termwise | 3141.27079 | yes |  | 33.4651752 | 1.32709301 | rhs construction |
| 8192 | cuda | torch_dense | 50 | split2 | merged | 2285.09064 | yes |  | 12.2377136 | 1.24499957 | rhs construction |

Scalar loop was checked for 2 row(s); larger scalar batches are skipped by --scalar-cap.
