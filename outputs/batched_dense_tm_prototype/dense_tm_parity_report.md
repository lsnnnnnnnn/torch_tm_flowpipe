# Dense Batched Taylor Model Parity Report

The tests compare dense coefficient arithmetic against the sparse Polynomial path for small cases. This demo report adds sampled fixed-step Van der Pol containment checks.

| batch | device | dense contains samples | scalar checked | scalar contains samples | dense contains scalar ranges | max dense width | max scalar width |
| ---: | --- | --- | --- | --- | --- | ---: | ---: |
| 1 | cpu | yes | yes | yes | yes | 0.07257 | 0.07253 |
| 8 | cpu | yes | yes | yes | yes | 0.07674 | 0.07669 |
| 32 | cpu | yes | yes | yes | yes | 0.07674 | 0.07669 |
| 128 | cpu | yes | no |  |  | 0.07674 |  |
| 512 | cpu | yes | no |  |  | 0.0771 |  |
| 2048 | cpu | yes | no |  |  | 0.0771 |  |
| 1 | cuda | yes | yes | yes | yes | 0.07257 | 0.07253 |
| 8 | cuda | yes | yes | yes | yes | 0.07674 | 0.07669 |
| 32 | cuda | yes | yes | yes | yes | 0.07674 | 0.07669 |
| 128 | cuda | yes | no |  |  | 0.07674 |  |
| 512 | cuda | yes | no |  |  | 0.0771 |  |
| 2048 | cuda | yes | no |  |  | 0.0771 |  |

## Limitations

Remainder multiplication uses interval bounds for the retained polynomial ranges and dropped dense monomial products. This is conservative for the sampled small cases here, but it is not a proof of production reachability performance or tightness.