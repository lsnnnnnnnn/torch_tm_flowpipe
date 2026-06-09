# Batched Dense NNCS Demo Report

## Scope

This is a batched controller-bound plus dense plant loop. It is not CROWN-Reach parity and does not use Flow*.

## Direct Answers

- End-to-end CPU run: yes
- End-to-end CUDA run: yes
- First CUDA win batch: 2048
- Dominant part: plant step
- Closed-loop sampled containment: pass
- Controller bound overhead: not dominant
- Plant overhead: dominant
- Representation redesign needed next: tighter remainder/range bounding for wider closed-loop boxes, then richer controller linear bounds.
- Recommendation: GPU_PATH_CONTINUE

## Timing Rows

| batch | device | controller | elapsed ms | controller ms | plant ms | range ms | containment | speedup CPU | dominant |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1 | cpu | affine | 34.4861774 | 2.85635237 | 23.9627343 | 6.09677937 | yes |  | plant step |
| 8 | cpu | affine | 34.9308997 | 2.21117679 | 25.5495692 | 5.76052628 | yes |  | plant step |
| 32 | cpu | affine | 35.8812595 | 2.44524889 | 25.9588463 | 6.04753476 | yes |  | plant step |
| 128 | cpu | affine | 40.7044441 | 2.45161913 | 29.5449896 | 7.22994003 | yes |  | plant step |
| 512 | cpu | affine | 58.3674675 | 2.7954001 | 42.1936316 | 11.6468985 | yes |  | plant step |
| 2048 | cpu | affine | 133.371701 | 3.76458187 | 93.0934539 | 33.8103985 | yes |  | plant step |
| 1 | cpu | relu_ibp | 32.7068493 | 2.53548659 | 23.3528987 | 5.47155831 | yes |  | plant step |
| 8 | cpu | relu_ibp | 35.1803433 | 2.65286583 | 25.3390633 | 5.77610079 | yes |  | plant step |
| 32 | cpu | relu_ibp | 36.0692739 | 2.73597147 | 25.8633038 | 6.04101829 | yes |  | plant step |
| 128 | cpu | relu_ibp | 40.6961562 | 2.89065577 | 28.9826142 | 7.27881491 | yes |  | plant step |
| 512 | cpu | relu_ibp | 53.3248959 | 3.32232378 | 36.5199568 | 11.6955787 | yes |  | plant step |
| 2048 | cpu | relu_ibp | 105.319518 | 4.14315984 | 65.2120067 | 33.3242603 | yes |  | plant step |
| 1 | cuda | affine | 249.578772 | 36.0252997 | 140.372863 | 69.5879729 | yes | 0.138177526 | plant step |
| 8 | cuda | affine | 120.057197 | 4.75958828 | 97.3697864 | 14.5886606 | yes | 0.290952151 | plant step |
| 32 | cuda | affine | 114.588744 | 4.64810431 | 92.8659486 | 13.8267037 | yes | 0.313130751 | plant step |
| 128 | cuda | affine | 115.057463 | 4.46729828 | 93.5073355 | 13.8844047 | yes | 0.353774914 | plant step |
| 512 | cuda | affine | 115.348511 | 4.52110823 | 93.7326448 | 13.8762984 | yes | 0.506009716 | plant step |
| 2048 | cuda | affine | 115.944743 | 4.42628283 | 94.0050939 | 14.2813148 | yes | 1.15030399 | plant step |
| 1 | cuda | relu_ibp | 133.853084 | 22.2894503 | 94.62531 | 13.7623176 | yes | 0.244348865 | plant step |
| 8 | cuda | relu_ibp | 115.219763 | 5.87638095 | 92.435536 | 13.7232766 | yes | 0.305332545 | plant step |
| 32 | cuda | relu_ibp | 117.816347 | 7.17002805 | 93.376914 | 14.0120592 | yes | 0.306148297 | plant step |
| 128 | cuda | relu_ibp | 116.538339 | 5.91922645 | 93.476491 | 13.8863716 | yes | 0.349208307 | plant step |
| 512 | cuda | relu_ibp | 117.057282 | 5.9350878 | 94.0221706 | 13.8709377 | yes | 0.455545313 | plant step |
| 2048 | cuda | relu_ibp | 121.812142 | 6.00301567 | 98.6820571 | 13.8918972 | yes | 0.864606069 | plant step |