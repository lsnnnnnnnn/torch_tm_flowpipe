# Contract field map

| Field | Xiangru native Q3 | Torch candidate | Flowstar reference | Matched? |
|---|---|---|---|---|
| Model | closed-loop homogeneous TORA | plant-only Van der Pol | plant-only Van der Pol | no |
| State | x1,x2,x3,x4,u1 | x,y | x,y,t | no |
| Controller | frozen ReLU ONNX, auto_LiRPA, period 1 s | none | none | no |
| Initial set | 4D box + held u, B48 | [1.1,1.4]x[2.35,2.45] | same VDP box | no |
| Horizon | 20 | 10 | 10 | no |
| Q/order | dense total-degree Q3, K2+10 DR-RP | current VDP order 4; order-3 sparse K3 | order 4 | predicate only |
| Step | fixed 0.1 | adaptive 0.002-0.1 | adaptive 0.002-0.1 | no |
| Device/workload | CUDA closed-loop | CPU plant-only | CPU plant-only | no |

Full evidence and per-field reasons are in the three machine-readable candidate contracts.
