# Airplane continuous order-6 全初盒 smoke：资源阻断

本次是新隔离的一盒完整官方初集尝试；远端原始目录为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/airplane_continuous_huan_smoke1_001`。官方物理初集中的 `u,v,w,phi,theta,psi` 均为 `[0,1]`；旧单点初集没有参与。本轮未运行内容摘要校验。

CPU-only [完整初盒预检](../airplane_continuous_fullbox_20261002/PREFLIGHT.json)已通过：19 条 ODE、8 条全时约束与固定 ONNX `[N,12]→[N,6]` 可解析。该数值候选沿用旧 Huan `h=0.01, Taylor order=6`，它不是官方强制参数。[START.json](START.json) 与 [payload 配置](payload/config.yaml) 保存实际身份。

启动后尚未产生 `Step 0` 或任何 ODE 小步，`observations.jsonl` 和 `ranges.bin` 均为空。约 3 分钟时子进程 CPU 使用率 100%、RSS 54,006,540 KiB、GPU2 仅 438 MiB。出于资源控制，仅向本次子进程组发出 SIGTERM；[外层 RESULT](RESULT.json)记录 wall 252.247 s、exit `-15`、failed；无 payload RESULT。原始文件保留，未重启同一身份。

阻断来自当前 GPU 引擎的全单项式表结构：`flowstar_gpu/monomials.py::build_tables(n=19,k=6)` 建立工作基底 `T=C(26,6)=230,230`，并将所有工作基底两两配对，`T²=53,005,852,900`。仅两个 int64 pair 索引的理论存储就约 848,093,646,400 字节，且积基底 `C(32,12)=225,792,840` 的 20 列 int64 指数另需约 36,126,854,400 字节。因此当前后端无法把这个 order-6 候选推进到第一个数值步；这不是 Airplane 性质的 `UNKNOWN` 或 `UNSAFE` 判定。

[资源采样](RESOURCE_OBSERVATION.json)是停止前的单次观测，不宣称实际峰值。后续若用 Taylor order 3，必须以独立命名 profile、新配置和新目录报告，不能回填本次 order-6 尝试。
