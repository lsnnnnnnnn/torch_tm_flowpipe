# NAV standard native 首盒首周期诊断

新运行 ID：`nav_author_standard_native_smoke1_001`。原始 [START](START.json)、[RESULT](RESULT.json)、[原生日志](native.log)、[控制器服务日志](server.log)、[控制器 RPC 记录](controller_rpc.jsonl)、[保存区间](ranges.bin)均从服务器原目录复制；原目录为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/nav_author_standard_native_smoke1_001`。新源码与构建记录在相邻的 [`native_nav_standard_author_build_001`](../native_nav_standard_author_build_001/BUILD.json)。没有重新运行旧 NAV 作业或计算文件摘要。

- 固定官方 2026 `nn-nav-point.onnx`；作者可执行原序 `[x,y,speed,heading]` 与 `[speed_rate,heading_rate]`；Flow* 0.01 秒固定步长、阶数 4、原生进程 CPU 6–9、控制器 GPU 1、RPC 端口 5110。
- 初盒来自旧 640 盒台账第 0 盒：`x∈[2.9,2.9050000000000002]`、`y∈[2.9,2.9125]`、其余状态零。只跑 `t∈[0,0.2]`，不把 `t=6` 目标提前检查。
- 外层 RESULT 为 `completed`、exit 0、wall `3.875127676874399 s`；原生日志为 `COMPLETED_PERIODS 1/1` 和 `PREFIX_COMPLETE_ONLY`。
- [独立区间扫描](INDEPENDENT_SAVED_RANGE_SCAN.json)确认 1 盒 × 20 小步 = 20 条，记录大小 152 字节，完整网格、有限、有序，端点在同小步 tube 内。[NAV 性质投影扫描](INDEPENDENT_NAV_PROPERTY_SCAN.json)确认 20 条保存 tube 均与闭障碍盒 `[1,2]²` 分离；终点目标未检查。
- `t=0.2` 的保存末端 x 为 `[2.880198146241115,2.885198281114323]`，y 为 `[2.8973532536068043,2.9098525088179685]`。这只是单盒首周期的保存数值区间观察，不是完整初集、完整 6 秒性质或独立端到端浮点证明。此 wall 含短作业启动开销，不能用于速度排序。
