# TORA remain：Flow* native `h=0.05` 隔离诊断

2026-10-02，在 Huan/Xiangru 的[同合同减半步长诊断](../author_tora_remain_h005_probe_20261002/SUMMARY.md)之后，新建服务器目录 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_tora_remain_h005_probe_20261002`。事前 [INTENT](INTENT.json) 固定一期门检、首败即停和主表隔离；旧 `h=0.1` 原生二进制与原始作业均未运行或改动。

## 变更及执行门

[两份新 C++ 源码](build/archcomp/TORA/tora_remain_h005_full20.cpp)由[旧 h=0.1 源码快照](frozen_h01_source/tora_remain_full20.cpp)复制；一期版亦分别保留。逐字文本比较确认每份**唯一源码改动**是 `setting.setFixedStepsize(0.1, 3);` → `setting.setFixedStepsize(0.05, 3);`。仍是官方 TORA remain 四态 ODE、12 盒完整初集、官方 ONNX、`x4'=u−10`、1 秒控制保持、20 秒时域和全时 `[-2,2]^4` 安全集。[编译收据](BUILD.json)、[一期](smoke1.build.log)及[全程](full20.build.log)编译日志保留原始命令和零退出码；[RPC 服务器源码](build/archcomp/TORA/crown_paper.py)、[模型副本](build/official_controllerTora_2026.onnx)、range exporter 与[执行脚本](run_native_pair.sh)亦随本次新目录保存。两个新二进制保留在服务器上述隔离 build 目录，未把旧二进制当作 h=0.05 结果。

服务器先核对 GPU2、CPU10–13 和 RPC 端口 5100 空闲；与另一原生短例的编译错开。新二进制导出的 [12 个实际初盒](initial_boxes.json)与旧原生 `h=0.1` 的保存分区逐值相同。先经 no-hash supervisor 对[一期门检](smoke1_001/START.json)限时 120 秒；只有其结果和全部 240 条区间的本地独立扫描通过，才对新 20 期 binary 发起限时 240 秒的[完整诊断](full20_001/START.json)。两次进程后 5100 均无监听。

| 隔离作业 | 外层进程 | 周期 / RPC | ODE 小步 | 保存范围 | 作者 checker | 独立保存范围审计 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [一期门检](smoke1_001/RESULT.json) | exit 0，wall 4.227550 s | 1 / 1 | 20 | 12×20 = 240 | `VERIFIED` | [完整、有限有序、逐段包含、全时安全](smoke1_001/INDEPENDENT_SCAN.json) |
| [完整 20 期](full20_001/RESULT.json) | exit 0，wall 13.301876 s | 20 / 20 | 400 | 12×400 = 4,800 | `VERIFIED` | [完整、有限有序、逐段包含、全时安全](full20_001/INDEPENDENT_SCAN.json) |

[独立扫描器](audit_native_h005_nohash.py)读取原始 [4,800 条二进制范围](full20_001/ranges.bin)，逐个检查 `(lane,step,h)` 的 12×400 完整网格、四态每个 tube/endpoint 的有限及有序、endpoint 位于同段 tube、每个已保存 tube 在 `[-2,2]^4` 内，并核对 20 条 RPC 的 12×4 输入与 12×1 输出有限有序。保存全时四态 tube union 为 `x1=[-0.9778407860,0.8475905369]`、`x2=[-1.0044494350,0.9187716269]`、`x3=[-1.1719374899,1.0304714949]`、`x4=[-1.3987920160,1.5556432564]`；`T=20` endpoint union 见扫描 JSON。原生 `ranges.bin` 本身不编码逐步 accepted/status；完整的 20 次 `Step`、最终 `VERIFIED`、exit 0、RPC 和原始 [native.log](full20_001/native.log)构成相邻运行观察。

这是**另一个数值步长 profile 的单次完整原生结果**，不替换固定 `h=0.1` 四方主表，也不拿跨步长 wall 排速度。作者 checker 的 `VERIFIED` 与保存 tube 安全观察分开记录；原始 [server.log](full20_001/server.log)仍有 auto_LiRPA batch-dimension constant 警告。这里没有独立端到端浮点 NN/CROWN/Flow* 组合证明；本次没有做内容摘要或哈希校验。
