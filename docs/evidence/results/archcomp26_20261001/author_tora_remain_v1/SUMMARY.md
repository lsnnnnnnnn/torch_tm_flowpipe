# 2026 TORA remain — Huan / Xiangru 新尝试

日期：2026-10-01。两方使用固定 2026 remain 规格与同一共享控制驱动，分别加载 Huan / Xiangru plant engine：12 个覆盖子盒、20 个 1 s 控制周期、Taylor order 3、0.1 s ODE 小步、全时四物理态 `[-2,2]` 安全带。ONNX 原始 `f(x)` 注入 `u1`，plant 的 `x4'=u1-10` 只减一次 10。详见[合同审计与入口](../../../../ARCHCOMP26_TORA_REMAIN_CONTRACT_20261001.md)及[CPU-only 预检](PREFLIGHT.json)。没有重启旧作业，没有计算内容摘要。

| 新尝试 | 监督进程 wall | 小步记录 | 接受的盒步 | 首次保存 tube 越出安全带 | 首次盒拒绝 | 作者 checker |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [Huan smoke](huan_smoke1_001/RESULT.json) | 4.934499 s | 10/10 | 120/120 | 无 | 无 | 静默；仅 T=1 plumbing |
| [Xiangru smoke](xiangru_smoke1_001/RESULT.json) | 5.028029 s | 10/10 | 120/120 | 无 | 无 | 静默；仅 T=1 plumbing |
| [Huan full 尝试](huan_full20_001/RESULT.json) | 8.371523 s | 200/200 | **2357/2400** | 第 185 步 | 第 190 步 | `Unknown.`，六条 `Flow* terminated.` |
| [Xiangru full 尝试](xiangru_full20_001/RESULT.json) | 8.270158 s | 200/200 | **2357/2400** | 第 185 步 | 第 190 步 | `Unknown.`，六条 `Flow* terminated.` |

“200/200 小步记录”只表示驱动循环观察到第 200 小步；43 个盒步未获接受。两方都**没有完整全盒 T=20 安全结论或合格终点宽度**。以全部盒接受且每盒保存 tube 在安全带内为联合条件，保守前缀为 184 步，即 `t∈[0,18.4]`。第 185 步是保存的接受盒 tube 首次越出 `[-2,2]^4`，这说明区间不能证明安全，**不是实际轨迹反例**。首次盒拒绝在第 190 步；对后续幸存盒的末步 endpoint 不可冒充全初集终点。

两方已接受记录的观察前缀 tube union 在四个物理态上相同：

| 状态 | 接受盒的已观察 tube union | 安全带 |
| --- | --- | --- |
| `x1` | `[-0.9837674002258086, 0.855522141122938]` | `[-2,2]` |
| `x2` | `[-1.0100905230104735, 0.926754746809219]` | `[-2,2]` |
| `x3` | `[-1.2404441286947256, 2.0927939776780704]` | `[-2,2]` |
| `x4` | `[-2.8624570065374475, 3.441900578401031]` | `[-2,2]` |

每次运行的原始 `START.json`、`RESULT.json`、stdout/stderr、`payload/config.yaml`、`payload/START.json`、`payload/RESULT.json`、`payload/metrics.json`、`payload/observations.jsonl` 和 `payload/ranges.bin` 均保留在上表对应的独立目录。[标准库扫描器](../../../../../tools/archcomp26_scan_tora_ranges_nohash.py)对四份 `ranges.bin` 独立解码；各运行目录中的 `INDEPENDENT_INTERVAL_SCAN.json` 记录唯一 `(lane,step)`、有限有序的接受区间、首次越界与拒绝。两份 full `ranges.bin` 经直接字节比较相同，但共享驱动和数值源，不构成独立正确性证明。作者控制器浮点包络注入仍无端到端 NNCS 证书。

[原生 Flow* 新 T=20 完整运行](../native_tora_remain_full20_001/SUMMARY.md)在独立目录报告 2400/2400 范围记录与 checker `VERIFIED`。它和本页两方未完成的 8.37/8.27 s 尝试**不能据此形成完整时域速度排名**；每方也只有一次进程时间样本。
