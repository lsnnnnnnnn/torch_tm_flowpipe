# Unicycle 论文常值速度扰动：保存时间窗内的全初盒入目标检查

只读扫描本合同四个已完成 run 的全部 500 个保存步；单初盒的每一步物理四态 **endpoint** 必须同时包含于目标 `[-0.6,0.6]×[-0.2,0.2]×[-0.06,0.06]×[-0.3,0.3]`，才记作该时刻的终点充分条件。每步的 **whole-step tube** 独立检查，不拿整段并集替代瞬时终点，也不把某坐标在不同时刻的入目标拼成一个四维时刻。步长 `0.02 s`；步号 `k` 的 endpoint 名义时刻为 `0.02k s`，tube 覆盖该小步。

| 方法 | 四维 endpoint 全入目标的保存步 | 最早充分终点 | 四维整步 tube 全入目标的保存步 | 最早整步区间 |
| --- | ---: | ---: | ---: | ---: |
| Huan | 无 | 无观察到 | 无 | 无观察到 |
| Xiangru | 无 | 无观察到 | 无 | 无观察到 |
| ours/P3 | 486–500 | `t=9.72 s` | 487–500 | `[9.72,9.74] s` |
| Flow* native | 490–500 | `t=9.80 s` | 490–500 | `[9.78,9.80] s` |

Huan/Xiangru 的保存 endpoint 中，`x1` 只有第 486–500 步的**整区间**落入对应目标，而 `x3` 只有第 216–234 步的整区间落入对应目标；所以 500 个已保存 endpoint 没有一个四态同时全入目标。其作者性质标签 `UNKNOWN` 与此一致，但**不能推出 10 秒内不可达**：保存轴盒可能过宽，步间时刻和不同时刻到达语义也不能用这些有限步终点排除。P3/原生的最早时刻早于已报告的 `T=10` 终点；这些是保存数值包络满足用户选定终点充分条件的观察，不是独立端到端浮点 NNCS 证明。原生 `VERIFIED` 仍只按作者终点 checker 口径记录。

[机器可读逐法结果](SAVED_ENDPOINT_WINDOW_AUDIT_20261002.json)由[只读复算脚本](audit_saved_endpoint_window_nohash.py)从 Huan、Xiangru、P3 各自 `payload/ranges.jsonl` 和[原生 `ranges.bin`](../native_unicycle_paper_speed_full50_001/ranges.bin)产生。脚本核对连续 500 步、作者逐步接受、原生单盒完整网格、有限有序区间、endpoint 在同小步 tube 中、最后保存 endpoint 与相邻 RESULT/既有范围扫描相同。原生二进制本身不编码 `accepted`；相邻进程 exit 0、500 段和作者 checker 日志另见[原生摘要](../native_unicycle_paper_speed_full50_001/SUMMARY.md)。没有运行求解器或内容摘要校验。
