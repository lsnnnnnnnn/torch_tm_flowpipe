# TORA remain：作者两方的数值拒绝与性质 Unknown

本页只读复核已保存的 2026-10-01 Huan / Xiangru 两次 `full20_001` 作业，没有启动实验或修改原始收据。逐项计数见[审计收据](evidence/results/archcomp26_20261001/author_tora_remain_v1/NUMERIC_FAILURE_READONLY_AUDIT_20261002.json)，原始 `RESULT.json`、`observations.jsonl`、stdout 和独立区间扫描的路径均列在该收据中。

| 两方各自的记录 | 数值 |
| --- | ---: |
| 观察到的 ODE 小步 | 200/200 |
| 接受的初盒小步 | 2357/2400 |
| 12 盒均接受的最后一步 | 189（`t=18.9`） |
| 保存的接受盒 tube 首次越出 `[-2,2]^4` | 第 185 步 |
| 首次数值拒绝 | 第 190 步，初盒编号 2 |
| 最后一步接受的初盒 | 6/12 |
| 全盒接受且保存 tube 在安全带内的共同前缀 | 184 步，`t≤18.4` |

[共享驱动](../research/gpu_verified_20260930/source/integration/crown_reach.py)第 1442–1460 行先调用 `advance_sparse`，只有 `ok=false` 且 flowpipe 状态不再是 `ACTIVE` 才将初盒标为 `broken` 并打印 `Flow* terminated.`。第 1471–1500 行随后检查安全带：不能证明区间在带内时设置 `safe_unknown`；它不会中断积分。第 1509–1541 行的提前中断条件为全部初盒 broken 或 `safe_unsafe`。第 1553–1560 行在循环结束后才依据 `safe_unknown` 打印 `Unknown.`。

两份 stdout 各有六条 `Flow* terminated.` 和一条末尾 `Unknown.`，均无 `Unsafe.`；观察记录显示第 190 步开始发生盒拒绝，直到第 200 步仍有六盒被拒。第 185 步的区间越界是性质无法证实，并非实际轨迹反例；第 190 步的盒拒绝是独立的数值积分/有效性失败。现有收据没有记录具体失败状态子类，因此不能把原因进一步定为收缩失败或除法失败。

只关闭安全带检查并不能消除 `advance_sparse` 返回的失败，也无法产生 12 盒的完整 `T=20` 数值流管。因此不启动这种重复实验。若后续需研究拒绝原因，应另建隔离、明确标成数值方法变体的诊断；不能把它替代本合同四方主表或宣称原任务安全。
