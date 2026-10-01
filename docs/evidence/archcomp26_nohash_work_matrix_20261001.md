# ARCH-COMP26 无哈希工作状态矩阵（2026-10-01）

来源：`benchmarks/archcomp26/manifest.json` 与 `docs/evidence/archcomp26_nohash_attempts_20261001.json`。仅按路径归属，不做内容摘要；与历史冻结的 `benchmarks/archcomp26/execution_matrix.json` 独立。

16 个实例 × 4 种方法 = 64 个单元；本轮索引 79 条尝试。完成 20；运行中 1；早停 5；失败 1；未尝试 37。

| 实例 | P3 GPU | Huan | Xiangru | FlowStar native |
| --- | --- | --- | --- | --- |
| acc-safe-distance | 完成 | 完成 | 完成 | 完成 |
| airplane-continuous | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| airplane-discrete | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| attitude-control-avoid | 完成 | 完成 | 完成 | 完成 |
| balancing-reach | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| docking-constraint | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| double-pendulum-less-robust | 失败 | 完成 | 完成 | 完成 |
| double-pendulum-more-robust | 未尝试 | 早停 | 早停 | 早停 |
| nav-standard | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| nav-robust | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| quad-reach | 完成 | 完成 | 完成 | 运行中 |
| single-pendulum-reach | 完成 | 完成 | 完成 | 完成 |
| tora-remain | 完成 | 早停 | 早停 | 完成 |
| tora-reach-sigmoid | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| tora-reach-tanh | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| unicycle-reach | 未尝试 | 未尝试 | 未尝试 | 未尝试 |

“完成”仅表示所选尝试记录了完整数值时域；“早停”包含 native DP more 的 UNKNOWN 和 Huan/Xiangru DP more 的 checker Unsafe；“失败”包括 P3 DP less 诊断以及入口装载失败。较早 smoke 和各尝试原始状态保存在 JSON。

所有 64 个单元在本索引中均不具备四方法排名资格。重复次数按各条 attempt 计，不能从完成状态推断计时资格；保存的数值结果不构成端到端浮点 NNCS 证明。标为运行中的尝试尚无终点时间或宽度；DP more 早停没有完整 T=0.4 结果。

重算：`python3 tools/build_archcomp26_nohash_work_matrix.py`。脚本只读取上述两个 JSON，写此 JSON/CSV/Markdown，并检查 JSON/CSV 各有 64 个数据单元。
