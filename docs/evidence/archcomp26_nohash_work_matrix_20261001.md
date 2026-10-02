# ARCH-COMP26 无哈希工作状态矩阵（2026-10-01）

来源：`benchmarks/archcomp26/manifest.json` 与 `docs/evidence/archcomp26_nohash_attempts_20261001.json`。仅按路径归属，不做内容摘要；与历史冻结的 `benchmarks/archcomp26/execution_matrix.json` 独立。

16 个实例 × 4 种方法 = 64 个单元；本轮索引 112 条尝试。完成 27；运行中 0；早停 6；失败 4；仅短前缀 6；未尝试 21。

| 实例 | P3 GPU | Huan | Xiangru | FlowStar native |
| --- | --- | --- | --- | --- |
| acc-safe-distance | 完成 | 完成 | 完成 | 完成 |
| airplane-continuous | 失败 | 失败 | 失败 | 失败 |
| airplane-discrete | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| attitude-control-avoid | 完成 | 完成 | 完成 | 完成 |
| balancing-reach | 仅短前缀 | 早停 | 未尝试 | 未尝试 |
| docking-constraint | 完成 | 完成 | 完成 | 完成 |
| double-pendulum-less-robust | 完成 | 完成 | 完成 | 完成 |
| double-pendulum-more-robust | 仅短前缀 | 早停 | 早停 | 早停 |
| nav-standard | 仅短前缀 | 仅短前缀 | 未尝试 | 完成 |
| nav-robust | 未尝试 | 仅短前缀 | 未尝试 | 未尝试 |
| quad-reach | 完成 | 完成 | 完成 | 完成 |
| single-pendulum-reach | 完成 | 完成 | 完成 | 完成 |
| tora-remain | 完成 | 早停 | 早停 | 完成 |
| tora-reach-sigmoid | 未尝试 | 仅短前缀 | 未尝试 | 未尝试 |
| tora-reach-tanh | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| unicycle-reach | 未尝试 | 未尝试 | 未尝试 | 未尝试 |

“完成”仅表示所选尝试记录了完整数值时域，包括 Docking 原生数值完成但性质 UNKNOWN/外层 exit 2 的独立状态；“早停”包含 native DP more 的 UNKNOWN、Huan/Xiangru DP more 的 checker Unsafe 和 Balancing raw4 的数值拒绝；“失败”包括 P3 DP less 诊断、Airplane 完整初盒首步拒绝及入口/资源失败；“仅短前缀”未覆盖完整时域，其中 DP more P3 首周期覆盖全部 225 初盒、4 个小步但性质 Unknown，TORA reach-sigmoid Huan 官方文件 profile 覆盖完整初盒一期、50 小步但未检查 5 秒目标，Balancing P3 raw4 一期 4 小步不进入 8–10 秒性质窗。较早 smoke 和各尝试原始状态保存在 JSON。
Balancing Huan 的早停及 P3 的短前缀属于明确命名的固定仓库四原态 `balancing-fixed-repo-raw4` profile；论文五特征控制器仍缺，不可把这些状态当作论文主合同结果。Airplane P3 回调 trace 把原数值 profile 的首步拒绝定位为自映射收缩失败；另一仅放宽 x/y/z 余项初猜的 profile 在保存观察器的合并区间检查失败，具体谓词未记录，二者均无可用流管。NAV standard 当前 P3 仅有完整 640 初盒的首周期短前缀；原生新全程结果由独立 640 盒×600 小步运行和完整范围扫描支持，不从其单盒一期 smoke 推断。旧 NAV 同合同记录另作历史审计，不并入新 attempt。

所有 64 个单元在本索引中均不具备四方法排名资格。重复次数按各条 attempt 计，不能从完成状态推断计时资格；保存的数值结果不构成端到端浮点 NNCS 证明。标为运行中的尝试尚无终点时间或宽度；DP more 早停没有完整 T=0.4 结果。

重算：`python3 tools/build_archcomp26_nohash_work_matrix.py`。脚本只读取上述两个 JSON，写此 JSON/CSV/Markdown，并检查 JSON/CSV 各有 64 个数据单元。
