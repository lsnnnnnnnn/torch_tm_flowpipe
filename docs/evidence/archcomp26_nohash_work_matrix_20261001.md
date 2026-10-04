# ARCH-COMP26 无哈希工作状态矩阵（2026-10-01）

来源：`benchmarks/archcomp26/manifest.json` 与 `docs/evidence/archcomp26_nohash_attempts_20261001.json`。仅按路径归属，不做内容摘要；与历史冻结的 `benchmarks/archcomp26/execution_matrix.json` 独立。

16 个实例 × 4 种方法 = 64 个单元；本轮索引 289 条尝试。完成 38；运行中 0；早停 10；失败 4；仅短前缀 3；未尝试 9。

| 实例 | P3 GPU | Huan | Xiangru | FlowStar native |
| --- | --- | --- | --- | --- |
| acc-safe-distance | 完成 | 完成 | 完成 | 完成 |
| airplane-continuous | 失败 | 失败 | 失败 | 失败 |
| airplane-discrete | 未尝试 | 未尝试 | 未尝试 | 未尝试 |
| attitude-control-avoid | 完成 | 完成 | 完成 | 完成 |
| balancing-reach | 早停 | 早停 | 早停 | 早停 |
| docking-constraint | 完成 | 完成 | 完成 | 完成 |
| double-pendulum-less-robust | 完成 | 完成 | 完成 | 完成 |
| double-pendulum-more-robust | 早停 | 早停 | 早停 | 早停 |
| nav-standard | 完成 | 仅短前缀 | 未尝试 | 完成 |
| nav-robust | 完成 | 仅短前缀 | 未尝试 | 未尝试 |
| quad-reach | 完成 | 完成 | 完成 | 完成 |
| single-pendulum-reach | 完成 | 完成 | 完成 | 完成 |
| tora-remain | 完成 | 早停 | 早停 | 完成 |
| tora-reach-sigmoid | 完成 | 完成 | 完成 | 完成 |
| tora-reach-tanh | 完成 | 仅短前缀 | 未尝试 | 未尝试 |
| unicycle-reach | 完成 | 完成 | 完成 | 完成 |

“完成”仅表示所选尝试记录了完整数值时域，包括 Docking 原生数值完成但性质 UNKNOWN/外层 exit 2 的独立状态；“早停”包含 DP more 原生的 UNKNOWN、Huan/Xiangru 的 checker Unsafe、P3 新 affine-split4 在第 72/80 小步后 checker Unsafe，以及 Balancing raw4 Huan/P3 各自的数值拒绝。P3 原区间控制余项在第 9 小步的数值拒绝仍保留在索引。“失败”包括 P3 DP less 诊断和 Airplane 完整初盒首步入口/资源失败。“仅短前缀”仍不覆盖完整时域，例如 TORA reach-tanh Huan 官方文件 profile 的本轮新诊断仅有完整初盒一期、50 小步；同合同旧四方全程结果另按历史证据审计，未重复运行。较早 smoke 和各尝试原始状态保存在 JSON。
Balancing Huan/Xiangru/P3/原生的早停属于明确命名的固定仓库四原态 `balancing-fixed-repo-raw4` profile。Huan/Xiangru/P3 分别在第 99/99/87 小步拒绝；原生另立修正入口在第 21 周期前三段后、第 84 小步返回 `UNCOMPLETED_SAFE`，保存 83 段。Xiangru 已接受步的保存 endpoint 在 189 处超出同小步 tube 至多 2.49e-14；论文五特征控制器仍缺，不可把这些状态当作论文主合同结果。Airplane P3 回调 trace 把原数值 profile 的首步拒绝定位为自映射收缩失败；仅放宽 x/y/z 余项初猜的 profile 首步引擎接受后，观察器因 endpoint 越出 tube 1–2 个 binary64 相邻值而拒绝，0 个可用保存步，也无全时安全结论。NAV standard 当前 P3 与原生各自有独立 640 盒×600 小步全程运行和扫描，NAV robust 当前 P3 另有独立 25 盒×600 小步全程运行和扫描；旧 NAV `ours` 是另一代引擎，其同合同记录只作历史审计，不并入新 attempt。
TORA reach-tanh 当前 working P3 已另立 0.5 秒首周期门检和 500 步完整数值作业；旧 `engine_linear_leaf_v2` P3 同合同全程仅保留作历史对照，不填本轮新 attempt。作者 Huan 的新一期仍仅为短前缀，其旧同合同全程另列历史证据。
TORA remain 新的 Huan 第 190 步只读追踪、仅改 x2 余项初猜的第 192 步早停及新 ID 的第 192 步只读首拒追踪均作为补充诊断登记，`matrix_eligible=false`；后者在初盒 2 的 x2 提议超出 ±0.02，保存的 2304 条范围与先前补充变体直接相同。本格仍选择原 `h=0.1` 统一余项全时尝试。Airplane 二分 64 盒各保存一个数值接受的 `h=0.01` 小步：旧两角盒加本轮 62 个独立子进程，首步 8 盒 SAFE、56 盒性质 Unknown；新 62 条均 `matrix_eligible=false`。另立高角盒首个 0.1 秒常值控制期诊断在第 5 小步数值拒绝，仅保存前 4 段，也 `matrix_eligible=false`。这些短前缀无完整控制周期或 `T=2` 结果，不能替代完整初盒失败格。

所有 64 个单元在本索引中均不具备四方法排名资格。重复次数按各条 attempt 计，不能从完成状态推断计时资格；保存的数值结果不构成端到端浮点 NNCS 证明。标为运行中的尝试尚无终点时间或宽度；DP more 早停没有完整 T=0.4 结果。

重算：`python3 tools/build_archcomp26_nohash_work_matrix.py`。脚本只读取上述两个 JSON，写此 JSON/CSV/Markdown，并检查 JSON/CSV 各有 64 个数据单元。
