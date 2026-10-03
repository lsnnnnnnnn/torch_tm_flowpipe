# ARCH-COMP26 16×4 新尝试与历史全程覆盖附表

从[本轮工作矩阵](archcomp26_nohash_work_matrix_20261001.json)的 **225 条新尝试**生成；旧运行只在下方有原始结果与合同审计可对应时覆盖缺口，不计入新尝试。全程表示数值时域完成，不表示性质证明或速度排名资格。

**64 格：新全程 38；同合同旧全程 8；无全程 14；Airplane discrete 合同阻塞 4。**

| 实例 | P3 GPU | Huan | Xiangru | FlowStar native |
| --- | --- | --- | --- | --- |
| acc-safe-distance | 新全程 | 新全程 | 新全程 | 新全程 |
| airplane-continuous | 无全程 | 无全程 | 无全程 | 无全程 |
| airplane-discrete | 合同阻塞 | 合同阻塞 | 合同阻塞 | 合同阻塞 |
| attitude-control-avoid | 新全程 | 新全程 | 新全程 | 新全程 |
| balancing-reach | 无全程 | 无全程 | 无全程 | 无全程 |
| docking-constraint | 新全程 | 新全程 | 新全程 | 新全程 |
| double-pendulum-less-robust | 新全程 | 新全程 | 新全程 | 新全程 |
| double-pendulum-more-robust | 无全程 | 无全程 | 无全程 | 无全程 |
| nav-standard | 新全程 | 旧全程 | 旧全程 | 新全程 |
| nav-robust | 新全程 | 旧全程 | 旧全程 | 旧全程 |
| quad-reach | 新全程 | 新全程 | 新全程 | 新全程 |
| single-pendulum-reach | 新全程 | 新全程 | 新全程 | 新全程 |
| tora-remain | 新全程 | 无全程 | 无全程 | 新全程 |
| tora-reach-sigmoid | 新全程 | 新全程 | 新全程 | 新全程 |
| tora-reach-tanh | 新全程 | 旧全程 | 旧全程 | 旧全程 |
| unicycle-reach | 新全程 | 新全程 | 新全程 | 新全程 |

“无全程”保留本轮原始早停或失败，具体状态和路径见[工作矩阵](archcomp26_nohash_work_matrix_20261001.json)；“合同阻塞”仅指 Airplane discrete 四格，目前缺 2026 参与者权威离散转移及控制更新顺序，见[执行门](../ARCHCOMP26_AIRPLANE_DISCRETE_EXECUTION_GATE_20261002.md)。新全程也只在各自具名合同内成立：Single Pendulum 当前是两物理态加辅助时钟，官方第三态执行身份仍未建立。

## 8 条同合同历史全程来源

| 实例 | 方法 | 本轮状态 | 历史原始结果 | 合同审计 |
| --- | --- | --- | --- | --- |
| nav-standard | Huan | short_prefix_only | [result](../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_huan/result.json) | [审计](../ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md) |
| nav-standard | Xiangru | not_attempted | [result](../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_standard_xiangru/result.json) | [审计](../ARCHCOMP26_NAV_XIANGRU_HISTORICAL_CONTRACT_AUDIT_20261002.md) |
| nav-robust | Huan | short_prefix_only | [result](../../../../../results/archcomp_review_20260923/evidence_v2/timing_v1/nav_robust_r2_huan/result.json) | [审计](../ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md) |
| nav-robust | Xiangru | not_attempted | [result](../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/nav_robust_xiangru/result.json) | [审计](../ARCHCOMP26_NAV_XIANGRU_HISTORICAL_CONTRACT_AUDIT_20261002.md) |
| nav-robust | FlowStar native | not_attempted | [result](results/archcomp26_20261001/nav_robust_native_historical_20260923/result.json) | [审计](../ARCHCOMP26_NAV_P3_NATIVE_HISTORICAL_AUDIT_20261002.md) |
| tora-reach-tanh | Huan | short_prefix_only | [result](../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/tora_relu_tanh_huan/result.json) | [审计](results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_firstperiod_diag_001/HISTORICAL_REUSE_AUDIT.json) |
| tora-reach-tanh | Xiangru | not_attempted | [result](../../../../../results/archcomp_review_20260923/evidence_v1/suite_v1/tora_relu_tanh_xiangru/result.json) | [审计](results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_firstperiod_diag_001/HISTORICAL_REUSE_AUDIT.json) |
| tora-reach-tanh | FlowStar native | not_attempted | [result](../../../../../results/archcomp_review_20260923/evidence_v2/native_matched/tora_relu_tanh/result.json) | [审计](results/archcomp26_20261001/tora_reach_tanh_official2026_mat_u11_firstperiod_diag_001/HISTORICAL_REUSE_AUDIT.json) |

NAV 的历史合同是固定官方 point/set ONNX 加作者可执行 `[x,y,v,θ]` 顺序；作者另一仓库 Git LFS 模型与官方模型的二进制身份尚未建立。Huan/Xiangru NAV 的相应保存范围使用共享数值核心且直接逐字节相同，不能当作两份独立证明。旧 NAV `ours` 的全程记录另见[审计](../ARCHCOMP26_NAV_P3_NATIVE_HISTORICAL_AUDIT_20261002.md)，本附表的 P3 格优先采用当前工作 P3 新全程。

TORA reach-tanh 的旧四方记录只属于固定官方 ReLU³/tanh、`u=11f` 合同；其中 P3 格现优先采用另立的当前 working P3 新全程，旧 `engine_linear_leaf_v2` P3 不改名或并入新尝试。Huan、Xiangru、原生三格继续复用同合同历史全程。论文合并文字的 sigmoid 隐层是来源冲突；原生旧 `VERIFIED` 仅是作者终点 checker 标签。Airplane discrete 的两个方法外 CPU 前缀诊断也不填四方格。

重算：`python3 tools/build_archcomp26_coverage_overlay_nohash.py`。脚本只读本轮矩阵、九条旧 result 和已保存的 TORA 复用审计，不启动实验；生成本页、[CSV](archcomp26_coverage_overlay_20261002.csv)及[JSON](archcomp26_coverage_overlay_20261002.json)。
