# TORA remain `h=0.1`：仅放宽 `x2` 余项初猜的隔离 profile

[第 190 步只读首拒追踪](../author_tora_remain_h01_refusal_trace_20261002_001/README.md)显示：原 Huan `h=0.1`、order 3、统一余项初猜 `[-0.01,0.01]` 时，初盒 2 的 `x2` Picard 提议超出该初猜，而其他盒/分量在该步通过。这里的[新入口](profile_driver.py)只把 `x2` 初猜改为 `[-0.02,0.02]`；另外五维仍为 `[-0.01,0.01]`。官方 ONNX、12 盒初集、ODE、1 秒控制保持、`T=20`、全时 `[-2,2]^4` 性质、`h=0.1`、order 3、cutoff、SR queue 和 CROWN/strict 设置均保持。逐字段之外的实际 YAML 文本比较见[审计](AUDIT.json)。这是**数值参数补充变体**，不替换冻结主表的统一余项 `h=0.1` 结果。

| Huan 新 ID | 原始结果 | 独立范围扫描 |
| --- | --- | --- |
| [一期 smoke](huan_smoke1_firstreject_001/payload/RESULT.json) | `T=1`，10/10 小步、120/120 盒步接受，保存 tube 在安全盒内；作者 checker 没有明确文字输出 | [120 条记录](huan_smoke1_firstreject_001/INDEPENDENT_INTERVAL_SCAN.json)，有限有序，同段 endpoint 在 tube 内 |
| [全程首拒即停](huan_full20_firstreject_001/payload/RESULT.json) | 第 192 步初盒 2 再次 `FAILED_CONTRACTION`，只观察 192/200 小步、接受 2303/2304 盒步；未到 `T=20`，checker 没有返回性质结论 | [2304 条记录](huan_full20_firstreject_001/INDEPENDENT_INTERVAL_SCAN.json)，已接受记录有限有序、同段 endpoint 在 tube 内；保存 tube 从第 185 步越出安全盒，全盒接受且保存安全的前缀仍只有 184 步 |

原始外层 [`START`](huan_full20_firstreject_001/START.json) 与 [`RESULT`](huan_full20_firstreject_001/RESULT.json)、stdout/stderr、实际配置、每步观察和 `ranges.bin` 均原样保存；[事前意图](INTENT.json)列出两个新 run ID 和失败首拒规则。`x2` 加宽使首次数值拒绝从原第 190 步推迟到第 192 步，**没有取得完整 `h=0.1` 数值时域**，且保存盒安全前缀没有延长。因此没有启动同设置 Xiangru 镜像作业，也没有进一步猜测余项大小。另立的 [`h=0.05` 四方完整补充对照](../tora_remain_h005_fourway_saved_20261002/SUMMARY.md)仍是当前同数学合同的完整数值时域路径，但数值设置不同；保存盒检查不能提升成独立端到端 NNCS 浮点证明。这里的区间越带亦不是实际轨迹反例。
