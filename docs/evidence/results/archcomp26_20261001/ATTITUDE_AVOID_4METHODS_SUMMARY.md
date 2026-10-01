# ARCH-COMP26 Attitude Control `avoid`：修正性质后的四方全盒结果

执行合同见[仓库独立审计](../../../ARCHCOMP26_ATTITUDE_CONTROL_CONTRACT_20261001.md)。固定官方 2026 `attitude_control_3_64_torch.onnx`、六维全初盒、`T=3 s`、30 控制期、每期两段。四方新运行均修正了保存的旧 checker 错误，把第四维 unsafe 约束 `-x4-0.4` 改为官方 `-x4-0.7`；旧空集性质的 `VERIFIED` 不计入本表。

| 方法 | 覆盖 | 外层进程 wall | 正确 unsafe 的本次输出 | 保存盒独立检查 |
| --- | --- | ---: | --- | --- |
| 原生 Flow* | 30/30 期、60/60 段、30 RPC | 6.281498344 s | 修正原生 checker `VERIFIED` | 60/60 六维 tube 盒分离，第四维最小间隔 0.010800050132425 |
| Huan | 30/30 期、60/60 accepted | 7.094217066 s | 作者 checker 无失败输出 | 60/60 分离，第四维最小间隔 0.010911586423864 |
| Xiangru | 30/30 期、60/60 accepted | 6.933375641 s | 作者 checker 无失败输出 | 60/60 分离，第四维最小间隔 0.010911586423864 |
| 我们 P3 | 30/30 期、60/60 accepted | 13.017036134 s | 作者 checker 无失败输出 | 60/60 分离，第四维最小间隔 0.011263404703439 |

[六态绝对边界对照 CSV](../../archcomp26_attitude_avoid_4methods_abs_bounds_20261001.csv) 从四份已保存的 `ranges.bin` 直接读出每个物理态的末端 `lo/hi/width` 和全时段 tube union `lo/hi/width`，共 24 行；检查了每份 60 条小段记录完整、有限、有序，末端被同段 tube 包含。原生记录由 C++ Flow* 导出，三个 GPU 记录由各自 P3/作者观察器导出，宽度差异须连同观察器和浮点路径解释。Huan 和 Xiangru 完整二进制范围记录经直接字节比较完全相同；两者也共享主要控制和数值路径，不能解释为两份独立证明。

各方法原始记录及详细解释：[原生](native_attitude_avoid_full30_001/SUMMARY.md)、[Huan/Xiangru](author_attitude_avoid_v1/SUMMARY.md)、[P3](p3_attitude_avoid_v1/SUMMARY.md)。四次完整运行各只有一个样本，不能据 wall 排稳定速度；保存区间的盒分离和作者 checker 输出也不是独立端到端 NNCS 浮点正确性证书。
