# 2026 论文方程 QUAD：四方法保存区间 `t–x3` 图

[PNG](quad_paper_fourway_t_x3_pooled_tube.png) · [PDF](quad_paper_fourway_t_x3_pooled_tube.pdf) · [MATLAB `.m`](quad_paper_fourway_t_x3_pooled_tube.m) · [几何 JSON](quad_paper_fourway_t_x3_pooled_tube.geometry.json) · [几何 CSV](quad_paper_fourway_t_x3_pooled_tube.geometry.csv) · [生成脚本](plot_saved_x3.py)

这张图只从 **已经完成的 2026 论文方程 QUAD 原作业**和其它三方法已保存的证据读取数据；生成图时没有启动求解器、控制器或新数值实验。四方合同为 `[-0.4,0.4]^6×{0}^6` 的 1,024 个初始子盒、50 个 `0.1 s` 控制期、每期 20 个 `0.005 s` ODE 小步，终点 `T=5` 目标 `x3∈[0.94,1.06]`。

| 方法 | 图上可画的数据 | `T=5` 的 `x3` 终点 union | 来源 |
|---|---|---:|---|
| Flow* native | 1,024 盒 × 1,000 步的 whole-step `x3` tube union、各步 endpoint | `[0.965771839016746, 1.0167484756616678]` | [原生 `SCAN.json`](../native_quad_paper_full50_001/SCAN.json) 和 [逐步 pooled CSV](../native_quad_paper_full50_001/pooled_x3_1000steps.csv)；原始 417,792,000 B `ranges.bin` 留在服务器 `.../native_quad_paper_full50_001/ranges.bin` |
| Huan | 仅 `T=5` endpoint | `[0.967434441417146, 1.015176258384569]` | [`parity/metrics.json`](../quad_paper_huan_full50_001/parity/metrics.json) |
| Xiangru | 仅 `T=5` endpoint | `[0.967434441417146, 1.015176258384569]` | [`data/metrics.json`](../quad_paper_xiangru_v1/full50_001/data/metrics.json) |
| ours/P3 | 1,024 盒 × 1,000 步的 pooled whole-step `x3` tube union、各步 endpoint | `[0.9584732146312499, 1.0256989643984427]` | [`data/observations.jsonl`](../quad_paper_p3_nohash_v1/full50_001/data/observations.jsonl) |

原生扫描核对 1,024,000/1,024,000 条唯一有序 `(box,step)` 记录，所有有限性、区间次序和 endpoint 包含检查通过；其全时保存 `x3` tube union 为 `[-0.40817151558335585,1.452252535346294]`。原始二进制范围没有 accepted/status 字段，完成结果与 checker 是相邻而未绑定的另份收据。P3 每行的 1,024 个 lane 均记录 `accepted`、状态 0 且无 rejected，完整 1,000 行；图取这些保存行，最后一行 endpoint 与其 driver `final_hull` 在末位有轻微差异。图不以 driver 数值替换原始 observer 数值。

Huan 服务器原作业目录只有终点 metrics 和日志等，没有逐步范围文件。Xiangru 的 1,000 行 `observations.jsonl` 仅含接受计数/状态，没有 tube/endpoint 坐标；已对服务器原目录只读列举。图中因此只把两者画在右侧终点面板，不对 `t<5` 插值或假造流管。左侧 native/P3 是各步 1,024 盒的**单坐标轴对齐区间合并**；原生 Flow* 可绘 octagon，但这里的 `ranges.bin` 只支持 box 投影，不能从图恢复 octagon 方向界或 lane 间相关性。

[12 个物理态的四方终点绝对区间与宽度 CSV](terminal_12states_fourway.csv) 从各自已保存的 `T=5` 对象只读归约；[生成脚本](../../../../../tools/archcomp26_quad_fourway_terminal_widths_nohash.py)没有重新运行任何数值方法。Huan/Xiangru 和 P3 主行取驱动最终 `final_hull`，原生取完整原始范围扫描的最后一步 endpoint union。CSV 另列 P3 最后一步观察器 endpoint 作为第五种**数据对象**，不是第五种方法：它与 P3 驱动终态在 `x5` 的单个界上相差最多约 `5.69e-6`。比较宽度时应保留对象来源，不能将最后一步观察器界和驱动终态互换。

紫色 `[0.94,1.06]` **只表示 `T=5` 终点目标**，不是全时安全带；全时 tube 穿出该带并不直接反驳终点目标。四个保存终点均在目标内，但这张图不是独立的端到端浮点 NNCS 证明，也不提供速度排名。Huan 与 Xiangru 的终点区间完全相同；两者共享主要 GPU 数学核心，不可当成两个独立正确性证明。

## 保存区间的到达后持续入带诊断

[只读重算脚本](../../../../../tools/audit_archcomp26_quad_saved_remain_nohash.py) 与 [数值收据](SAVED_REMAIN_AUDIT.json) 逐行检查已保存的 1,000 个 `x3` 小步包络。从完整 1,024 盒的 pooled 坐标区间看，native 的第 775–1000 步 tube 全在 `[0.94,1.06]` 内，名义起点为 `t=3.87 s`；P3 的第 791–1000 步全在带内，名义起点为 `t=3.95 s`。各自紧前一步上界仍超出 1.06。P3 使用独立保存的 tube 与 endpoint 的并集包络判定，因为 174/1000 步的 `x3` endpoint 比同一步 tube 多出至多 `8.88e-16`；其中 9 步位于第 791–1000 步。该并集在这 210 步仍全在带内。

这只是已有数值包络的后处理；没有取得 2026 参与者的 reach-and-remain checker，也不消除 P3 观察器端点与 tube 的末位差异，更不是独立端到端浮点 NNCS 证明。Huan/Xiangru 只保存终点区间，不能从它们的终点标签推断中间何时进入并保持目标带。

运行 `MPLCONFIGDIR=/private/tmp/archcomp26_matplotlib_cache /opt/anaconda3/bin/python3 plot_saved_x3.py` 可从上述已保存数据再生 PNG、PDF、MATLAB 脚本和几何文件。已检查图像、单页 PDF 与 2,002 行几何数据；**MATLAB/Octave 未实跑**。本目录未执行内容摘要或哈希校验。
