# Unicycle 论文常值速度扰动：四方已保存范围对照图

[PNG](unicycle_paper_speed_fourway_saved.png) · [PDF](unicycle_paper_speed_fourway_saved.pdf) · [几何数据 JSON](unicycle_paper_speed_fourway_saved.geometry.json) · [MATLAB 绘图脚本](unicycle_paper_speed_fourway_saved.m) · [生成脚本](plot_fourway_nohash.py)

同一坐标轴上的左列叠加四法保存的全部 `500 × 0.02 s` **整步轴对齐 tube**：上为 `x1`，下为 `x3`。Huan 与 Xiangru 的这两个坐标的保存边界相同，所以曲线重合；这不代表缺失数据。洋红短线只表示 **`T=10 s` 时的目标区间**，并非全时域约束。右列独立显示四法在 `T=10 s` 的 `x3`、`x4` **endpoint 区间**，绿色为对应目标区间；右下为了看清 `x4=-0.3` 边界，仅放大其附近，并未展示完整 `[-0.3,0.3]` 横轴。

本图对应用户选定的 2026 论文动力学合同：扰动 `w` 仅加在速度导数中，且沿每条轨迹为常值。每法均为完整初始盒的一次 10 秒运行。目标四态盒为 `[-0.6,0.6] × [-0.2,0.2] × [-0.06,0.06] × [-0.3,0.3]`。图中右列只画两态；四态同时包含的结论由[全部 500 步的独立保存范围审计](../../SAVED_ENDPOINT_WINDOW_AUDIT_20261002.md)及几何 JSON 中的四态末端区间给出。

| 方法 | `T=10 s` 四态 endpoint 全入目标 | 最早保存的四态 endpoint 全入目标 | 相邻运行性质标签 |
| --- | --- | ---: | --- |
| ours/P3 | 是 | `t=9.72 s` | 保存数值充分条件；没有作者 `VERIFIED` 行 |
| Huan | 否 | 500 个保存时刻未观察到 | `UNKNOWN` |
| Xiangru | 否 | 500 个保存时刻未观察到 | `UNKNOWN` |
| Flow* native | 是 | `t=9.80 s` | 作者终点 checker 打印 `VERIFIED` |

“否”只表示对应保存的轴盒 endpoint 没有全入目标；不能推出 10 秒内不可达。整步 tube 与 endpoint 分开，不能把不同时刻的各坐标拼成一次四态到达。P3/native 的结果是保存数值包络满足用户指定的“10 秒内到达”充分条件，不构成独立端到端浮点 NNCS 证明，也不用于速度排名。

源数据： [P3](../../p3_full50_001/payload/ranges.jsonl)、[Huan](../../huan_full50_001/payload/ranges.jsonl)、[Xiangru](../../xiangru_full50_001/payload/ranges.jsonl)、[Flow* native](../../../native_unicycle_paper_speed_full50_001/ranges.bin)。生成脚本只读这些范围，逐法复算并与[已保存审计 JSON](../../SAVED_ENDPOINT_WINDOW_AUDIT_20261002.json)一致后绘图；本次没有运行求解器，也没有执行内容摘要校验。MATLAB 脚本依赖同目录 JSON，已保存但未在 MATLAB 中运行。
