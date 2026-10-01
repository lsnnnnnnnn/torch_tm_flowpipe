# 2026 TORA remain：四方保存 tube 的 `t–x4` 投影

本图从四份**新运行**的原始 `ranges.bin` 逐步读取 12 盒保存的 whole-step 区间，并在每步取 `x4` 的轴对齐并集。绿色 Safe 带 `[-2,2]` 覆盖整个 `t∈[0,20]`，黑色线段是初盒 `x4∈[0.5,0.6]`。图中曲线是保存 tube 的投影，不是单条轨迹，也不是逐盒曲线。未运行求解器或内容摘要校验。

| 方法 | 图中时间 | 保存记录与资格 |
| --- | --- | --- |
| Flow* native | `0–20`，200 小步 | 12 盒 × 200 步完整；保存 tube 全时在四态 Safe 盒内；相邻原生 checker 输出 `VERIFIED`。原始 `ranges.bin` 没有 acceptance 字段。 |
| ours/P3 | `0–20`，200 小步 | 2400/2400 盒步接受，保存 tube 全时在四态 Safe 盒内；作者 checker 没有显式最终 `VERIFIED` 行。 |
| Huan、Xiangru | **仅 `0–18.4`，184 小步** | 各自前 184 步均为 12/12 接受且保存 tube 在四态 Safe 盒内。两条保存区间曲线逐值重合。 |

Huan/Xiangru 各有 2357/2400 盒步接受；第 **185** 步（`[18.4,18.5]`）首次有保存的已接受 tube 越出 Safe 盒，第 **190** 步（`[18.9,19.0]`）首次拒绝盒，作者 checker 为 `Unknown.`。因此图在 `t=18.4` 截断这两方；灰色后段标为未知，不能从幸存盒补画全初集到 `T=20` 的合格 tube 或终点。保存区间越界不等于物理轨迹反例。

## 文件与来源

- [PNG](tora_remain_2026_fourway_t_x4_saved_tube.png)、[PDF](tora_remain_2026_fourway_t_x4_saved_tube.pdf)、[MATLAB `.m`](tora_remain_2026_fourway_t_x4_saved_tube.m)、[几何及绝对来源路径 JSON](tora_remain_2026_fourway_t_x4_saved_tube.geometry.json)。
- [只读绘图脚本](../plot_x4_nohash.py) 调用同目录的 [原始范围解析与安全扫描](../scan_saved_ranges.py)，并逐项对照 [RANGES.csv](../RANGES.csv) 的 `x4` 窗口并集；完整比较和源运行目录见 [上级摘要](../SUMMARY.md)。
- JSON 记录每份 `ranges.bin`、相邻 `START.json`/`RESULT.json` 和观察日志的绝对路径及文件字节数。配置与结果是相邻运行声明，**未与范围文件作内容绑定**。

图只展示 `x4` 投影；四态安全资格依据原始保存区间的四坐标扫描。它不构成独立端到端浮点 NNCS 证书。PDF 已渲染检查为单页，无文字、曲线或图例截断；MATLAB 脚本已核对四方段数，未在 MATLAB 中执行。
