# TORA reach-tanh：当前 P3 四方保存图（2026-10-04）

[主图 PNG](tora_tanh_current_fourway.png)和[矢量 PDF](tora_tanh_current_fourway.pdf)采用 **当前 working P3＋同合同历史 Huan/Xiangru/native**。左列将四方 x1/x2 的 500 个整步 tube 放在同一坐标轴，右列放大 `T=5` endpoint 下界/上界相对 Huan 的差异。Huan/Xiangru 界重合，仍保留两个方法图例。

[P3 代际附图](tora_tanh_p3_generations.png)和[PDF](tora_tanh_p3_generations.pdf)只比较当前 working P3 与旧 `engine_linear_leaf_v2` P3；旧 P3 不占四方法主格。下排给逐段绝对 tube 宽度差，不作时间推断。

合同是官方 ReLU³/tanh 控制器、`u=11f`、完整四态初盒、10 个 0.5 秒控制期，500 个 0.01 秒小步。initial 只显示在 `t=0`，目标 x1∈[-0.1,0.2]、x2∈[-0.9,-0.6] 只显示在 `T=5`。保存终点入目标是“5 秒内到达”的充分数值观察，不能把紫线误读为全时安全带。状态单位未在保存绘图合同中声明，不作换算。

[geometry.json](geometry.json)保留 5 个来源的全部四态记录；[saved_bounds.csv](saved_bounds.csv)共有 **10,000 行**（5×500×4）。[absolute_widths.csv](absolute_widths.csv)有 20 行，分别列四态终点、末段 tube 与全时 tube union 的上下界及绝对宽度。本例每方法只有一个初盒，单步每盒宽度 mean/max 均等于该步 union 宽度。

本次只读取原始 `ranges.bin` 和现存 START/RESULT/AUDIT。复用旧脚本的只读解析器，未调用其主函数或遗留导出器；原始运行与旧图均未覆盖。检查各源 500 个顺序记录、`h=0.01`、有限有序界、endpoint 包含于同段 tube，缺步直接拒绝。主图 2000 条范围记录，历史 P3 附图再用 500 条；没有外推或补画。源路径/大小、点数、时间/图层资格及派生时间见 [AUDIT.json](AUDIT.json)。主图与代际附图已目视检查；全部 PNG/PDF 使用 Matplotlib，本目录没有 `.m`。

当前 P3 与历史 GPU 原始收据均没有作者性质 verdict；历史 native 作者终点 checker 为 `VERIFIED`。这些保存坐标盒和标签不构成独立端到端浮点 NNCS 证书，不提供混合代际速度排名。

从仓库根目录向新的输出目录再生：

```bash
MPLCONFIGDIR=/private/tmp/archcomp26_matplotlib_cache /opt/anaconda3/bin/python -B tools/plot_archcomp26_tora_tanh_current_saved_nohash.py --root docs/evidence/results/archcomp26_20261001 --output-dir /private/tmp/tora_tanh_current_p3_redraw_20261004
```
