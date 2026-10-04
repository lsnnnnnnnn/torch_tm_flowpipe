# 保存范围补充图 2026 年 10 月 4 日

本目录从既有原始范围重画三项报告主图，不调用求解器，不增加 benchmark 尝试。

- [Attitude 四方 t–x4](attitude_fourway_t_x4.png)及 [PDF](attitude_fourway_t_x4.pdf)：选取已有四方计时 campaign 的各自 first00 作业，每法完整 60 段。初值只画在 t=0；红带是六维危险盒的 x4 投影，不能用单轴相交决定六维性质。
- [Docking 四方 t–sx](docking_fourway_t_sx.png)及 [PDF](docking_fourway_t_sx.pdf)：每法完整 400 段。sx 初值为 [70,106]，只在 t=0 标出；径向速度安全条件耦合四态，不在该单轴上虚构 Safe 区域。四方性质仍为 UNKNOWN，另见原报告的 q 上界图。
- [论文 QUAD 四方 T=5 的 x1–x3](quad_paper_fourway_endpoint_x1_x3.png)及 [PDF](quad_paper_fourway_endpoint_x1_x3.pdf)：初盒和四方终点均为轴对齐 box，右图仅放大同一终点高度。Huan/Xiangru 终点相同；未恢复不存在的中间轨迹。P3 明确使用 driver final hull，而不是另存 observer endpoint。[0.94,1.06] 仅表示所选 T=5 高度目标，未验证全时 reach-and-remain。

[saved_geometry.csv](saved_geometry.csv)有 7,840 条逐态范围，Attitude 为 4×60×6，Docking 为 4×400×4。[absolute_widths.csv](absolute_widths.csv)有 40 条方法/状态终点和全时 tube 联合绝对界。[quad_endpoint_geometry.csv](quad_endpoint_geometry.csv)保留八条实际采用的 QUAD 坐标界及来源对象。[geometry.json](geometry.json)记录来源、字节数、图层和派生/渲染时间，没有计算内容摘要。

重建时检查固定 lane、连续小步、保存 h、区间有限有序及 JSONL 的 accepted 标记；时间图用逐段阶梯带，不跨缺失小步插值。Docking 的全部 16 个方法/状态终点与全时 tube 界同旧宽度 CSV 直接逐值一致。三张 PNG 已目检，PDF 为 Matplotlib 矢量输出；没有生成 `.m`。这些派生检查不构成独立 NNCS 证书，不改变原生 octagon 生产门。

在仓库根目录运行：

```bash
/opt/anaconda3/bin/python -B tools/plot_archcomp26_remaining_saved_nohash.py --root docs/evidence/results/archcomp26_20261001 --output-dir docs/evidence/results/archcomp26_20261001/report_saved_projections_20261004_001
```

该命令只重画本目录派生文件，历史原始运行文件不改动。二进制记录存在本身不证明 solver 接受；接受和性质资格沿用对应原始 RESULT、日志与已有独立扫描。
