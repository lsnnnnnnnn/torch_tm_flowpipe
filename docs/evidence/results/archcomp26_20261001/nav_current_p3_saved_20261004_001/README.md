# NAV：两变体的当前 P3 四方保存图（2026-10-04）

本目录只读取已保存 CSV，不运行求解器、神经网络或新实验；旧图和原始证据未覆盖。

- [四方同轴 tube 主图](nav_saved_xy_tubes_with_working_p3.png)及[矢量 PDF](nav_saved_xy_tubes_with_working_p3.pdf)：standard 与 robust 都采用当前 working P3、历史 Huan、历史 Xiangru、各自 native。standard native 是本轮全程，robust native 是同合同历史全程。所有主图面板共用状态尺度。
- [绝对 tube 宽度图](nav_saved_xy_union_width_with_working_p3.png)及[PDF](nav_saved_xy_union_width_with_working_p3.pdf)。
- [相对 native 宽度差](nav_each_method_width_difference_vs_native_with_working_p3.png)及[PDF](nav_each_method_width_difference_vs_native_with_working_p3.pdf)另保留历史 `engine_linear_leaf_v2` 的 ours；该旧代引擎不占当前 P3 主格。Huan/Xiangru 保存界相同，差值附图合并标注，主图仍各列图例。
- `nav_standard_each_method_xy_tubes` 和 `nav_robust_each_method_xy_tubes` 是历史四方法分列附图，不代替当前 P3 主图。

[geometry.json](geometry.json)和[saved_bounds.csv](saved_bounds.csv)保存 2 个变体 × 5 条代际/方法系列 × 2 状态 × 600 小步，共 **12,000 行**；四方法主图每面板是 4×600 个完整保存段。每行包括 tube/endpoint 上下界、union 宽度和每盒宽度 mean/max。数据只涉及 x/y，不能把其余状态的统计推断出来。

黑线 initial 只放在 `t=0`；紫线目标 `[-0.5,0.5]` 只放在 `T=6 s`。灰色 `[1,2]` 阴影是全时二维联合障碍 `[1,2]²` 的**单轴投影**，单轴图不能证明避障。tube 逐段在 `[t_start,t_end]` 阶梯显示；读取器对缺步、无序时间、非有限界和 endpoint 超出同段 tube 拒绝，不插值填空。状态单位未在原绘图合同中声明，本图不擅自换算。

[AUDIT.json](AUDIT.json)记录四份源 CSV 的路径/大小、完整时间网格、图层与派生/渲染时间。实际核验了缺最后一行时读取器拒绝；主图和差值图已目视检查，PNG/PDF 均由 Matplotlib 生成。本目录没有 `.m`。相邻收据与保存区间不是独立浮点 NNCS 证明，混合代际和资源不能形成速度排名。

从仓库根目录向新的输出目录再生：

```bash
MPLCONFIGDIR=/private/tmp/archcomp26_matplotlib_cache /opt/anaconda3/bin/python -B tools/plot_archcomp26_nav_fourway_saved_nohash.py --root docs/evidence/results/archcomp26_20261001 --output-dir /private/tmp/nav_current_p3_redraw_20261004 --working-p3-csv docs/evidence/results/archcomp26_20261001/nav_author_standard_working_p3_full30_001/xy_saved_curves.csv --robust-working-p3-csv docs/evidence/results/archcomp26_20261001/nav_author_robust_working_p3_full30_001/xy_saved_curves.csv
```
