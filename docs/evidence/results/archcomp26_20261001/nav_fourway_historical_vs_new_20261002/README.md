# NAV 保存范围同轴比较：standard 五条、robust 四条

[standard 五条、robust 四条同轴叠加主图 PNG](nav_saved_xy_tubes_with_working_p3.png)、[PDF](nav_saved_xy_tubes_with_working_p3.pdf)；[x/y 联合宽度同轴图 PNG](nav_saved_xy_union_width_with_working_p3.png)、[PDF](nav_saved_xy_union_width_with_working_p3.pdf)。旧四方法的两份逐步汇总 CSV 各有 600 步 × 2 状态 × 4 方法 = **4,800 行**；新[当前 working P3 CSV](../nav_author_standard_working_p3_full30_001/xy_saved_curves.csv)再增加 standard 的 600 步 × 2 状态 = **1,200 行**。主图的重合反映保存数值接近；旧 Huan 和旧 Xiangru 的每一步保存上下界完全相同，必然重合。其它线的差异在约 0–3 的纵轴上很小，绘制层也会互相遮盖；**没有缺少已完成方法的曲线**。原[四方法同轴图](nav_saved_xy_tubes.png)保留为新 P3 加入前的图像截点。

[相对原生的宽度差放大辅助图 PNG](nav_each_method_width_difference_vs_native_with_working_p3.png)、[PDF](nav_each_method_width_difference_vs_native_with_working_p3.pdf)量化同轴主图难辨的差异。供逐法查看的[standard 分列图 PNG](nav_standard_each_method_xy_tubes.png)、[PDF](nav_standard_each_method_xy_tubes.pdf)和[robust 分列图 PNG](nav_robust_each_method_xy_tubes.png)、[PDF](nav_robust_each_method_xy_tubes.pdf)仅为旧四方法附图，不代替同轴主图。差值图的 0 线只表示原生宽度参照，**不是零宽流管**。

从仓库根目录重绘当前五线同轴主图和辅助图（本机 `/opt/anaconda3/bin/python` 已装 Matplotlib）：

```bash
/opt/anaconda3/bin/python tools/plot_archcomp26_nav_fourway_saved_nohash.py --root docs/evidence/results/archcomp26_20261001 --output-dir docs/evidence/results/archcomp26_20261001/nav_fourway_historical_vs_new_20261002 --working-p3-csv docs/evidence/results/archcomp26_20261001/nav_author_standard_working_p3_full30_001/xy_saved_curves.csv
```

差值图把旧 Huan/Xiangru 这条完全相同的曲线合并标注；附图则各列分别显示旧四方法来源。新图只重新绘制已保存 CSV，未为绘图重新运行旧实验。

差值图计算 `本方法 tube 联合宽度 − 同变体原生 tube 联合宽度`。在最后一步 `t=6`，standard 当前 working P3 相对新原生为 x `−0.001313132`、y `−0.000142738`；旧 ours 为 x `−0.000434515`、y `+0.000041110`；旧 Huan/Xiangru 为 x `−0.006344828`、y `−0.001473943`。robust 的旧 ours 相对旧原生为 x `+0.000378740`、y `+0.000080937`；旧 Huan/Xiangru 为 x `−0.000661962`、y `−0.000232496`。这些是**末步保存 tube 的单轴并集宽差**，不要与报告中的 `t=6` endpoint 宽度表混用；差值正负也不是严格证明或跨代性能排名。

图由仓库的 [`plot_archcomp26_nav_fourway_saved_nohash.py`](../../../../../tools/plot_archcomp26_nav_fourway_saved_nohash.py) 从两份[standard 旧四方法 CSV](../nav_standard_fourway_saved_20261002/xy_saved_curves.csv)、[robust 旧四方法 CSV](../nav_robust_fourway_saved_20261002/xy_saved_curves.csv)及[新 P3 CSV](../nav_author_standard_working_p3_full30_001/xy_saved_curves.csv)生成。附有 [MATLAB 旧四线重绘脚本](nav_saved_xy_tubes.m)，**尚不包含新 P3 第五线且未在 MATLAB 实跑**。旧四方法的源原件路径、记录数与末端区间见 [standard 来源审计](../nav_standard_fourway_saved_20261002/SOURCE_AUDIT.json)和 [robust 来源审计](../nav_robust_fourway_saved_20261002/SOURCE_AUDIT.json)；新 P3 见[独立运行扫描与说明](../nav_author_standard_working_p3_full30_001/SUMMARY.md)。

standard 五条曲线中，旧 `ours` GPU、旧 Huan、旧 Xiangru 来自 2026-09-23 已完成的同合同历史任务；原生和当前 working P3 分别来自 2026-10-02 两个新隔离全程作业。robust 四条均来自历史完整任务。旧 `ours` 的引擎为 `engine_linear_leaf_v2`，**不是当前 working P3**。旧四方法逐行读取 standard 每法 640×600、robust 每法 25×600 条保存记录，每条 152 字节；新 P3 逐行读取 640×600 条、每条 136 字节。各自独立扫描为有限、有序且同小步 endpoint 包含于 tube。曲线是每小步所有初始盒的 x 或 y 投影联合上下界；宽度图为相应绝对宽度。旧四方法逐盒 mean/max 另保留在原 CSV，新 P3 独立 CSV 也保存相同字段。

二维障碍是**联合闭盒** `[1,2]²`，单独的 x/y 投影图不能证明避障。各自独立读取原始每条保存 tube 后，standard 五方法与 robust 旧四方法均无 x/y 联合区间同该闭盒相交的记录，末端各盒 x/y 区间均处于闭目标 `[-0.5,0.5]²`。这些是已保存数值区间的性质观察，仍缺独立端到端浮点神经网络闭环证书。图的任务代际与资源不同，不支持速度排名，也不把作者 checker 的 `VERIFIED` 标签提升为独立证明。
