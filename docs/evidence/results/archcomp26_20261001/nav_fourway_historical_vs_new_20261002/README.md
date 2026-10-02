# NAV 保存范围四方法图：历史 GPU 与新原生 standard

[x/y 流管投影 PNG](nav_saved_xy_tubes.png)、[PDF](nav_saved_xy_tubes.pdf)；[x/y 联合宽度 PNG](nav_saved_xy_union_width.png)、[PDF](nav_saved_xy_union_width.pdf)。图由仓库的 [`plot_archcomp26_nav_fourway_saved_nohash.py`](../../../../../tools/plot_archcomp26_nav_fourway_saved_nohash.py) 从两份[standard 紧凑 CSV](../nav_standard_fourway_saved_20261002/xy_saved_curves.csv)与[robust 紧凑 CSV](../nav_robust_fourway_saved_20261002/xy_saved_curves.csv)生成。附有 [MATLAB 重绘脚本](nav_saved_xy_tubes.m)；本轮未运行 MATLAB。源原件路径与每种方法的记录数、末端联合区间详见 [standard 来源审计](../nav_standard_fourway_saved_20261002/SOURCE_AUDIT.json)和 [robust 来源审计](../nav_robust_fourway_saved_20261002/SOURCE_AUDIT.json)。

四条 standard 曲线中，旧 `ours` GPU、旧 Huan、旧 Xiangru 均来自 2026-09-23 已完成的同合同历史任务；原生曲线来自 2026-10-02 唯一新 `nav_author_standard_native_full30_001`。robust 四条均来自历史完整任务。旧 `ours` 的引擎为 `engine_linear_leaf_v2`，**不是当前 working P3 的新数值实现**。各方法源文件逐行读取：standard 每法 640×600=384,000 条，robust 每法 25×600=15,000 条，二进制每条 152 字节；每条有限、有序，同小步端点包含于保存 tube。曲线是对每小步所有初始盒的 x 或 y 投影取联合上下界，宽度图显示这两个联合区间的绝对宽度；每盒平均/最大宽度另保留在 CSV。

二维障碍是**联合闭盒** `[1,2]²`，单独的 x/y 投影图不能证明避障。读取原始每条保存 tube 后，四种方法的 standard 与 robust 均无 x/y 联合区间同该闭盒相交的记录，末端各盒 x/y 区间均处于闭目标 `[-0.5,0.5]²`。这些是已保存数值区间的性质观察，仍缺独立端到端浮点神经网络闭环证书。图的任务代际与资源不同，不支持速度排名，也不把作者 checker 的 `VERIFIED` 标签提升为独立证明。
