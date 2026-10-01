# 2026-10-01 保存区间的无哈希图

两图由[固定合同绘图入口](../../../../../../tools/archcomp26_plot_saved_nohash.py)只读解析本地已保存证据生成；没有重新运行数值任务、修改源证据或计算内容摘要。完整绝对来源路径、文件大小、配置位置和未内容绑定的相邻运行状态见各自 `.geometry.json` 与 `.render.json`。

| 图 | PNG / PDF | MATLAB / 数据 / 收据 |
|---|---|---|
| ACC 四方 `t–safe-distance margin` 保存 tube | [PNG](acc_four_method_t_safe_distance_margin_tube.png) · [PDF](acc_four_method_t_safe_distance_margin_tube.pdf) | [`.m`](acc_four_method_t_safe_distance_margin_tube.m) · [geometry](acc_four_method_t_safe_distance_margin_tube.geometry.json) · [render receipt](acc_four_method_t_safe_distance_margin_tube.render.json) |
| P3 论文方程 QUAD `t–x3` 1024 盒 pooled union tube | [PNG](quad_paper_p3_1024x1000_t_x3_pooled_tube.png) · [PDF](quad_paper_p3_1024x1000_t_x3_pooled_tube.pdf) | [`.m`](quad_paper_p3_1024x1000_t_x3_pooled_tube.m) · [geometry](quad_paper_p3_1024x1000_t_x3_pooled_tube.geometry.json) · [render receipt](quad_paper_p3_1024x1000_t_x3_pooled_tube.render.json) |

ACC 输入是 [native corrected-VAR `ranges.bin`](../../acc_native_var_tail_full50_001/ranges.bin)、[Huan `ranges.jsonl`](../../acc_huan_full50_001/ranges.jsonl)、[Xiangru `ranges.jsonl`](../../acc_xiangru_full50_001/ranges.jsonl)、[ours/P3 `ranges.jsonl`](../../acc_p3_full50_001/ranges.jsonl)。每一步保存 tube 盒按 `x_lead−x_ego−1.4·v_ego−10` 映成保守区间，十进制系数 `1.4` 按精确 `7/5` 处理，原始 binary64 端点先转精确有理数，结果向外取整；未恢复状态间相关性。四方 50 步最低 margin 下界依次为 **16.254753759211283**、**16.165034707564207**、**16.165034707564207**、**16.434858569716976**，均在全时 `margin≥0` 线之上。native binary 不含 accepted/status；GPU 三方 JSONL 每行 `accepted=true`。相邻运行状态未与保存区间内容绑定，图不是独立端到端 NN 证书。图源是四方每期保存 tube，不是另行导出的全时 union CSV。

QUAD 输入是[P3 完整新运行 pooled JSONL](../../quad_paper_p3_nohash_v1/full50_001/data/observations.jsonl)。1000 行各有 1024 accepted、无 rejected，逐步 `x3` 取 `tube_endpoint_union_12x4` 的 tube 两列。该值是 **1024 盒的单坐标合并区间**，不能恢复逐 lane 二维投影或 octagon。`[0.94,1.06]` 只画在 `t=5` 的终点目标处；最后一行 JSONL 的 endpoint `x3=[0.9584732146312499,1.0256989643984427]` 与运行 `RESULT.json` 的 final hull 略异，图采用 JSONL 保存原值。

两次出图均在 SHA 构造器抛错的守卫下成功。PNG 已目视检查公式、零线、终点目标、说明文字及来源没有截断；PDF 各为 1 页并由同一 Matplotlib 图对象输出。MATLAB 脚本已生成，**尚未在 MATLAB/Octave 执行**。
