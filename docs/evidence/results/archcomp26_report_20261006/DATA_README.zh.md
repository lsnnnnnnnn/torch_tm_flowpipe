# 10 月 6 日报告数据层

`build_current_data.py` 只读取保存的结果、范围和旧报告表，输出到本目录下的 `current/`；不运行 solver、旧 checker 或实验。旧 10 月 5 日报告、数据和实验包不改写。

运行：`python3 -B research/p3_speed_tightness_20261006/report_data/build_current_data.py`。

- `selection.json` 明确指定当前采用的 P3 结果。新结果即使完整，也不会自动取代旧结果；失败、较慢候选及未选择的数值方案仍列入 `current/RUN_INDEX.json`。
- `current/timing_index.json` 保留旧计时层，并将新行加入 `runs`。报告应读取每个 `benchmarks[instance].methods[method].current_selected_run`；不要继续使用旧 builder 的五项硬编码选择。`current_selected_four_way.csv` 保留 16 × 4 格，失败的实际耗时不能用于完整时域速度排名。
- `current/selection_audit.json` 给出每项旧、新 run 及各计时层。新实验各只有一次；process、wrapper、payload、driver 口径分别保留，共享服务器也没有独占计时资格。
- `current/summary.json` 和 `summary.csv` 保留全部物理状态。仅有完整保存输出等值证据的纯实现候选可以沿用原范围，并在每个状态行附新比较收据。较慢的 NAV standard 256 候选不自动推广。
- 修改 cutoff 或阶数的 TORA 候选从自己的 `ranges.bin` 重新解析四态、500 步、tube 与 endpoint，共 4,000 行。`*_widths_long.*`、`*_summary.json`、`*_width_statistics.csv` 独立保留；旧范围不代替新范围。只有选择 `new_saved_ranges` 才更新主宽度表。当前取舍在 `width_alternatives.json` 中记录。
- `current/widths_long.csv/json` 是完整 168,888 行整合表；从旧 CSV 流式读取，未选择数值改变的对象保留原字段及顺序。每个被选择的新 TORA 数值对象完整替换原 P3 的 4,000 行，`WIDTHS_MERGE_AUDIT.json` 记录保留、移除和新增数量。整合 JSON 沿用旧 `fields`/`rows` 紧凑结构；单候选 JSON 是带字段名的独立行列表。
- QUAD 新数值主项只在完整 1024×1000 通过、经明确选择时派生：从实际 `tube_endpoint_union_12x4` 观察记录生成 24,000 行 pooled 物理范围，并从同一次 payload 的 `final_hull` 生成 12 条独立 driver 终点对象。主汇总只用 pooled；备用终点表同时替换新 driver 对象。没有逐盒全程几何或隐藏 TM/SR 一致性证明。40 步候选不能进入完整主表。
- 旧 QUAD 比较脚本把数值宽度放在 `SAVED_COMPARISON.json`；helper 根据实际 `width_rows` 等字段分类，不因文件名而声称输出等值。缺失的配置资格用实际新旧 config 字节、仅输出路径迁移后的 driver argv，以及原来源/模型/方法字段比较补齐。控制包络候选的 driver 基础 NN 次数与额外包络计算后的实际 NN 次数分别保留。
- `current/status_64cells.json` 保留原 64 格的覆盖、失败原因、所缺证据和证明边界，更新入选 P3 的当前 RESULT/START、各层时间和实际性质字段。旧当前收据另存为历史。新数值轨迹的旧图转入 `historical_figure_paths`，当前图留空等待报告组用新范围绘制，避免冒用旧轨迹。
- `sources.json` 指向保存几何来源，`pairwise_comparisons.*` 逐状态和几何比较。不存在把不同量纲宽度相加的整体紧度评分，也不将更小宽度当成全程集合包含关系。
- `RUN_INDEX.json` 的合同证据来自比较收据；旧 NAV 收据缺少的配置/argv 项用已镜像的两份配置和 START 做直接比较，路径与结果保存在索引内。成功状态本身不证明合同一致。

报告集成时仍需保留旧 64 格阻塞、性质解释和证明边界；新保存输出等值不证明隐藏的 TM/SR 状态一致。TORA 新端点落入目标的观察不升级为完整 reach 证明。Docking 的原 UNKNOWN、具名两态 SP 与官方第三态合同缺失的区别、QUAD pooled 输出及独立 NNCS/native octagon 资格均不因提速改变。

最终选择为 20 个新阶段中的 8 个 P3 主项；失败、短前缀和未选完整方案仍保留。QUAD 主项为 `quad_paper_joint_full1000_001`：process 978.0667734360031 秒、driver 969.5491336009873 秒，完整 1024×1000，实际 NN 100 次（基础 50 次、额外包络 50 次）。其自身 24,000 条 pooled 几何与 12 条 driver 终点对象进入主表；相对原保存参考有 21,990 条变窄、2,002 条相等、8 条增宽。8 条均为 x9 的 tube/endpoint 在第 43–46 步，最大绝对增宽约 1.11013e-10；不声称所有步骤宽度非增或全程集合包含。全部 12 态终点宽度非增。trig-only 完整运行稍快但宽度不变，保留为替代方案。

QUAD 高度带后缀及 pooled/driver 差异由当前运行自身记录派生，见 `quad_paper_joint_full1000_001_geometry_observations.json`。联合与独立控制收紧运行的已保存观察/config/宽度 CSV 直接字节相同、终点对象相同的补充收据只证明这些已保存对象；不是第二次实验或隐藏状态证明，也不代替联合自身的原始来源。


正式包布局：`timing/` 保存全部计时层和当前选择，`widths/` 保存完整几何与汇总，`blockers/` 保存当前 64 格及性质来源。独立 rebuild 入口和选择文件仍在上列研究目录；本文件不代表实验重复验证。
