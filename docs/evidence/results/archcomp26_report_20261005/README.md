# ARCH COMP26 当前四方比较报告

证据截至 2026-10-05；报告编制于 2026-10-06 北京时间。当前总报告统一包含 16 个实例、64 个方法格、五个新 P3 完整加速候选及每项未完成原因。

- [中文总报告](../../../ARCHCOMP26_FINAL_REPORT_DRAFT.md)
- [可编辑 Word](ARCHCOMP26_REPORT_20261005.docx) · [同版 36 页 PDF](ARCHCOMP26_REPORT_20261005.pdf)
- [当前 64 格时间](timing/current_selected_four_way.csv) · [全部逐进程时间](timing/runs.csv) · [重复进程分布](timing/campaign_statistics.csv)
- [所有状态宽度汇总](widths/summary.csv) · [全部逐步上下界与宽度](widths/widths_long.csv) · [P3 对三方宽度差值](widths/pairwise_comparisons.csv)
- [64 格具体状态和不能完成的原因](blockers/README.zh.md) · [结构化状态](blockers/status_64cells.json)
- [时间口径与来源](timing/README.md) · [宽度定义、来源和缺失](widths/README.md)

正文逐 benchmark 展开四方法各层计时、全初集有效时域、每个物理状态的共同终点 endpoint 宽度和共同前缀最大单步 tube 宽度。提前停止时另列自身末态，绝不把不同终止时刻当成可比终点。JSON/CSV 包含未保存数据的明确空值和原因。

原 289 条尝试与后续 13 个优化/资格阶段分开。覆盖仍是 38 新完整、8 同合同历史完整、14 无完整数值时域、4 离散合同阻塞；五个加速完整候选只更新已有 P3 格。时间边界、数值完成、性质标签、保存几何和独立证明严格分列。

## 复建报告

从仓库根目录读取已保存数据：

```text
python tools/build_archcomp26_timing_20261005.py
python tools/build_archcomp26_widths_20261005.py
python tools/plot_archcomp26_speed_20261005.py
python tools/build_archcomp26_report_20261005.py --docx
```

这些脚本不启动求解器或旧数值检查器。Word/PDF 生成用 Codex 文档运行环境，速度图用 Python ReportLab，既存科学图为 Python 输出。最终 PDF 由同一 DOCX 渲染导出；36 页图像均已完成视觉检查；排版与数据复核见 [QA 回执](REPORT_QA.json)。没有执行内容摘要计算或校验。

[此前正文快照](../../../ARCHCOMP26_REPORT_HISTORY_20261004.md)与[历史 Word/PDF](../archcomp26_20261001/stage_report/README.md)保留原日期资格。
