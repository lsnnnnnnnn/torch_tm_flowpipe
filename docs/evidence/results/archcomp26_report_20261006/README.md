# ARCH COMP26 速度优化与逐状态宽度比较

当前完整正文是 [中文报告](../../../ARCHCOMP26_FINAL_REPORT_DRAFT.md)。本版在已有功能和 Python 绘图基础上加入新速度候选及收紧实验，保留失败、较慢结果及全部证据边界。

- [可编辑 Word](ARCHCOMP26_REPORT_20261006.docx)
- [同版 PDF](ARCHCOMP26_REPORT_20261006.pdf)
- [全部 benchmark 四方法当前进程时间](timing/current_selected_four_way.csv)，[所有计时层与来源](timing/timing_index.json)
- [全部状态宽度汇总](widths/summary.csv)，[全部逐步上下界和宽度](widths/widths_long.csv)，[逐态差值与包含](widths/pairwise_comparisons.csv)
- [64 格状态和每个未完成项的原因](blockers/README.zh.md)
- [本轮候选原始结果与实现](../../../../research/p3_speed_tightness_20261006/README.md)
- [Python 生成图](figures/)，[数据重建说明](DATA_README.zh.md)

时间层级不混用，短前缀不当全程，宽度不跨单位加总。独立 NNCS 浮点证明仍未闭合。新数值宽度从新记录重算；保持输出的实现优化只继承实际直接比较过的范围。旧报告和原始实验包保持原样。报告重建不运行旧 solver、旧数值 checker 或内容摘要运算。
