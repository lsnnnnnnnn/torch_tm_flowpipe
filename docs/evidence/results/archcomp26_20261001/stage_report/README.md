# ARCH-COMP26 阶段报告交付说明

本目录的 `ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx` 是可编辑稿；同名 `.pdf` 由 Microsoft Word 导出。封面和页脚标为 2026-10-02 阶段更新版。固定文件名沿用 20261001 的工作目录命名。

本版统计到 ACC 四方 24 次独立新进程 campaign，以及 TORA remain、Single Pendulum、修正危险盒的 Attitude 各自最新 P3 完整尝试：16×4 工作矩阵中有 **20 个完整方法格**。ACC 表中的后五次中位数只是共享主机下的描述性计时；不同数值引擎、观察器与启动路径不能据此排稳定速度名次。保存区间、作者 checker 输出和图不构成独立端到端浮点 NNCS 证明。论文方程 QUAD 原生完整结果未纳入本版。

主要图源为 `../plots/nohash_saved_20261001/`，包含 ACC 四方 `t–safe-distance margin` 与论文方程 QUAD P3 pooled `t–x3` 的 PNG、PDF、MATLAB `.m`、几何 JSON 和出图收据；MATLAB 脚本尚未执行。ACC 六态绝对区间宽度来自 `../acc_fourway_saved_ranges_20261001/acc_t5_endpoint_and_full_tube_wide.csv`。24 次 campaign 的逐次事件和复核口径见 `../acc_fourway_campaign_001/SUMMARY.md`。TORA P3 保存范围扫描见 `../p3_tora_remain_v1/full20_001/INDEPENDENT_INTERVAL_SCAN.json`。

可重建 DOCX 的入口是[仓库脚本](../../../../../tools/revise_archcomp26_stage_report_20261001.py)；它读取此目录的 `ARCHCOMP26_STAGE_REPORT_DRAFT_PRE_ACC_ATTITUDE_20261001.docx` 和上述已保存证据，不启动数值实验。PDF 由 Word 打开 DOCX 后“另存为 PDF”生成，避免替代排版器丢失中文字符。本版用文档技能的 `render_docx.py` 和 PDF 技能的 Poppler 路径渲染检查；Word PDF 为 16 页，逐页目视检查了文字、表格与四幅图，无截断或孤页。没有计算或校验内容摘要。
