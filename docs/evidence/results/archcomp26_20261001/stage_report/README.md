# ARCH-COMP26 阶段报告交付说明

## 当前 200 条尝试截点

[16 节可编辑 DOCX](ARCHCOMP26_FULL_STAGE_DRAFT_200_20261003.docx)与[同版 Word PDF](ARCHCOMP26_FULL_STAGE_DRAFT_200_20261003.pdf)由[本地生成脚本](build_archcomp26_full_draft_docx_200_20261003.py)从[200 条截点 Markdown 总报告](../../../../ARCHCOMP26_FINAL_REPORT_DRAFT.md)生成。64 格按证据来源为 38 格本轮新完整数值时域、8 格同合同历史完整时域、14 格无完整时域、4 格 Airplane discrete 合同阻塞。新增的 DP more P3 20 期请求因作者 checker `Unsafe.` 停于 72/80 小步，不计完整时域。

本版收入[原生 QUAD 八方向生产门](../../../../ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)的后续短程证据：隔离复制库的 1,024 源盒首个 `0.005 s` plant 小步，在保存的 CROWN 控制包络有效这一前提下通过独立区间检查；首盒第二小步另有条件性门检。[新录制的 1,024 盒 CROWN 同斜率数值回放](../native_quad_crown_same_slope_batch_20261003_001/README.md)、[float32 系数传输审计](../native_quad_crown_f32_transport_gate_20261003_001/README.md)和[逐盒条件性修正账本](../native_quad_crown_transport_correction_20261003_001/README.md)已收入报告。[隔离 C++ 校正控制构造门](../native_quad_corrected_control_construct_20261003_001/README.md)与首盒 NN 相关仿射门仍为 `UNDECIDED`；原生八方向生产门保持关闭。这些诊断不计入 200 条 benchmark 尝试，也不构成完整闭环或性质证明。

另收入[旧作者 QUAD 256 行加权图全 1,000 步候选](../../huan_quad_stage_a_40_20261001/weighted_chunk256_full1000_v1/README.md)：1,024,000/1,024,000 盒步接受，所存 1,000 步逐盒范围、接受掩码、状态和 pooled 观察与旧 P3/trig 完整参考直接逐字节相同。旧参考未保存可对照的 post-driver final NPY，内部 TM/SR 也未逐项比较；异日单次 wall 差异不构成稳定或因果加速。此候选属旧作者动力学合同，不进入 2026 论文 QUAD 主表。[Single Pendulum 官方第三态来源复核](../../../../ARCHCOMP26_SINGLE_PENDULUM_THIRD_STATE_SOURCE_FOLLOWUP_20261003.md)也已纳入本版；具名两物理态四方结果不能冒充该官方第三态实例。

PDF 于 2026-10-03 经 Microsoft Word 从更新后的 DOCX 导出，为 **23 页 A4**，含 **14 幅保存数据图、10 个表格**。“最终发布门”作为完整小节排在末页；PDFium 逐页检查中文字形、图表、页码与页边界，无截断。本机 Poppler 缺 Adobe-GB1 语言包，不能用其中文字形渲染结果评价此 Word PDF。文档技能的 `render_docx.py` 也已运行，但本机 LibreOffice 输出缺中文字形，故交付版面以 Word PDF 为准。本稿仍是可审阅阶段稿；旧 189/187 截点文件继续保留。

## 当前 189 条尝试截点

[16 节完整阶段稿 DOCX](ARCHCOMP26_FULL_STAGE_DRAFT_189_20261002.docx)及[同版 PDF](ARCHCOMP26_FULL_STAGE_DRAFT_189_20261002.pdf)由[本地生成脚本](build_archcomp26_full_draft_docx_189_20261002.py)从[最新 Markdown 总报告](../../../../ARCHCOMP26_FINAL_REPORT_DRAFT.md)重建。固定覆盖为 189 条新尝试、38 格本轮完整数值时域、8 格同合同历史全程、14 格无完整时域及 4 格 Airplane discrete 合同阻塞；旧 187 条截点文件保留在下节。13 幅图均来自保存范围，MATLAB 脚本仍未实跑；“完整数值时域”不等于性质证明或稳定速度排名。

本版已纳入[原生 QUAD 符号余项短步门禁](../../../../ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)：隔离首盒/一步的冻结库保存末界漏包 CROWN 仿射余项放松集内的数值样本；该样本未证实为真实神经网络输出，不能直接判原 50 期闭环性质错误。PDF 于 2026-10-02 经 Word 从上述 DOCX 导出，为 **21 页 A4**；PDFium 逐页渲染并目视检查中文、13 幅图、表格及分页，无截断。此稿仍为可审阅阶段稿。

## 187 条尝试阶段快照

新增的[16 节完整阶段稿 DOCX](ARCHCOMP26_FULL_STAGE_DRAFT_20261002.docx)与[同版 PDF](ARCHCOMP26_FULL_STAGE_DRAFT_20261002.pdf)由本目录的[生成脚本](build_archcomp26_full_draft_docx_20261002.py)从 [Markdown 总报告](../../../../ARCHCOMP26_FINAL_REPORT_DRAFT.md)构建。两者明确标为可审阅阶段稿，固定在 187 条新尝试及 37 格新全程、9 格审计旧全程、14 格无全程、4 格 Airplane discrete 合同阻塞；不把数值全程写成性质证明或稳定速度排名。DOCX 包含 16 个实例小节和 10 幅保存数据图，包括 TORA reach-sigmoid 官方 `u=11f` 四方 500 小步 x1/x2 图。PDF 于 2026-10-02 20:06 CST 经 Word 导出，为 18 页；PDFium 逐页渲染目视核查了中文、图表和分页。PDF 已收入 [Unicycle 保存端点时间窗审计](../unicycle_paper_speed_w_constant_v1/SAVED_ENDPOINT_WINDOW_AUDIT_20261002.md)的结论；此后完成的 [TORA remain 四方 `h=0.05` 补充对照](../tora_remain_h005_fourway_saved_20261002/SUMMARY.md)与[原生八方向失败门禁](../../../../ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)见最新 Markdown 与证据目录。LibreOffice 的本机字体环境会丢失中文，故应阅读 Word 导出的 PDF。旧版两页状态报告和 105/26 长篇快照均保留。

## 较早的阶段快照

本目录的 `ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx` 是可编辑稿；同名 `.pdf` 由 Microsoft Word 导出。封面和页脚标为 2026-10-02 阶段更新版。固定文件名沿用 20261001 的工作目录命名。

新增的 [2026-10-02 当前截点 DOCX](ARCHCOMP26_STATUS_20261002.docx)和[同版 PDF](ARCHCOMP26_STATUS_20261002.pdf)是单独的两页状态报告，由[只读生成脚本](../../../../../tools/build_archcomp26_status_docx_20261002.py)从当前无哈希工作矩阵及[历史覆盖附表](../../../archcomp26_coverage_overlay_20261002.md)生成。它固定在 187 条新尝试：37 格新完整数值时域、9 格已审计的同合同旧全程、14 格无全程、4 格 Airplane discrete 合同阻塞。旧长篇阶段稿下述 105/26 数字只代表原快照。新版 DOCX 在 Word 中逐页查看；PDF 由 Word 导出，再用 PDFium 渲染两页逐页核查中文、表格和截断。两份文件均不是最终完整实验报告。

本版固定在 **105 条尝试**的索引截点：16×4 工作矩阵中有 **26 个完整数值时域方法格**。论文方程 QUAD 原生旧作业自然完成；外层 completed/exit 0，独立重扫确认 1,024,000 条保存范围覆盖 1024 盒 × 1000 小步，T=5 的 x₃ endpoint union 为 `[0.965771839016746,1.0167484756616678]`。作者 checker 打印 VERIFIED，但原生保存源码只在终时逐盒检查目标，不等于论文 reach-and-remain 全时证明。四方单次进程时间和终点区间列于正文，不据不同引擎和资源路径排稳定速度名次。

Double Pendulum less-robust 的 P3 定向仿射四分控制 residual 全程诊断完成 225 盒、20 期、100 小步；四方保存区间图收入正文。先前未分区的 P3 早停作业仍单列。DP more P3 只有全初盒首周期 900/900 盒步数值接受，性质 Unknown；TORA reach-sigmoid Huan 官方 `u=11f` 配置只有完整初盒一期前缀，终点未检，均不计完整格。Docking 四方的全时性质均为 Unknown，原生外层记录 failed/exit 2；“完整”仅说明 400 个数值段保存齐全。ACC 表中的后五次中位数只是共享主机下的描述性计时。保存区间、作者 checker 输出和图不构成独立端到端浮点 NNCS 证明。

新 Huan NAV standard/robust 各自仅有一盒一周期接受结果，已记录在[最新工作矩阵](../../../archcomp26_nohash_work_matrix_20261001.md)和[执行合同审计](../../../../ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)，没有增加本版完整数值时域方法格。其后的 NAV standard 原生完整运行、当前 P3 短前缀、Balancing P3 一期、Airplane 首拒诊断和 NAV 历史对新原生图只进入持续更新的[Markdown 总报告](../../../../ARCHCOMP26_FINAL_REPORT_DRAFT.md)与工作矩阵；DOCX/PDF 保留上述 105 条尝试的可审阅阶段快照。

主要图源为 `../plots/nohash_saved_20261001/`，包含 ACC 四方 `t–safe-distance margin`、旧 P3 QUAD pooled `t–x3` 和 Docking 四方全时约束 `q` 上界的 PNG、PDF、MATLAB `.m`、几何 JSON 和出图收据。正文 QUAD 新图来自 `../quad_paper_fourway_saved_20261002/`：native/P3 可画完整 1000 步 pooled tube，Huan/Xiangru 原始保存物只有 T=5 endpoint，图中仅标终点；目标 `[0.94,1.06]` 只适用于 T=5。新增 DP less 四方保存 tube 图在 `../dp_less_fourway_split4_20261002/`。MATLAB 脚本尚未执行。ACC 六态绝对区间宽度来自 `../acc_fourway_saved_ranges_20261001/acc_t5_endpoint_and_full_tube_wide.csv`。24 次 campaign 的逐次事件和复核口径见 `../acc_fourway_campaign_001/SUMMARY.md`。Docking 四方全程证据见 `../DOCKING_FULLBOX_3METHODS_SUMMARY.md` 和 `../native_docking_full40_001/SUMMARY.md`。Airplane 完整初盒四方入口失败分别见 `../airplane_continuous_order3_fullbox_20261002/SUMMARY.md`、`../AIRPLANE_P3_FULLBOX_SMOKES_20261002.md` 和 `../native_airplane_fullbox_smokes_20261002/SUMMARY.md`。TORA P3 保存范围扫描见 `../p3_tora_remain_v1/full20_001/INDEPENDENT_INTERVAL_SCAN.json`。

可重建 DOCX 的入口是本目录的 [`refresh_native_quad_20261002.py`](refresh_native_quad_20261002.py)；它先调用[仓库原阶段脚本](../../../../../tools/revise_archcomp26_stage_report_20261001.py)，再读取已保存的原生 QUAD `RESULT.json`、`SCAN.json` 和四方图，不启动数值实验。PDF 由 Word 打开 DOCX 后“另存为 PDF”生成，避免替代排版器丢失中文字符。本版用文档技能的 `render_docx.py` 和 PDF 技能的 Poppler 路径渲染检查；Word PDF 为 17 页，逐页目视检查了中文、表格与六幅图，无截断或孤页。没有计算或校验内容摘要。
