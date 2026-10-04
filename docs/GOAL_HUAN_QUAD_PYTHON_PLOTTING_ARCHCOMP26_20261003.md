# 持续目标：Huan QUAD、Python 可达域绘图与 ARCH-COMP26 非 VCAS 对照

更新：2026-10-04。本文接续本地 `/Users/shengenli/Downloads/GOAL_HUAN_QUAD_PLOTTING_ARCHCOMP_20260930.md`，以用户在本聊天中的较新指令为准。原文件要求 MATLAB 绘图和内容摘要校验；**当前绘图交付改为 Python，且禁止任何哈希校验及 SHA256**。原文件其余科学目标保留，但其中过时的状态、旧 SSH socket 和“从头启动”的步骤不再作为执行依据。用户随后明确要求恢复实验并完成目标，2026-10-04 回复已连接；**本轮四项交付已完成并发布，科学未决项按附件边界保留**。此前暂停的定时续跑未自动恢复。

**2026-10-04 最新明确授权：** 用户已在本聊天明确授权向自己的 GitHub（`https://github.com/lsnnnnnnnn/torch_tm_flowpipe.git`）发布，并授权使用 `shengenli@chicago.huan-zhang.com:62252` 做实验；禁止向 Huan 或 Xiangru 的 GitHub 写入。此前两项目的地授权阻断已解除，不再重复询问。新任务仍须独立目录、首次数值拒绝即停、不重跑旧实验、不做哈希校验；发布成功和实验结论只按后续原始执行收据记录。

## 目标与完成条件

1. 用冻结源码和实测解释 Huan 旧作者合同的 QUAD 约 80 秒、strict/parity 的数学与执行差别；只移植有明确正确性条件和端到端证据的优化，失败和模式变化单列。
2. 将 **Python** 可达域绘图作为直接可用的工程功能：从保存的 flowpipe/范围和几何数据生成可查看的 PNG、矢量 PDF 与机器可读 JSON/CSV。画出 initial box、按各 benchmark 时刻定义的 Safe/Target/Unsafe 区域、tube 或 endpoint，并在一张同轴图中对比多方法；必要的差值放大图另列。缺失小步和早停后的时段不得插值。新工作不制作或测试 MATLAB 图；三个正式绘图 CLI 已取消自动 `.m` 副产物，现有保存数据验收见下文。
3. 按 ARCH-COMP26 AINNCS 非 VCAS 的 16 个已识别实例/变体，对 P3、Huan、Xiangru、原生 Flow* 四方保留匹配合同的数值时域、性质、绝对宽度、过程时间和图。源缺失或数值失败要以原始收据说明，不捏造完整结果或排名。
4. 维护中文可审阅报告、可编辑 DOCX、PDF、原始数据索引及 Python 出图脚本，并推送到现有执行分支。完整数值时域、作者 checker 标签和独立端到端浮点 NNCS 证明必须分开。

本轮四项交付已达到可审阅状态，报告与原始证据已发布并逐文件直接读回；见[交付核对](ARCHCOMP26_DELIVERY_CHECKLIST_20261004.md)和[发布收据](evidence/archcomp26_publication_receipt_20261004.json)。这不把短前缀升级为完整时域，不把合同缺件改写为官方复现，也不宣称独立 NNCS 证明。

## 接续入口与不可更改约束

- 当前执行仓库：`/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_latest_20260930/repo`，分支 [`codex/huan-plot-archcomp-execution-20260930`](https://github.com/lsnnnnnnnn/torch_tm_flowpipe/tree/codex/huan-plot-archcomp-execution-20260930)。原始交接分支 `codex/gpu-verified-handoff-20260930` 保留为来源，不要在上面重复实验。
- 先读[进度](GOAL_EXECUTION_PROGRESS_20261001.md)、[本轮原始尝试索引](evidence/archcomp26_nohash_attempts_20261001.json)、[16×4 矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)、[历史覆盖附表](evidence/archcomp26_coverage_overlay_20261002.md)及[原生八方向生产门](ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)。每次先只读核对 Git 分支、服务器进程、旧作业 RESULT 和资源；不得重启已完成或运行中的原实验。新诊断用独立 ID，首次数值拒绝即保存前缀并停止该队列。
- **不做任何哈希校验、SHA256 或内容摘要计算**，包括旧 goal 中的校验建议。证据以冻结路径、源码/配置、原始 START/RESULT、日志、范围与直接数值/字节比较追溯；不把直接一致当成独立正确性证明。
- SSH 服务器为 `shengenli@chicago.huan-zhang.com:62252`，当前复用 socket `/private/tmp/codex-huan-2252-20261003-resume.sock`。先检查连接；失效且确需远端动作时再请用户挂接，不读取私钥或猜口令。服务器研究根为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`。
- 2026 QUAD 新主表使用论文方程；旧约 80 秒速度研究单列作者旧方程。TORA reach 新主表采用官方 2026 模型和 `u=11f`；Unicycle 采用论文方程，`w` 只加速度导数且每轨迹常值。不能把不同动力学、初盒、扰动或性质的结果并为同一排名。

## 已核对状态（2026-10-04 接续；benchmark 仍为 289 条）

- 本轮索引已到 **289 条尝试**；64 个方法格为 **38 个新完整数值时域、8 个同合同历史完整、14 个无完整时域、4 个 Airplane discrete 合同阻塞**，主表没有运行中作业。这里的完整仅指选定合同下数值时域完整。287 条截点的 23 页 DOCX/Word PDF 保持历史快照；[2026-10-04 的 36 页 DOCX/PDF](evidence/results/archcomp26_20261001/stage_report/README.md)已与当前 Markdown 的 289 条截点同步，16 小节、20 图、11 表全部页面校对通过。
- 2026 论文方程 QUAD 四方法已有完整 1024 盒×1000 小步数值收据；旧作者 QUAD 的 128/256 行同包装 P3 全程候选也已完成，均无需重跑。原生 Flow* 八方向**生产门关闭**：原库在放松控制的首步有数值漏包；复制库的有限修补与条件性首步检查不足以证明真实 NN/CROWN、后续控制、内部浮点和全时性质。
- Python 绘图已有[工程入口与说明](flowpipe_plot_nohash.md)、保存几何及多方法叠加的 PNG/PDF，均由 Matplotlib 直接渲染。三个正式 CLI 已取消自动 `.m`；[保存数据验收](evidence/results/python_plot_cli_cleanup_20261004_001/README.md)覆盖五条实际路径、四项针对测试和六组 PNG/PDF，未启动求解器，历史输入直接字节一致。NAV 同轴曲线相近是已核查的保存数值相似；差值放大图与分列图是辅助视图。旧 QUAD 轴对齐范围只能画 box；真实 octagon 要有接受段的相关 Taylor 模型方向界，且原生生产门仍关闭。
- TORA reach-sigmoid 的四方法 24 次有效计时已完成。TORA remain 固定 `h=0.1` 的 Huan/Xiangru 主表仍早停；`h=0.05` 四方是另列完整补充变体。新隔离追踪只证实 Huan `x2±0.02` 补充设置在第 192 步初盒 2 的 `x2` Picard 提议 `[-0.03005730939678987,0.028837305588174926]` 越过 `[-0.02,0.02]`，保存安全前缀止于第 184 步；它不是主表修复或 `T=20` 完整结果。
- Airplane continuous 的 64 个二分初盒各接受首个 `h=0.01` 数值小步，其中 8 盒首步 SAFE、56 盒性质 Unknown。单独高角盒首控制期诊断只接受 4/10 小步，第 5 步 x/y Picard 首拒；无完整 `0.1 s` 控制期或 `T=2 s` 流管。Double Pendulum more P3 全时请求在 72/80 小步因作者 checker `Unsafe.` 早停，不等于完整流管或独立真实轨迹反例。

## Python 绘图验收

- 从原始保存范围或流式方向支持出发，保存独立几何数据，再由 Python 重画；记录几何归约、绘图和数值求解的不同时间边界。图例标明方法、合同、状态维和单位，以及 Safe/Target/Unsafe 的实际时间语义。
- 至少保留 QUAD 时间–高度和一张状态–状态图。多方法同轴主图要共用坐标尺度，已核对的重合曲线仍保留；附图可显示小差值。只显示共同有效前缀，缺值和失败段显式留空。
- box 图不冒充 octagon；只有从相关 Taylor 模型直接计算并通过包含门禁的八方向支持界才可标 octagon。现有原生门禁失败不能用绘图程序遮盖。
- 用 Python 读取保存数据重建 PNG/PDF 与 JSON/CSV，检查初盒、区域、时间、数据点数和图层；正式出图入口不再自动生成 `.m`。**MATLAB/Octave 是否安装、`.m` 是否运行均不影响本目标验收**。

## 保留的科学边界与后续研究条件

- Airplane discrete 缺参与者权威的离散转移及控制更新顺序；此前已向用户提出具名 Euler 补充合同选择，未获答复前不猜主合同。Single Pendulum 官方第三态初值/重置/MATLAB 闭环入口、Balancing 论文五特征控制器或权威映射、QUAD 全时 reach-and-remain checker 均缺，必须保留各自身份与空格。
- 本地分支已存在 2026-10-03 10:44:16+08 的提交《Document paused ARCH-COMP26 handoff and 289th TORA trace》，原交接中“289 条未提交、部分暂存”的描述已过时；本轮远端内容已由官方 API 发布并逐文件直接读回。本次已读取全部 289 条远端原始 RESULT，并直接比对 223 份有显式本地 RESULT 路径的收据一致；其余 66 条不能据缺该字段推断文件丢失。2026-10-04 检查未见匹配研究进程。本地源码清理、保存图补齐、64 格交付索引和新版 36 页报告已完成验收，[交付核对](ARCHCOMP26_DELIVERY_CHECKLIST_20261004.md)逐项列证；用户已明确授权本人 GitHub；174 个交付文件已通过官方 API 发布到现有分支并逐文件直接读回，原始收据见[发布记录](evidence/archcomp26_publication_receipt_20261004.json)；已完成的 QUAD、TORA sigmoid、Unicycle、Airplane 首步与已拒绝诊断不重跑。对仍无解的格提供首拒证据与所需外部材料，而非改短时域、缩小初盒或隐藏失败。
- [全盒余项离线计划](evidence/results/archcomp26_20261001/native_quad_allbox_adaptive_remainder_plan_20261004_001/README.md)已对 1,024 盒、3,072 输出行完成 196,608 个精确有理数顶点检查，仍以原始实数仿射 CROWN 不等式为前提，生产门关闭。用户明确授权后，[新目录原生回放](evidence/results/archcomp26_20261001/native_quad_allbox_adaptive_remainder_replay_20261004_001/README.md)仅运行一次：1,024 盒、3,072 行、196,608 顶点全部通过，build/run/check/wrapper 均 exit 0，原始收据和数据已取回并直接字节比对。该结论仅针对保存首批的条件性控制构造，不扩张为 NN/CROWN、plant 或全时证明，不增加 benchmark 计数。
- 将来若恢复定时续跑，只在有实质进展、失败或需要用户操作时通知；状态不变时保持安静。
