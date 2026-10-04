# Huan QUAD / Python 绘图 / ARCH-COMP26 交接

**当前更新：2026-10-04。用户已明确恢复实验并要求完成 goal，随后回复“已连接”；goal 正在执行，旧定时续跑未自动恢复。** 2026-10-03 的暂停交接及历史结果在下文保留为时间截点；当前以[修订 goal](docs/GOAL_HUAN_QUAD_PYTHON_PLOTTING_ARCHCOMP26_20261003.md)和[进度末节](docs/GOAL_EXECUTION_PROGRESS_20261001.md)为准。

**恢复前本地与原始证据核对：** 本地执行分支已有 2026-10-03 10:44:16+08 提交《Document paused ARCH-COMP26 handoff and 289th TORA trace》，暂存区为空；恢复前仅本文件及修订 goal 各有一处陈旧状态补充，另有五个不属于交付的 Word 锁文件。原“289 条尚未提交/部分暂存”已过时；GitHub 发布状态仍未验证。全部 289 条远端原始 RESULT 已读取；223 份有显式本地路径的 RESULT 直接字节一致，另 66 条缺显式本地路径字段，不据此推断证据丢失。旧加权 128/256 行完整结果、native 六小时超时和第 289 条 TORA 首拒均已核对，未重跑。2026-10-04 05:47 UTC 进程快照未见匹配实验进程。

**当前交付：** 三个绘图 CLI 已取消自动 `.m`；[保存数据验收](docs/evidence/results/python_plot_cli_cleanup_20261004_001/README.md)、[64 格交付索引](docs/evidence/archcomp26_delivery_index_20261004.md)、补齐图表和新版 36 页 DOCX/PDF 均已通过检查。报告包含 16 小节、20 图、11 表，205 条本地链接有效。[新原生应用回放](docs/evidence/results/archcomp26_20261001/native_quad_allbox_adaptive_remainder_replay_20261004_001/README.md)在授权后仅运行一次，1,024 盒、3,072 行、196,608 顶点检查全部通过，四类退出码均 0；8 输入及 15 原始输出已与服务器直接字节比对一致。条件仍是原始实数仿射 CROWN 不等式有效；无 NN/CROWN/ODE 调用，生产门仍关闭。

**本轮发布状态：** 用户已明确授权本人 GitHub。常规 push 因网络故障及本地部分克隆缺旧对象失败，未补取或校验这些对象。官方 API 已可直连，正在将第 289 条证据及本轮交付作为现有分支上的新内容提交发布；保留本地提交和远端已有历史，不强制覆盖。发布完成只以后续原始 API 收据和逐文件直接读回为准。

**2026-10-04 最新明确授权：** 用户已在本聊天明确授权向自己的 GitHub（`https://github.com/lsnnnnnnnn/torch_tm_flowpipe.git`）发布，并授权使用 `shengenli@chicago.huan-zhang.com:62252` 做实验；禁止向 Huan 或 Xiangru 的 GitHub 写入。此前两项目的地授权阻断已解除，不再重复询问。新任务仍须独立目录、首次数值拒绝即停、不重跑旧实验、不做哈希校验；发布成功和实验结论只按后续原始执行收据记录。

## 1. 接续位置与用户已经定下的合同

- 本地仓库：`/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_latest_20260930/repo`。执行分支：[`codex/huan-plot-archcomp-execution-20260930`](https://github.com/lsnnnnnnnn/torch_tm_flowpipe/tree/codex/huan-plot-archcomp-execution-20260930)。最初给出的 `codex/gpu-verified-handoff-20260930` 是旧交接来源，不是在上面重跑实验的分支。
- **不执行任何哈希校验、SHA256 或内容摘要计算。** 老文档中保留的摘要字符串只是历史记载，不是下一对话的执行步骤。核对原始 START/RESULT、日志、配置、保存范围和直接数值/字节比较；直接相同也不是独立正确性证明。
- 绘图交付是 **Python Matplotlib 生成的 PNG/PDF，加保存的几何 JSON/CSV**。用户明确不要 MATLAB 绘图。三个正式 CLI 已取消自动 `.m`，旧目录中的遗留文件仅作归档，不参与当前交付。旧文件中要求运行 MATLAB 的文字已由用户更正。
- 2026 新 QUAD 主表按**论文方程**；约 80 秒研究保持**旧作者方程**，两者不能混成同一速度比较。TORA reach 新主表采用官方 2026 模型和 `u=11f`；Unicycle 采用论文方程，`w` 只加到速度导数且每条轨迹常值。
- 主表每格须区分完整数值时域、性质 checker 的标签、独立浮点 NNCS 证书、早停、单盒 smoke 和未尝试。不能凭一条曲线、一个终点盒或一次 wall time 宣称完整证明或稳定排名。

## 2. 2026-10-03 暂停截点及保留的科学证据

### Huan 旧作者 QUAD 速度与模式

[速度和 strict/parity 报告](docs/HUAN_QUAD_SPEED_AND_MODES.md)已有源码路径、模式差异表、阶段时间和可移植性判断。Huan parity/box/native-f64/SR 分块在 1,024 盒×1,000 小步上完成五个独立进程，进程 wall 中位 **75.250099 s**；历史作者日志是 **83.048290 s**，运行环境和源码状态不完全相同。旧 Huan strict 在第 **597** 步数值拒绝，失败进程时间不能当全程成绩。当前 P3+trig 旧合同单次全程为 **1533.752052 s**。同包装的 P3 加权图 128/256 行全程候选分别为 **1350.901441 / 1093.267186 s outer wall**，各完成 1,024,000/1,024,000 盒步；保存的 1,000 步范围、掩码和状态与旧参考直接相同。两次为顺序单样本，不能称因果或稳定加速，也不与 Huan P2/parity 直接排名。原始证据位于 [Stage A 结果目录](docs/evidence/results/huan_quad_stage_a_40_20261001/)，尤其 `weighted_chunk128_samewrapper_full1000_v1/` 和 `weighted_chunk256_full1000_v1/`。

### ARCH-COMP26 非 VCAS 四方法主表

[新尝试索引](docs/evidence/archcomp26_nohash_attempts_20261001.json)共有 **289 条本轮尝试**，不是 289 个方法格。[16×4 工作矩阵](docs/evidence/archcomp26_nohash_work_matrix_20261001.md)为 64 格：**38 完整数值时域、3 仅短前缀、10 早停、4 失败、9 未尝试、0 运行中**。[覆盖附表](docs/evidence/archcomp26_coverage_overlay_20261002.md)另外审计了同合同旧证据，互斥覆盖为 **38 新完整＋8 旧完整＋14 无完整＋4 Airplane discrete 合同阻塞**；旧 NAV 五格和 TORA reach-tanh 三格没有伪装成本轮新运行。每格的入口、原始路径、时间与状态以索引和矩阵为准，不从本段概括推断性质。

2026 论文方程 QUAD 四方法均有 1,024 盒×1,000 小步的完整**数值**收据。ACC、修正危险集 Attitude、Docking、DP less、具名两物理态 Single Pendulum、官方 `u=11f` TORA reach-sigmoid、论文方程 Unicycle 等也各有四方完整数值尝试；但合同、性质及证书资格分别见 [当前 Markdown 总报告](docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md)。TORA reach-sigmoid 另有四方法×六个进程的 24 次有效计时；其结论不是跨全部实例的稳定速度排名。Docking 四方虽数值完整，性质仍为 Unknown。

**不完整格仍保留：** Airplane continuous 四方没有 `T=2 s` 完整流管；64 个二分初盒仅各接受首个 `h=0.01` 小步（8 个首步 SAFE、56 个性质 Unknown），另立高角盒首控制期诊断仅接受 4/10 小步，随后第 5 步 x/y Picard 首拒。DP more 四方没有 `T=0.4` 完整流管；新 P3 请求在 72/80 小步因作者 checker `Unsafe.` 早停，这不是独立真实轨迹反例。Balancing 固定仓库四输入补充 profile 的四方均早停，不能冒充论文五特征主合同。TORA remain 固定 `h=0.1` 的 Huan/Xiangru 主格仍早停；`h=0.05` 四方完整只是另列补充变体。

最后一条第 **289** 条是 [TORA remain 第 192 步补充诊断](docs/evidence/results/archcomp26_20261001/author_tora_remain_h01_x2rem002_step192_trace_20261003_001/README.md)。它独立复现原 `x2±0.02` 变体的生成 YAML、192 行观察与 2,304 个 152-byte 范围记录；第 192 步初盒 2 的 `x2` Picard 提议 `[-0.03005730939678987,0.028837305588174926]` 越出初猜 `[-0.02,0.02]`，接受 2303/2304 盒步。外层 `failed/exit 1`、非超时，checker 未返回，全盒安全保存前缀止于第 184 步。它在索引中 `matrix_eligible=false`，未改变 Huan 主格，也不是 `T=20` 结果。原始 START/RESULT、日志、配置、观察、`ranges.bin`、追踪行及直接比较 [AUDIT](docs/evidence/results/archcomp26_20261001/author_tora_remain_h01_x2rem002_step192_trace_20261003_001/AUDIT.json) 均已镜像；冻结 Huan/Xiangru 引擎该路径没有可启用的自映射重试开关。

### 原生 QUAD octagon/证明门

[八方向生产门](docs/ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md)必须保持 **关闭**。原始库在普通振子和论文 QUAD 的放松控制首步存在有限样本漏包；这不是已经证实的真实 NN 轨迹反例，也不直接推翻每条旧长作业范围。隔离修补复制库在**保存 CROWN 控制包络有效的前提下**通过 1,024 个源盒的第一个 `0.005 s` plant 小步及独立区间检查。首控制期剩余步、后续控制、NN/CROWN 独立证书、传输与 Flow* 内部浮点、全时 reach-and-remain 性质都未闭合。固定 `2^-50` 余项外扩在 lane 1/output 1 首拒，不能称通用修复。现有 QUAD 图是 box 投影，不是原生已认证 octagon。

## 3. 已生成文件和使用顺序

| 用途 | 文件或目录 | 当前资格 |
| --- | --- | --- |
| 下一对话目标 | [修订 goal](docs/GOAL_HUAN_QUAD_PYTHON_PLOTTING_ARCHCOMP26_20261003.md) | 覆盖原文的 MATLAB/摘要检查要求；用户已明确恢复；保持冻结合同与失败边界 |
| 逐日事实与最新状态 | [执行进度](docs/GOAL_EXECUTION_PROGRESS_20261001.md)、[Markdown 总报告](docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md) | 均已写到第 289 条，仍是阶段性执行记录 |
| 速度与方法边界 | [Huan 报告](docs/HUAN_QUAD_SPEED_AND_MODES.md)、[原生门](docs/ARCHCOMP26_NATIVE_OCTAGON_PRODUCTION_GATE_20261002.md) | 有证据与未闭合项；不是最终独立 NNCS 证明 |
| 数值结果定位 | [289 条尝试 JSON](docs/evidence/archcomp26_nohash_attempts_20261001.json)、[矩阵 MD](docs/evidence/archcomp26_nohash_work_matrix_20261001.md)、[覆盖 MD](docs/evidence/archcomp26_coverage_overlay_20261002.md) | JSON/CSV 同目录；历史覆盖不算新尝试 |
| 本地原始证据镜像 | [结果索引](docs/evidence/results/README.md)、[本轮结果目录](docs/evidence/results/archcomp26_20261001/) | 187 个顶层结果目录、排除五个 Word 锁文件后 4,800 个文件，约 0.191 GiB；较大原始范围仍在服务器 |
| 当前可编辑报告 | [2026-10-04 DOCX](docs/evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_REPORT_20261004.docx)、[36 页 PDF](docs/evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_REPORT_20261004.pdf) | 289 条截点及后续方法诊断、Python 图补齐；全部页面与本地链接已检查 |
| 历史可编辑阶段报告 | [287 截点 DOCX](docs/evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_FULL_STAGE_DRAFT_287_20261003.docx)、[对应 23 页 PDF](docs/evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_FULL_STAGE_DRAFT_287_20261003.pdf) | 已校对的**历史快照**；第 288/289 条只在 Markdown、索引和证据中，不在此 DOCX/PDF |
| Python 绘图入口 | [保存范围 CLI](src/torch_tm_flowpipe/flowpipe_plot_nohash.py)、[旧 observer CLI](src/torch_tm_flowpipe/flowpipe_plot.py)、[方向几何 CLI](src/torch_tm_flowpipe/tm_octagon_nohash.py)、[现行说明](docs/flowpipe_plot_nohash.md) | PNG/PDF 是 Matplotlib 直接渲染；三个 CLI 已取消自动 `.m` |
| 索引与报告重建脚本 | [单格执行器](tools/run_archcomp26_nohash.py)、[矩阵构建](tools/build_archcomp26_nohash_work_matrix.py)、[历史覆盖构建](tools/build_archcomp26_coverage_overlay_nohash.py)、[287 截点 DOCX 构建](docs/evidence/results/archcomp26_20261001/stage_report/build_archcomp26_full_draft_docx_287_20261003.py) | 仅在明确恢复或修订报告时运行；旧哈希绑定入口不用于当前工作 |
| 四方图实例 | [论文 QUAD 图和 CSV](docs/evidence/results/archcomp26_20261001/quad_paper_fourway_saved_20261002/)、[NAV 同轴与差值图](docs/evidence/results/archcomp26_20261001/nav_fourway_historical_vs_new_20261002/)、[Single Pendulum 两态图](docs/evidence/results/archcomp26_20261001/sp_two_state_fourway_saved_20261002/) | 各目录有 PNG/PDF 和数据/说明；NAV 多条线相近源于已保存数值相近 |
| 其他 Python 图 | [ACC/Docking/QUAD 保存图](docs/evidence/results/archcomp26_20261001/plots/nohash_saved_20261001/)、[Unicycle 图](docs/evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/plots/fourway_saved_20261002/)、[TORA sigmoid 图](docs/evidence/results/archcomp26_20261001/tora_reach_sigmoid_u11_fourway_saved_figure_20261002/) | 只画有保存数据的时段，早停后不插值 |

2026-10-03 截点的本地 4,800 个文件中有 34 个 PNG、41 个 PDF（含报告）、73 个 CSV、1,647 个 JSON、212 个 BIN，也保留 32 个未使用的遗留 `.m`。本地证据镜像**没有**把所有服务器大文件复制到 GitHub。例如论文 QUAD 原生完整 `ranges.bin`（417,792,000 bytes）和 NAV standard 原生范围（58,368,000 bytes）仍在各自原始服务器目录；本地存有原始小收据、范围扫描和来源说明。第三方 ONNX/MAT 也未系统重发。旧目录里的 `~$*` 是 Word 临时锁文件，不是交付物。

## 4. 仍缺什么，不能猜什么

1. **Airplane discrete：** 参与者权威离散状态转移和神经网络更新顺序缺失；四格合同阻塞。曾提出具名 Euler 补充合同选择，用户尚未确认，不能把提议叫官方结果。
2. **Single Pendulum：** 当前四方是具名两物理态＋辅助时钟；缺官方第三态初值、重置和 MATLAB 闭环执行入口/checker。这里的 MATLAB 指来源模型程序，不是绘图要求。
3. **Balancing：** 论文五特征控制器或权威输入映射缺失；已有固定仓库四输入补充 profile 不能改名为论文主合同。
4. **QUAD：** 参与者全时 reach-and-remain checker、独立 NN/CROWN 与原生八方向/内部浮点闭合材料缺失；既有终点或保存后缀不替代它们。
5. **数值失败：** Airplane continuous、DP more、TORA remain `h=0.1` 等已经记录首拒/早停；没有可信的全时结果时保留空格和原始原因。冻结引擎中不存在的修复开关不可凭名称推断可用。

## 5. 下一对话的安全接续方式

先读本文件、修订 goal、进度末节、尝试索引、矩阵/覆盖和原生门，接着只读核对执行分支、本地未提交文件、远端作业 RESULT/进程及资源。交接前只读检查时复用 SSH socket 可用，未发现匹配的实验进程；新 TORA RESULT 已失败退出，旧原生 QUAD 长任务 RESULT 已完成退出。这个进程快照不能代替将来的实时核对。服务器研究根为 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`，本轮 run 根为其下 `runs/archcomp26_20261001`。SSH 用户/主机/端口为 `shengenli@chicago.huan-zhang.com:62252`，曾用复用 socket `/private/tmp/codex-huan-2252-20261003-resume.sock`；先检查是否还可用，不读取私钥或猜口令。**当前已恢复；服务器写入仍须遵守实际授权及审批结果。** 原作业、新 QUAD 四方、TORA sigmoid 计时、Unicycle 四方、Airplane 64 盒首步、已首拒的诊断均不重复启动。新工作必须独立 ID、冻结合同、保存原始收据，数值首拒即停。

2026-10-03 交接本身未启动新数值实验；2026-10-04 接续没有重跑旧实验或执行哈希校验。当前图表按既有数据生成；新原生条件回放已在明确授权后单次完成，原始收据与条件边界见上文。
