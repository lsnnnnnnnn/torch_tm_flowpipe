# Huan QUAD / 绘图 / ARCH-COMP26 执行进度

日期：2026-10-01。此页记录本次接续后的实际状态；完整目标仍在进行中。

## 已核对的接续点

- GitHub 交接分支为 `codex/gpu-verified-handoff-20260930`；本地已有后续执行分支 `codex/huan-plot-archcomp-execution-20260930`，包含 A 阶段报告草稿、绘图功能、16×4 新版矩阵和最终报告草稿。接续前的旧矩阵 64 个格子均未填入新实验结果；本轮新增尝试另见下方无哈希矩阵，旧成绩没有挪用。
- 服务器旧 Huan parity QUAD 完成 1,024 盒 × 1,000 步；旧 strict 首次在第 597 步拒绝。当前 P3 和 trig 版均有 1,000 步完成记录。原生 QUAD 六小时作业自然超时，只记录 600 步共同完整前缀；旧进程已退出。没有重启这些原作业。
- 用户要求本轮**不做任何哈希校验或 SHA-256**。本轮读取历史身份记录，但没有重新核验源码或预编译二进制的内容身份。一次读取远端 Git 提交号的请求被自动审批以此理由拒绝，此后未重试或变通。

## 本轮新增结果

1. 用全 1,024 初盒做了新的 Huan 40 步 parity / strict 两臂短程诊断，另做一次 parity 函数级计时。三次都是新目录、新进程，预载已有 CUDA 扩展并禁止 JIT。两臂均 `broken=0`。内部时间 parity 1.412598 s、strict 1.504988 s；进程时间 5.61 s、5.70 s。strict 的终点每盒 16 变量宽度和均值比 parity 高约 0.40%。阶段计时中，40 次 `advance_sparse` 共 1.051305 s，2 次 `crown_bounds` 共 0.316279 s。全部原始文件和计时边界见 [40 步诊断摘要](evidence/results/huan_quad_stage_a_40_20261001/SUMMARY.md)；分析见 [Huan 模式与速度报告](HUAN_QUAD_SPEED_AND_MODES.md)。这些数字不能外推为完整 5 秒的速度或性质结论。
2. 用户选定 2026 论文方程为新版 QUAD 四方主合同。论文与官方仓库/旧作者代码在 `x2'`、`x4'`、`x5'` 三处不同；旧 Huan 约 80 秒结果仍按旧作者合同解释。逐式来源和未定控制器见 [QUAD 合同决策](ARCHCOMP26_QUAD_PAPER_CONTRACT_DECISION_20261001.md)。
3. 已从论文、官方规格/动力学/控制器和旧入口核出 [Double Pendulum less-robust 的首个新版共同合同](ARCHCOMP26_DOUBLE_PENDULUM_LESS_CONTRACT_20261001.md)：四物理态、全初盒 `[1,1.3]^4`、225 子盒、20 个 0.05 s 控制期、全程四维安全带 `[-1.7,2]`。旧三套 GPU 结果只是回归线索，不能充当新版重跑。
4. 已添加 [无哈希单格执行器](../tools/run_archcomp26_nohash.py)，可独占新目录、保存命令与原始日志、拒绝重复运行并在超时时清理进程组。假子进程自检通过。现有 ARCH-COMP 预检/矩阵/报告程序依赖哈希绑定，本轮不会把无哈希数据伪装成它们的合格结果。
5. 对完成的 P3+trig 运行做了只读逐步计时审计：19 次非空图逐出步骤的混合计时合计 191.723381 s，其余 981 次合计 1.229124 s。前者包括全状态诊断签名和必要的 CUDA 同步/图逐出，不能当成净可省时间；调用链、候选改动和无哈希入口缺口见 [P3 优化审计](evidence/results/huan_quad_stage_a_40_20261001/P3_OPTIMIZATION_AUDIT_NOHASH.md)。没有改动旧数值源码或启动优化测速。

## 绘图和剩余边界

已有 QUAD 的时间—状态、状态—状态、原生保存范围的 box 投影，以及 MATLAB `.m`、PNG、PDF；见 [绘图说明](flowpipe_plotting.md) 和 [保存数据验证](FLOWPIPE_PLOT_VALIDATION_20261001.md)。本机与服务器 PATH 都没有 MATLAB/Octave，因此尚无 `.m` 实跑证据。保存的 P3 observer 只有逐坐标区间，无法恢复带相关性的 octagon；未来要在接受步保留的 Taylor 模型上导出八方向支持界，并单独验证。

新增 [无哈希绘图入口](../src/torch_tm_flowpipe/flowpipe_plot_nohash.py) 可从 native `ranges.bin` 直接生成几何 JSON、MATLAB `.m`、PNG/PDF 与只记录路径/大小的收据。其调用链在 SHA 构造器拒绝测试下通过。新完成的 DP less 原生全程数据已另存本地小型数据包并生成 [t–θ₁ tube 图](evidence/results/archcomp26_20261001/native_dp_less_full20_001/plots/dp_less_native_225x100_t_theta1_tube.png) 和 [θ₁–θ₂ endpoint 图](evidence/results/archcomp26_20261001/native_dp_less_full20_001/plots/dp_less_native_225x100_theta1_theta2_endpoint.png) 等三组。图中初盒为 `[1,1.3]^4`，Safe region 是全时 `[-1.7,2]^4`；native 范围记录本身不含 accepted/status，图上仅声明记录覆盖。

Double Pendulum 旧 native 300 秒作业超时。只读审计发现其二进制请求端口是 5200，而同目录的 server/launcher 监听 5100；旧 5625 条范围记录不能作为已配对的四方结果。已从服务器旧离线轮包在新目录提取 `gevent`、`tinyrpc` 等 RPC 依赖并通过导入 smoke。端口一致的新 native **225 盒、1 控制期 plumbing smoke** 在独立目录 `.../runs/archcomp26_20261001/native_dp_less_smoke1_001` 完成：53.211 秒、1 次 RPC、1125 条范围记录（225 盒 × 5 个 0.01 s ODE 小步），服务与求解器正常退出。这不是完整 benchmark 成绩。

随后另一新目录 `.../runs/archcomp26_20261001/native_dp_less_full20_001` 的原生 **225 盒 × 20 控制期**正式尝试已完成：进程 wall **1107.127423 s**（求解器自报 1100.101 s），20 次 RPC，22,500 条唯一且完整的 `(lane,step)` 范围记录（225 盒 × 100 小步），无超时或数值拒绝。作者 checker 报 `VERIFIED`；独立读取保存区间，四个物理态全时 tube 均位于官方 `[-1.7,2]` 安全带，未发现 NaN、逆序或 endpoint 越出 tube。完整 T=1 endpoint/tube 绝对区间、每盒宽度与计时边界见 [native DP less 摘要](evidence/results/archcomp26_20261001/native_dp_less_full20_001/SUMMARY.md)。这不构成独立的浮点 NN 证明，也不能单独填满四方比较。其它新版实例的模型、控制器和性质冲突仍需逐项冻结，不能猜测或删行。

本轮新尝试的机器可读索引是 [无哈希 attempt 记录](evidence/archcomp26_nohash_attempts_20261001.json)，与旧版摘要绑定 16×4 矩阵分开，明确短程 smoke、完整单次结果和失败预检的不同资格。

## 随后完成的新尝试

- **论文方程 QUAD / Huan**：独立新 [配置](../benchmarks/archcomp26/configs/quad_paper_p2_huan.yaml) 下，1,024 盒 × 50 控制周期、1,000 ODE 小步全部完成，`broken=0`，进程 wall 94.583 s，driver 90.471 s。作者现有终点 checker 给出 `VERIFIED`，T=5 的 `x3` union `[0.967434441417146,1.015176258384569]` 位于论文目标内。旧约75秒用的是不同动力学，不能直接作速度变化。见 [新运行摘要](evidence/results/archcomp26_20261001/quad_paper_huan_full50_001/SUMMARY.md)；reach-and-remain 的完整时间语义与独立浮点 NN 证明仍未闭合。
- **DP less / 四方实际尝试**：原生完整结果如上。新 Huan/Xiangru 共享驱动各完成 225 盒 × 20 周期、22,500 接受盒步，进程 wall 分别为 9.539174 s 和 8.387790 s；两份保存区间逐字节相同，独立扫描的 tube 与 endpoint 均在安全带内。它们共享驱动且各只有一个样本，不构成独立证明或稳定速度排名；保存 observer 的 endpoint 相对同小步 tube 有最多 `5.55e-15` 的微小越界，见 [两作者新结果](evidence/results/archcomp26_20261001/author_dp_less_v1/SUMMARY.md)。新版 P3 先用独立有向区间 NN residual 跑完整盒尝试，第 2 期首小步 225 盒全因收缩失败；更紧的有向仿射 ReLU residual 后已推进 60/100 小步，但从小步48起性质未决，并出现局部收缩失败。两者都保留为失败/部分前缀，见 [P3 诊断](ARCHCOMP26_DP_P3_DIRECTED_DIAGNOSTIC_20261001.md)。
- **DP more / 正确独立 controller**：新原生 225 盒跑到第16个控制周期，进程 wall 715.978257 s，45盒 `UNKNOWN` 后停；共同安全前缀到 `T=0.3`，无 T=0.4 结果，见 [原生摘要](evidence/results/archcomp26_20261001/native_dp_more_full20_001/SUMMARY.md) 和 [状态图](evidence/results/archcomp26_20261001/native_dp_more_full20_001/plots/dp_more_native_theta1dot_status.png)。新 Huan/Xiangru 用相同 225 盒和固定2026 more ONNX，均在第18期后由共享 checker 输出 `Unsafe.` 并早停；进程 wall 11.685967 s、11.724106 s，保存区间逐字节相同。独立区间扫描显示 `T≤0.3` 安全，步61初见未决，步72有15盒的整个保存 `θ̇₁` tube 位于安全带下方；这尚不是独立端到端浮点 NN 反例证明。见 [两作者早停摘要](evidence/results/archcomp26_20261001/author_dp_more_v1/SUMMARY.md)。

截至上节时，主要缺口是论文 QUAD 的其它三方法、新版其余 13 个非 VCAS 配置的逐项合同和四方运行、P3 严格控制器与更完整的可验证数值资格，以及正式多次计时。Single Pendulum 的第三时钟初值、Airplane 离散控制更新先后、QUAD reach-and-remain 检查的原始参与者执行源码仍未取得，已向用户具体询问；不由现有论文/README猜造执行语义。当时阶段性 DOCX/PDF 正在制作；旧哈希绑定 16×4 矩阵仍保留为恢复前快照，新数据以独立 [无哈希 attempt 索引](evidence/archcomp26_nohash_attempts_20261001.json)登记。

## 后续新证据（12:30 UTC 左右）

- **论文 QUAD / Xiangru**：另一独立新目录 `.../quad_paper_xiangru_v1/full50_001` 的 1,024 盒 × 1,000 步全部接受，进程 wall 108.018 s、driver 103.686 s，最终 16 维 hull 与新 Huan 论文合同 JSON 相同。两者共用作者数学核心和 P2 parity，不能充当彼此独立的正确性证明；各一次 wall 不能建立稳定排名。见 [Xiangru 摘要](evidence/results/archcomp26_20261001/quad_paper_xiangru_v1/SUMMARY.md)。原生论文 QUAD 在独立 `.../native_quad_paper_full50_001` 运行中；1,024 盒一期预检有 20,480 条完整范围记录，未重复启动旧六小时作业。
- **Single Pendulum / 论文两物理态 profile**：新 [配置](../benchmarks/archcomp26/configs/single_pendulum_paper_two_state.yaml) 对完整单盒、`0.05×20`、全时 `t∈[0.5,1]` 的 `0≤x1≤1` 设闭时间窗。Huan 与 Xiangru 各有一份完整 100/100 小步、`broken=0` 的新运行，wall 5.431 s / 5.529 s；两份保存区间完全相同，窗口内 `x1` tube union `[0.5663832766836561,0.9925703395905962]`。见 [SP 两方摘要](evidence/results/archcomp26_20261001/single_pendulum_prep_001/SUMMARY.md)。此 profile 把第三状态仅作辅助时钟；若要称为 2026 参与者 MATLAB 三态复现，仍缺第三初值/执行源码。新 native 仍在单独预检/正式运行。
- **TORA remain / 原生**：逐字节确认选定的固定 2026 ONNX 与服务器已有旧同名模型相同；原生隔离 build 的 12 盒一期 plumbing smoke 完成。新的 12 盒 × 20 期、200 小步正式单次原生运行在 8.339151 s 完成，20 RPC、2,400 条唯一范围记录，作者 checker `VERIFIED`。独立 [范围扫描](evidence/results/archcomp26_20261001/native_tora_remain_full20_001/SUMMARY.md) 所见全时四态 tube 都在 `[-2,2]` 内，端点也未越出同小步 tube。四方 [合同](ARCHCOMP26_TORA_REMAIN_CONTRACT_20261001.md) 已冻结；截至本节写作时 Huan/Xiangru 尚待结果，其后续终态见下方补记。此单次时间不进入四方排名。

无哈希 [16×4 工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md) 单独从新 attempt 索引重算，反映已尝试/运行中/早停/失败；旧冻结矩阵没有被改写。阶段性 [DOCX](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx) 与 [PDF](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.pdf) 已生成，仍是**未完成全套实验的阶段稿**；下一版须纳入本节新增运行。

## 21:55 本地补记：QUAD P3、SP 与 TORA 两方

- **2026 论文方程 QUAD / P3**：独立新目录 `quad_paper_p3_nohash_v1/full50_001` 完成 1,024 盒 × 1,000 小步，1,024,000/1,024,000 盒步接受，50 次控制刷新、`broken=0`，外层进程 wall 1,357.555 s；`T=5` 的 `x3` endpoint union 为 `[0.9584732146312492,1.0256989633476477]`，在目标 `[0.94,1.06]` 内。驱动打印 `VERIFIED`，但明确记录 `end_to_end_strict_certificate=false`，不能把该诊断提升为独立端到端浮点 NNCS 证书。其终点宽 0.06722574871639841，比新 Huan/Xiangru P2 parity 的 0.04774181696742297 宽；各只有一次样本、设备与路线不同，不给速度排名。见[原始证据摘要](evidence/results/archcomp26_20261001/quad_paper_p3_nohash_v1/SUMMARY.md)。该时点论文方程 native 全程作业仍在运行；最终结果见下方 10 月 2 日补记。
- **Single Pendulum / 原生补完**：两物理态加辅助时钟的原生全程已完成 1 盒 × 100 小步，wall 4.726027 s；Huan/Xiangru 分别为 5.430762 / 5.528731 s。三方在性质要求的闭时间窗 `t∈[0.5,1]` 保存的 `x1` tube 均位于 `[0,1]`。见[原生摘要](evidence/results/archcomp26_20261001/native_sp_two_state_full20_001/SUMMARY.md)和[两方摘要](evidence/results/archcomp26_20261001/single_pendulum_prep_001/SUMMARY.md)。
- **TORA remain / 作者两方新尝试**：原生已全程完成并由作者 checker 报 `VERIFIED`，进程 wall 8.339151 s、2,400/2,400 条范围，见[原生摘要](evidence/results/archcomp26_20261001/native_tora_remain_full20_001/SUMMARY.md)。Huan/Xiangru 各观察到 200 小步，但只接受 2357/2400 盒步，作者 checker 均报 `Unknown.`；首次保存 tube 出安全带在步185，首次拒绝在步190。两方全部盒接受且保存 tube 安全的共同前缀仅 `t≤18.4`。其失败进程 wall 8.371523 / 8.270158 s 不与原生完整时间排名，区间出带不是独立实际轨迹反例。见[两方摘要](evidence/results/archcomp26_20261001/author_tora_remain_v1/SUMMARY.md)。
- 新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)当前由 35 条 attempt 重算：10 个方法格有单次完整时域数值记录、1 格运行中、5 格早停或未完成。它独立于旧哈希绑定矩阵，完成格也尚不构成四方同资源、多次测量或端到端证明。阶段 [DOCX](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx) 与 [PDF](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.pdf) 已按这批事实更新并逐页检查，仍非最终完整报告。

## 继续推进：ACC 与未冻结的来源差异

- **ACC participant-order / 修复 VAR 尾项的原生 Flow\***：固定 2026 ONNX 与保存的旧同名文件直接逐字节相同；保存的两份参与者源码均明确第 5 个 NN 特征为 `v_lead-v_ego`，论文没有定义相对速度符号，故新[合同](ARCHCOMP26_ACC_PARTICIPANT_CONTRACT_20261001.md)明确命名这一输入顺序。独立一期 smoke 后，隔离修复版完成全初盒 B=1、50 控制期、T=5，50 RPC、50 条范围、作者 checker `VERIFIED`，外层 wall **7.930 s**。独立读取保存 tube 后，安全半空间 `x_lead-x_ego-1.4*v_ego-10` 的保守下界最小 **16.25475375921129>0**。见[新原生摘要](evidence/results/archcomp26_20261001/acc_native_var_tail_full50_001/SUMMARY.md)。这只证明本次数值记录与作者 checker 一致，不提升旧 VAR 修复针对性单步测试为端到端控制器证明；Huan/Xiangru 新入口仍在准备。
- [NAV 初次来源审计](ARCHCOMP26_NAV_PAPER_SOURCE_CONTRACT_20261001.md)发现论文状态文字 `(x,y,θ,ν)` 与固定官方 MATLAB 的 `(x,y,ν,θ)` 顺序冲突；当时新 NAV 数值任务尚未启动。后来找到控制器作者训练与闭环执行源码，后续合同及两次新短程尝试见[补充审计](ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)。
- [TORA reach 来源冲突](ARCHCOMP26_TORA_REACH_SOURCE_CONFLICT_20261001.md)涉及 sigmoid 最后一层和 plant 缩放；[无哈希网络预检](ARCHCOMP26_TORA_REACH_CONTROLLER_PREFLIGHT_NOHASH_20261001.md)已证实官方 2026 `.mat` 与旧 ONNX 的逐层参数、激活一致，并提供需显式选择缩放的构造器。用户的主合同选择未返回，reach 数值任务未启动。
- [Unicycle 来源冲突](ARCHCOMP26_UNICYCLE_SOURCE_CONFLICT_20261001.md)涉及扰动加入 yaw/speed 的位置与 `w` 是否随时间变化；固定 2026 ONNX 与保存版直接字节相同，但不能替代 plant 选择。[Airplane 入口审计](ARCHCOMP26_AIRPLANE_2026_ENTRY_AUDIT.md)亦已核出完整初盒和固定 ONNX，而离散更新顺序在官方目录没有执行程序；没有把旧单点连续任务或自拟 Euler 先后伪称官方离散复现。

当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)已收录 37 条新尝试：11 格单次完整数值时域、1 格仍运行的原生论文 QUAD、5 格早停、1 格失败。所有完整格的稳定多次比较和端到端浮点 NNCS 证明仍未完成。

## 后续补记：ACC 三方、Attitude checker 与 QUAD 长作业

- **ACC participant-order 三方全程**：Huan/Xiangru 各自在独立新目录完成一期 smoke 和全 50 期，均 50/50 小步接受，50 次安全评价，保存 tube 的半空间保守下界最小都是 `16.16503470756421>0`；两方全程区间记录和安全记录逐字节一致。进程 wall 分别为 7.259717 s、7.311630 s。与原生修复 VAR 尾项的 7.930 s 共同形成这个明确命名合同的三方单次完整数值记录；独立证书及稳定多次计时尚缺。见[三方证据汇总](evidence/results/archcomp26_20261001/ACC_PARTICIPANT_PROFILE_20261001.md)。我方 P3 适配正在独立评估，未填完整结果。
- **Attitude Control 来源错误**：旧 native/Xiangru checker 把官方危险盒的 `x4≥-0.7` 误写成 `x4≥-0.4`，而上界仍为 `x4≤-0.6`，所检危险集为空。旧 `VERIFIED` 不可沿用为 2026 正确规格结果；新隔离修正与预检继续进行。
- **论文方程 QUAD / native 当时状态**：只读确认独立新作业运行到第 26 个控制期；未启动第二份或重启旧六小时实验。该时点完整 T=5 结果尚未产生，最终结果见下方 10 月 2 日补记。

最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **41 条**新尝试重算：**13 格**有单次完整数值时域记录、1 格运行中、5 格早停、1 格失败，其余 44 格尚无尝试。上述先前“35 条”“37 条”是各阶段快照，不表示当前计数。

## ACC 四方数值记录闭合

- 我方 working-P3 ACC 最小适配在新独立目录先完成一期 smoke，再完成全单盒 50 期、50/50 小步接受，50 次控制特征构造/注入和 50 次安全事件。单次外层 wall **7.933729 s**，保存全时 tube 盒的安全半空间保守下界最小 **16.43485856971698>0**；见[P3 证据摘要](evidence/results/archcomp26_20261001/acc_p3_full50_001/SUMMARY.md)与[四方汇总](evidence/results/archcomp26_20261001/ACC_PARTICIPANT_PROFILE_20261001.md)。该入口复用 working-P3 数值核心与已有 CUDA 库，不能继承 QUAD 的全批次资格；用户要求的无摘要路径被保持。
- ACC participant-order 是第一个四方法均完成完整时域的实例。其“参与者顺序”指 `v_rel=v_lead-v_ego`，论文未定义符号；四条单次 wall 仅作各自进程事实，不能据此排名。四方独立端到端浮点 NNCS 证明、统一多次计时和每维完整宽度对比仍待完成。

当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **43 条**新尝试重算：**14 格**单次完整、1 格运行中、5 格早停、1 格失败，43 格尚未尝试。此前 41 条是上一阶段快照。

## Attitude 官方 unsafe 修正后的原生全程

旧 native/Xiangru 源码把 `x4` 的危险盒下界 `-0.7` 误写成 `-0.4`，结合上界 `-0.6` 形成空集；详见[合同和逐式来源](ARCHCOMP26_ATTITUDE_CONTROL_CONTRACT_20261001.md)。在独立新构建和运行目录修正这条约束后，原生先完成一期 smoke，再完成全部 30 控制期、60 个 ODE 小段、30 RPC，进程 wall **6.281498 s**。作者 checker 给 `VERIFIED`；保存的 60 个六维 tube 盒均与官方真正的 unsafe 盒不相交，`x4` 全时上界为 `-0.710800050132425`，低于危险盒下界 `-0.7`。见[保存盒扫描](evidence/results/archcomp26_20261001/native_attitude_avoid_full30_001/SCAN.json)和[危险盒独立判交](evidence/results/archcomp26_20261001/native_attitude_avoid_full30_001/BOX_UNSAFE_AUDIT.json)。旧 `VERIFIED` 不纳入新成绩；新数值结果也不声称独立端到端浮点 NN 证明。Huan/Xiangru 的修正 profile 正在另立目录预检。

目前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **45 条**新尝试重算：**15 格**单次完整、1 格运行中、5 格早停、1 格失败，42 格尚未尝试。此前 43 条是上一阶段快照。

新增[已保存流管绘图汇总](evidence/results/archcomp26_20261001/plots/nohash_saved_20261001/SUMMARY.md)：[ACC 四方 t–安全距离裕量 PNG](evidence/results/archcomp26_20261001/plots/nohash_saved_20261001/acc_four_method_t_safe_distance_margin_tube.png)直接从 50 期六态 tube 盒计算半空间保守像，并画安全阈值 0；[论文方程 QUAD P3 t–x3 PNG](evidence/results/archcomp26_20261001/plots/nohash_saved_20261001/quad_paper_p3_1024x1000_t_x3_pooled_tube.png)使用 1,000 行每步 1,024 盒 union，目标仅是 T=5 终点。两者另有 PDF、MATLAB `.m`、逐步几何 JSON 和不含内容摘要的出图收据；MATLAB 脚本仍未实跑，QUAD 图不冒充逐盒或 octagon。

[Docking 与 Balancing/CartPole 来源审计](ARCHCOMP26_DOCKING_BALANCING_SOURCE_CONTRACT_20261001.md)已完成。Docking 的固定官方附带说明可冻结四原态输入、两力输出、全初盒、1 秒×40 周期及非线性全时安全不等式；缺四方可执行入口与该不等式的流管 checker。Balancing 的论文五特征控制器与固定四输入 ONNX/MATLAB 不一致，闭时间窗 `[8,10]` 与规格 `8<t≤10` 也不同；可另立仓库四原态 profile，但不能冒充论文五特征主合同。这两格暂不填数值成绩。

修正危险集的 **Attitude Huan/Xiangru** 也在各自独立 smoke 后完成 30/30 期、60/60 接受小段。进程 wall 分别为 7.094217 / 6.933376 s；两者新保存范围直接逐字节相同，各 60 个六维 tube 盒与官方真正的 unsafe 盒不相交，修正后的作者 checker 未输出 Unsafe/Unknown。原生显式 `VERIFIED` 和独立扫描如上；旧空集 checker 成绩不升格。两作者 [保存结果](evidence/results/archcomp26_20261001/author_attitude_avoid_v1/huan_full30_001/payload/RESULT.json) 仍缺 P3 同合同方法格及独立浮点 NN 证明。

最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **49 条**新尝试重算：**17 格**单次完整、1 格运行中、5 格早停、1 格失败，40 格尚未尝试。先前 45 条是阶段快照。

用于 GitHub 审阅的[本轮证据镜像](evidence/results/README.md)已开始放入分支，包含已完成的小运行收据、保存范围、图、CSV 和阶段报告；官方第三方控制器模型不重复发布，原生 QUAD 长作业的完整二进制范围仍保留服务器原路径。新短任务产生的后续收据继续同步，不借用历史数据填新结果。

**Single Pendulum 两物理态 / P3** 在新独立 smoke 后完成全初盒一分区的 20 控制期、100/100 接受小步，外层 wall **6.583305 s**。闭窗 `[0.5,1]` 的 50 次作者安全事件全部满足，保存 tube 的 `x1` union 为 `[0.5645452654370386,0.9932406822927875]`；见[新 P3 证据摘要](evidence/results/archcomp26_20261001/single_pendulum_two_state_p3_full20_001/SUMMARY.md)。P3/Huan/Xiangru/native 现构成该**两物理态加辅助时钟**合同的四方单次完整数值记录，不能挪作缺第三初值的官方三态 MATLAB 复现或稳态速度排名。

最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **51 条**新尝试重算：**18 格**单次完整、1 格运行中、5 格早停、1 格失败，39 格尚未尝试。先前 49 条是阶段快照。

**Attitude ours/P3 修正 unsafe** 在独立新目录完成一期 smoke 后，全程完成 30 控制期、60/60 接受小段，外层 wall **13.017036 s**；修正后的作者 checker 未输出 Unsafe/Unknown，保存的 60 个六维 tube 盒逐段与官方非空 unsafe 盒分离。见[P3 保存结果](evidence/results/archcomp26_20261001/p3_attitude_avoid_v1/full30_001/payload/RESULT.json)及[合同与旧 checker 错误](ARCHCOMP26_ATTITUDE_CONTROL_CONTRACT_20261001.md)。P3/Huan/Xiangru/native 因此均有该修正合同的单次完整数值记录；旧空集 `VERIFIED` 未借用，四方仍缺独立浮点 NN 证明和重复计时。

最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **53 条**新尝试重算：**19 格**单次完整、1 格运行中、5 格早停、1 格失败，38 格尚未尝试。先前 51 条是阶段快照。

## 10 月 2 日凌晨补记：ACC 重复计时与 TORA P3

- **ACC 24 次独立进程 campaign**：在同一明确命名的 participant-order 合同下，四种方法各有 1 次本 campaign 首轮新进程和 5 次后续新进程；轮次轮换，共 24/24 全时域完整、已保存 tube 安全。统一外层 `Popen` 前至子进程回收 wall，GPU 2、CPU 10–13 顺序执行。后五次中位数分别为原生 **7.736304 s**、Huan **8.237208 s**、Xiangru **7.987532 s**、P3 **8.740510 s**；各自逐次 min/max、首轮时间、50 步/RPC/控制器调用和 24 个独立原始目录在[正式 campaign 摘要](evidence/results/archcomp26_20261001/acc_fourway_campaign_001/SUMMARY.md)。首轮不是重启主机后的冷机；GPU 1 同时运行 QUAD 长作业，且缺独立端到端浮点 NN 证明，这组时间只作当前资源条件下的描述，暂不宣称稳定速度排名。最初各方法的一次单独运行仍保留，没有混入这 24 个样本。
- **TORA remain / P3**：独立一期 smoke 后，新全程尝试完成 12 初盒 × 20 控制期 × 10 小步 = 2400/2400 盒步接受，外层 wall **10.411001 s**。对保存的 2400 条范围独立扫描，四态全时 tube 均在 `[-2,2]^4`，无拒绝、首越界或终点超出同段 tube；作者 checker 无失败输出。末端四态绝对 `lo/hi/width` 与原始收据见[新 P3 摘要](evidence/results/archcomp26_20261001/p3_tora_remain_v1/SUMMARY.md)。原生也完整，Huan/Xiangru 仍是 `t≤18.4` 共同安全前缀；[四方绝对区间和空值表](evidence/results/archcomp26_20261001/tora_remain_fourway_common_prefix_20261002/SUMMARY.md)区分了这两类结局。失败进程 wall 不和完整时间排名；P3 的安全数值记录不等于独立端到端 NNCS 浮点证书。

当时[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)由 **79 条**新尝试重算：**20 格**记录完整数值时域、1 格论文方程 QUAD 原生仍在运行、5 格早停、1 格失败，37 格尚未尝试。先前 53 条是上一阶段快照。

TORA remain 的[四方 t–x4 保存流管图](evidence/results/archcomp26_20261001/tora_remain_fourway_common_prefix_20261002/plots/SUMMARY.md)已从新运行原始范围逐步重画，提供 PNG/PDF/MATLAB `.m` 与几何 JSON。原生/P3 展示完整 200 步；Huan/Xiangru 只画 184 步合格前缀，后续标为未知，不把幸存盒延伸到 T=20。

## 10 月 2 日接续：Airplane 全初盒阻断与 Docking 四方全程

- **Airplane continuous**：按固定官方完整 12 态初盒、20×0.1 s 合同另立入口，旧点初盒结果没有复用。order-6 Huan 一周期预检在任何 ODE 小步之前，19 变量六阶单项式配对构建期间 RSS 超过 54,006,540 KiB，252.247 s 后只终止该子进程组；order-3 诊断 profile 的 Huan/Xiangru 均在首个 0.01 s 小步拒绝唯一全初盒，分别 6.785/6.985 s、0 个接受步。未启动完整 T=2，也没有性质结论；内部首步拒绝子类未记录。原始记录和两种阶数差异见[三次尝试摘要](evidence/results/archcomp26_20261001/airplane_continuous_order3_fullbox_20261002/SUMMARY.md)。
- **Airplane continuous P3**：六输出严格注入的 CPU 预检通过且未初始化 GPU；两个新独立完整初盒、一周期 smoke 均未得到可接受流管。默认工作 P3/验证 P4 在首步前的单项式表编码遇到 `9^20 >= 2^63` 限制，外层 wall 13.758 s、0 个数值段；另立严格同阶 `solution_order` P3 profile 越过该入口阻断，但唯一全盒在首个 0.01 s 小步 `accepted=false`，外层 wall 12.500 s、0/10 接受。首拒立即停止，未启动 T=2 full，也没有性质结论；内部拒绝子类没有记录。两次原始记录和方法区别见[P3 摘要](evidence/results/archcomp26_20261001/AIRPLANE_P3_FULLBOX_SMOKES_20261002.md)及[入口审计](ARCHCOMP26_AIRPLANE_P3_NATIVE_ENTRY_AUDIT_20261002.md)。
- **Airplane continuous Flow* native**：同一官方单完整初盒另立三次一期 smoke，依次为历史 order 6 / remainder `[-0.01,0.01]`、order 3 / 同余项、order 3 / 加宽余项 `[-1,1]`。三次均各完成一次 12→6 控制 RPC，但首个 0.01 s 段 `UNCOMPLETED_SAFE`（状态 4）、0/10 接受，外层原始状态各为 `failed/exit2`，wall 4.877421/3.925546/3.925656 s；性质无任何已接受 tube 可检查，日志末尾的 `UNKNOWN` 不可改写为安全或真实反例。加宽余项只是参数诊断，不是完整官方结果。原始记录与来源级首拒说明见[原生三次摘要](evidence/results/archcomp26_20261001/native_airplane_fullbox_smokes_20261002/SUMMARY.md)。
- **Docking P3/Huan/Xiangru**：官方单完整初盒、40×1 s、每期 10 个数值小步，三个新 full run 各 400/400 接受，外层 wall 分别 17.411771/12.698839/12.668129 s。三方对非线性全时约束的 checker 均为 `Unknown.`；首段保守安全裕量 `q` 上界约 +0.016083，不能当作实际轨迹越界反例或安全证书。先前 Huan 一次 smoke 因公共模块导入路径缺失在数值推进前失败，保留该记录，随后新 smoke 成功；七次尝试及末端四态绝对区间见[三方原始证据摘要](evidence/results/archcomp26_20261001/DOCKING_FULLBOX_3METHODS_SUMMARY.md)。原生新入口由独立工作继续。

当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)登记 **89 条**新尝试：64 个方法单元中 **23 格**有完整数值时域、1 格原生论文方程 QUAD 运行中、5 格早停、3 格失败、32 格尚未尝试。这里的“完整”只指数值覆盖，Docking 性质仍未知；上述 79 条/20 格是上一阶段快照。

**Docking / 原生第四方法** 在独立一期 smoke 和全时域新目录中运行。full 走完 40/40 期、400/400 段、40 RPC，外层 wall 9.139416 s；原始 `RESULT` 仍是 `failed/exit 2`，因为原生性质 checker 返回 `UNKNOWN`。逐段保守径向安全裕量 `q` 的 400 个区间都跨零，首段上界 +0.016082965、末段最大上界 +8.954347227；区间相交不等于实际轨迹反例。40 次 CROWN 调用的上下仿射斜率逐元素相同，保存范围与 400 条性质记录独立核对；见[原生 Docking 摘要](evidence/results/archcomp26_20261001/native_docking_full40_001/SUMMARY.md)。四方法现在都有完整数值时域，但没有一家证明全时安全或独立浮点 NNCS 证书。

**旧作者 QUAD / P3 观测器成对消融** 使用旧 x2/x4/x5 方程、同一 1,024 盒与 40 小步，两个隔离无 JIT/无内容摘要新进程各接受 40,960/40,960 盒步；终步 1,024×12 的 tube、endpoint、status 数组和控制记录直接逐值一致。每步 observer 开/关的 driver 时间为 53.554101/53.323556 s，只有一对样本，不能外推全程加速；首次 on 入口被 TorchScript 守卫在 0 步前拦截也保留。见[成对原始记录](evidence/results/huan_quad_stage_a_40_20261001/observer_pair_v1/SUMMARY.md)与[速度模式报告](HUAN_QUAD_SPEED_AND_MODES.md)。当前证据不支持为省去短前缀约 0.23 s 而删除绘图所需的逐步 observer。

加入 Airplane P3 前的矩阵快照对应 **91 条**新尝试：**24 格**完整数值时域、1 格原生论文方程 QUAD 仍运行、5 格早停、3 格失败、31 格尚未尝试。此“完成”包括原生 Docking 的数值完成/性质未知/外层 exit 2，三者在 attempt 记录中分开表示。上面的 89 条/23 格为更早快照。

加入两个 Airplane P3 新 smoke 后，当时矩阵为 **93 条**新尝试：64 格中 **24 格**完整数值时域、1 格论文方程 QUAD 原生仍运行、5 格早停、**4 格失败**、**30 格未尝试**。P3 连续 Airplane 格归类为入口/首小步失败，旧单点成绩没有填入。

再加入三次 Airplane Flow* native 完整初盒新 smoke 后，最新[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 **96 条**新尝试：64 格中 **24 格**完整数值时域、1 格论文方程 QUAD 原生仍运行、5 格早停、**5 格失败**、**29 格未尝试**。Airplane continuous 四方方法单元均已有实际失败尝试，但均没有完整 T=2 数值流管或全时性质结论；上面的 93 条为此前快照。

## 10 月 2 日补记：NAV 作者执行顺序与首周期 smoke

- [NAV 作者执行合同补充审计](ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md)从控制器作者训练脚本和 CORA 闭环例确认：网络原序输入为 `[x,y,speed,heading]`，输出为 `[speed_rate,heading_rate]`；固定官方方程同此顺序。论文文字写 `[x,y,heading,speed]`，隐藏层也写 `64/64`，而固定官方 ONNX 为 `64/32`。新尝试按固定官方 ONNX 与作者可执行顺序显式命名；仍不声称作者仓库 Git LFS 模型与官方模型的内容身份已比对。
- 新的 Huan standard 和 robust 各取历史分区台账**首盒**，用固定官方 point/set ONNX 跑首个 `0.2 s` 周期，GPU 2 / CPU 10–13、无 TCP RPC 端口。两个独立 ID [`nav_author_standard_huan_smoke1_001`](evidence/results/archcomp26_20261001/nav_author_standard_huan_smoke1_001/RESULT.json) 和 [`nav_author_robust_huan_smoke1_001`](evidence/results/archcomp26_20261001/nav_author_robust_huan_smoke1_001/RESULT.json) 均接受 20/20 个 `0.01 s` 小步，保存 tube 与官方障碍盒分离，进程 wall 分别为 4.142439 s 与 4.417727 s。各自独立读回的区间检查与配置在[运行档案](evidence/results/archcomp26_20261001/nav_author_smoke_v1/README.md)；**未覆盖 640/25 全部分块、余下 29 期及 `t=6` 终点目标**。[旧 Huan 全程合同逐项比对](ARCHCOMP26_NAV_AUTHOR_EXECUTION_CONTRACT_20261002.md#旧-huan-全程记录与当前可执行合同逐项比对)确认两条历史结果在模型内容、方程、初集分区、采样、性质与数值设置上同合同，故不重复启动全盒作业；历史证据和本轮新 attempt 索引仍分开。
- [Unicycle 执行门](ARCHCOMP26_UNICYCLE_EXECUTION_GATE_20261002.md)明确了阻断：2026 论文只在速度导数加入 `w`，固定官方 MATLAB 右端没有 `w`，旧四方在朝向和速度两导数均加入同一个常值 `w`。论文未定 `w` 是轨迹常值还是可随时间变化，也未冻结“10 秒内到达”的检查语义；没有开启新 Unicycle 数值尝试，旧成绩不移入新格。

当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 **98 条**新尝试：24 格完整数值时域、1 格运行中、5 格早停、5 格失败、**2 格仅短前缀**、27 格未尝试。两条 NAV smoke 只使 Huan 两格变成“仅短前缀”，没有增加完整结果数。

## 10 月 2 日补记：离散 Airplane 与 TORA reach 启动门

- [Airplane discrete 执行门](ARCHCOMP26_AIRPLANE_DISCRETE_EXECUTION_GATE_20261002.md)核对论文的同步 forward Euler、12 态完整初盒、20 次 `0.1 s` 转移和 `k=0..20` 共 21 个安全检查索引。四方现有 Airplane 程序都推进连续 ODE，不能充当离散成绩。`x_k` 先取控制、再同步 Euler 可作为**新命名的四方共同比较合同**；若要求忠实复现 2026 参与者离散提交，仍缺离散转移及控制应用顺序的执行源码或等价权威记录。此审计未启动作业。
- [TORA 两个 reach 执行门](ARCHCOMP26_TORA_REACH_EXECUTION_GATE_20261002.md)直接逐元素核对了固定官方 `.txt` 和 `.mat` 的各 961 个控制器参数，并沿用此前 `.mat` 与旧 ONNX 的数值预检。模型文件已到位，问题是合同：sigmoid 的论文/官方/旧三种激活与缩放组合不同，tanh 的论文与官方激活不同；“5 秒内到达”也要确定用终点包含作为充分证据，还是要求完整时间窗判定。未选定前不把旧四方 500 步运行填为新版成绩。
- [Balancing/CartPole 执行门](ARCHCOMP26_BALANCING_EXECUTION_GATE_20261002.md)核对论文五特征 `f(x1,x2,sin x3,cos x3,x4)` 与固定仓库 ONNX 四原态输入 `[x1,x2,x3,x4]` 的冲突。论文忠实主合同仍缺五输入控制器或作者权威的五到四映射；固定仓库四输入版本已明确命名 `balancing-fixed-repo-raw4`，可单列准备四方入口与全初盒、10 秒性质窗检查，不能用旧一秒小盒诊断替代。

## 10 月 2 日补记：Balancing 四输入仓库 profile 的实际尝试

- 新隔离 [Huan raw4 入口](../tools/archcomp26_balancing_raw4_huan_nohash.py)先完成 CPU 模型预检：固定四输入、一输出 Gemm/Tanh 图可读取，未初始化 CUDA。第一次一期 smoke 的性质窗 `[8,10]` 与其 0.02 秒前缀不相交，驱动在 ODE 前明确失败；第二次一期 smoke 关闭不适用的性质检查，全初盒 4/4 小步接受。其原始收据把零次检查误标 `VERIFIED_BY_SAVED_BOX_CHECKS`，原文件保留，并在[审计勘误](evidence/results/archcomp26_20261001/balancing_fixed_raw4_huan/AUDIT.md)明确改读为**性质不适用**。
- 同一新命名 profile 的 500 期独立作业在第 99 个 ODE 内步拒绝，前 98/2000 计划小步接受，最后有效流管到 `t=0.49 s`；外层 wall 8.218304 s。`[8,10]` 性质窗未进入，0 次检查，结论 **UNKNOWN/incomplete**。原始记录没有内部拒绝状态码，不从流管变宽推测原因。论文五特征控制器仍缺，不把这次早停当作论文主合同成绩或完整性能样本。
- 加入上述 Balancing 三条尝试时，[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)有 **101 条**新尝试：64 格中 24 格完整数值时域、1 格原生论文方程 QUAD 运行中、6 格早停、5 格失败、2 格仅短前缀、26 格未尝试；两条旧 NAV 同合同全程证据仍单列在索引外。

## 10 月 2 日补记：Double Pendulum less 的第四方完整数值运行

- P3 的新[有向仿射分区审计](ARCHCOMP26_DP_P3_PARTITION_DIAGNOSTIC_20261002.md)保持官方四态 ODE、225 初盒、20 期、100 小步、全时 `[-1.7,2]^4` 安全带及原 P3 数值阶数。控制器 residual 使用原全局仿射图 `T`；每维二分的 16 个闭子盒完整覆盖输入盒，首次性质未决/收缩失败分别推迟到小步 75/84，仍未完成。随后独立的每维四分 256 子盒尝试完成 **100/100 小步、22,500/22,500 盒步接受、broken=0**；逐条保存 tube 的独立读回确认 22,500 个四态盒均在安全带内，最小盒裕量 **+0.093919848**。外层 wall **74.274085 s** 包含逐盒观察写盘，不能与其它三方单次 wall 做稳定排名。
- 这使 DP less 在原生、Huan、Xiangru 和 P3 四方都有同初集、同 1 秒时域的完整数值记录。先前 P3 失败尝试保持原始状态，不被成功尝试覆盖；控制器 residual 的局部有向包含论证不等于 P3 七变量 plant、倒数/三角、余项、端点与性质组合的端到端浮点证明，`end_to_end_strict_certificate=false` 保留。当前[无哈希工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md)为 **103 条**尝试、**25 格**完整数值时域、1 格运行中、6 格早停、4 格失败、2 格仅短前缀、26 格未尝试。
- 已从四方**保存的全量数值数据**生成[DP less 四方绝对宽度汇总](evidence/results/archcomp26_20261001/dp_less_fourway_split4_20261002/SUMMARY.md)、16 行[终点统计 CSV](evidence/results/archcomp26_20261001/dp_less_fourway_split4_20261002/endpoint_stats.csv)、1,600 行[逐步 tube union CSV](evidence/results/archcomp26_20261001/dp_less_fourway_split4_20261002/tube_union.csv)及[四态叠加图](evidence/results/archcomp26_20261001/dp_less_fourway_split4_20261002/fourway_tube_union.png)。Huan/Xiangru 的保存区间直接相同；P3 在第四物理态的 `T=1` endpoint union 宽为 0.799090015，原生为 1.016584896，两作者为 1.109787875；每盒平均宽度的次序并不一致，不能用单一 union 宽度宣布优劣。PDF/MATLAB `.m` 同目录，脚本未在 MATLAB/Octave 实跑。

## 10 月 2 日补记：DP more 首周期与独立数值点诊断

- 固定官方 more-robust ONNX、官方四态 ODE 和完整 `5×5×3×3=225` 初盒的新[P3 首控制期诊断](evidence/results/archcomp26_20261001/dp_more_p3_firstperiod_interval_20261002_001/SUMMARY.md)只运行 `T=0.02`，4 个 0.005 s 小步、900/900 盒步均数值接受。作者驱动打印 `Step 0`、`Unknown.`；原始 JSONL 独立重扫的四步安全盒数依次为 `[225,225,181,120]`，第 3 步首次有 44 盒的 `θ̇₁` 上界超过 1.5，第 4 步增至 105 盒。首例 lane 6 保存区间 `[0.9912745512752368,1.5106057605641459]` 只表示区间无法证明安全，非实际轨迹反例。未启动 P3 `T=0.4` 全程，也未重复旧 Huan/Xiangru/原生区间作业。当前矩阵由 **104 条**尝试重算：25 格完整、1 格运行中、6 格早停、4 格失败、**3 格仅短前缀**、25 格未尝试。
- 先前固定官方 ONNX 的[名义角点 `(1.3)^4` 数值重放](evidence/results/archcomp26_20261001/dp_more_point_candidate_20261002/SUMMARY.md)显示越界，但 binary64 `1.3` 比精确初盒上界 `13/10` 大约 `4.44e-17`，故单凭这条记录不能称为精确初盒内的候选反例。为修正初点资格，另做唯一[严格内点 `(1.299)^4` 数值重放](evidence/results/archcomp26_20261001/dp_more_interior_point_candidate_20261002/SUMMARY.md)：每 0.02 s 以 float32 ONNX 控制后持值、官方 ODE 积分；DOP853 估计 `θ̇₁=-1.5` 下穿在 `t=0.3248652630`，`t=0.325` 网格值为 `-1.5004299419`，`t=0.36` 值为 `-1.6071729076`，位于已保存 Huan lane 224 endpoint 区间内。两档 RK4 与 DOP853 的保存期末差小于 `1.05e-14`。这是**合同内初点的数值候选**，一致性不是严格误差界，控制器浮点语义与 ODE 逐期包络仍需验证，不称为严格反例，也不作为第五种 flowpipe 方法计入四方索引。
- 已将 103 条截点的可编辑 [DOCX](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.docx)和同名 [PDF](evidence/results/archcomp26_20261001/stage_report/ARCHCOMP26_STAGE_REPORT_DRAFT_20261001.pdf) 更新到 DP less 四方完整数值记录和六幅图；Word 导出的 17 页 PDF 已逐页目视检查，避免替代排版器漏中文。新 DP more 诊断晚于该固定阶段快照，先记在本进展和 Markdown 完整草稿中。

## 10 月 2 日补记：Airplane 离散命名约定的有界入口

- 对已明确命名、但尚非参与者官方执行身份的 `paper-Euler-controller-first` 约定，使用固定 12→6 ONNX、完整 12 维单盒和 12 态同步 Euler 映射做了[首次一步 CPU 区间诊断](evidence/results/archcomp26_20261001/airplane_discrete_paper_euler_interval_smoke1_001/AUDIT.md)。`k=1` 的 `sy` 包络在安全带内；三个角度因通用外舍入各比精确 `[0,1]` 多一个极小浮点格，原始 checker 为 `Unknown`。固定 MATLAB 动力学的 `p=q=r=0` 使首步角度导数代数上确为零；保留原始结果后，[第二次新诊断](evidence/results/archcomp26_20261001/airplane_discrete_paper_euler_exactzero_prefix_001/AUDIT.md)只对该精确零恒等式特殊处理。`k=1` 四个安全坐标均在闭带，随后 `k=2` 包络大幅越带，首次实质性 `Unknown` 即停；安全端点前缀只有 **1/20 次转移**。两次均是四方方法之外的 CPU 区间入口，不给 Airplane 离散主表四格填成绩、证书或完整运行时间。参与者真实的离散控制取样与转移顺序仍缺。

## 10 月 2 日补记：TORA reach-sigmoid 官方文件 profile 的一期结果

- 为避免把旧 `u=22(f−0.5)` 成绩误认成 2026 官方 `u=11f`，另建唯一[TORA 官方文件一期 Huan 诊断](evidence/results/archcomp26_20261001/tora_reach_sigmoid_official2026_mat_u11_firstperiod_diag/SUMMARY.md)。固定 `.mat` 四层 sigmoid 经显式构造后，ONNX 图内 `Mul(11)+Add(0)`，外部 `output_scale=1/output_offset=0`，完整单初盒的一个 0.5 秒控制期、50/50 ODE 小步全部接受。原始 `START/RESULT`、50 条范围、日志与独立有限/有序扫描均保存；`T=0.5` 末端 `x1∈[-0.8850144075,-0.8573825037]`、`x2∈[-0.0066841850,0.0219011293]`，外层单次 wall `5.345883 s`。没有判定 5 秒内到达目标，没有启动旧 500 步或 tanh，且没有速度排名资格。由于论文和官方网络激活冲突，仍须用户选主合同；本结果按官方文件 profile 单列。纳入此短前缀后工作矩阵为 **105 条**尝试、25 格完整、1 格运行中、6 格早停、4 格失败、**4 格仅短前缀**、24 格未尝试。
- 原有论文方程 QUAD native 长作业在 2026-10-02 00:26:29 UTC 的只读复核仍为 `RESULT_PENDING`：日志在第 49 个控制期的 Flow* 计算，原始 `ranges.bin` 已写前 **980/1000** 小步，native 与 RPC 进程仍存活且 CPU 时间持续增长。未启动副本、重启或干预；没有第 50 期的完整时间、终点宽度或性质结论。

## 10 月 2 日补记：原生论文方程 QUAD 自然完成及四方数值边界

- 原有 `native_quad_paper_full50_001` 在 **2026-10-02 00:30:15 UTC** 自然完成，原始 `RESULT.json` 为 `completed`、exit 0、未超时，外层 wall **47058.886571 s**（单次）；native 日志打印 `VERIFIED`、内部 `time cost: 46918.890000 s`。这是上条 00:26 仍运行作业的**同一原始运行**，没有重启旧六小时实验、增加副本或重复登记 attempt。原始 START/RESULT、日志、50 次 RPC、构建源码及[只读范围扫描](evidence/results/archcomp26_20261001/native_quad_paper_full50_001/SUMMARY.md)已保存到分支的小证据包，417,792,000 字节 `ranges.bin` 保持在原服务器路径。
- 原始范围按“50 期→1024 盒→期内 20 小步”精确覆盖 **1,024,000/1,024,000** 条唯一 `(盒,小步)` 记录；有限性、区间顺序、endpoint 含于对应 tube、盒/步/步长检查均为零异常。`T=5` 的 `x3` endpoint union `[0.965771839016746,1.0167484756616678]`、宽 0.0509766366449218，位于目标 `[0.94,1.06]`。最后一小步 tube 宽 0.05097738677735586，与 endpoint 分列；全时 `x3` tube `[−0.40817151558335585,1.452252535346294]` 不能冒充终点。原生源码 checker 只用 `fp_end_of_time.isInTarget` 终点判定；step 775–1000 保存 tube 连续落目标带只是额外数值诊断，**论文 reach-and-remain 时间语义与独立端到端浮点 NNCS 证明仍未闭合**。
- 2026 论文方程 QUAD 的 P3、Huan、Xiangru、原生现各有一次完整 1024×1000 数值记录；四方 `T=5` 的十二态 endpoint 绝对下/上界与并集宽见[48 行 CSV](evidence/archcomp26_quad_paper_endpoint_4methods_20261002.csv)。四方 `x3` endpoint 宽依次为 **0.067225749 / 0.047741817 / 0.047741817 / 0.050976637**；运行路线、资源和观察器不同，不据单次时间或宽度做速度/正确性排名。原生原作业的索引行已就地更新，工作矩阵仍为 **105 条尝试**、64 个方法单元中 **26 格完整数值时域、0 格运行中、6 格早停、4 格失败、4 格仅短前缀、24 格未尝试**。
- 新[四方 QUAD t–x3 图与数据包](evidence/results/archcomp26_20261001/quad_paper_fourway_saved_20261002/SUMMARY.md)以原生和 P3 各 1,000 小步的完整初盒保存 tube 画曲线；Huan/Xiangru 没有保存逐步范围，只以 T=5 endpoint 标记。目标 `[0.94,1.06]` 只作为终点目标，MATLAB `.m` 已导出但没有 MATLAB/Octave 实跑证据，也不能由轴对齐投影恢复 Flow* octagon。

### 仍需外部材料或明确选择的字段

| 实例 | 具体缺件或选择 | 在此之前可独立推进的工作 |
| --- | --- | --- |
| QUAD reach-and-remain | 2026 参与者实际运行的全时间窗 checker 源码或同等权威执行记录；目前四方完整数值时域及原生终点检查已保存，仍不足以固定全时间窗语义。 | 保留四方终点和原生逐步 tube；所见 step 775–1000 连续入带仅作数值诊断。 |
| Single Pendulum 官方三态 | 第三状态的初值和三态 MATLAB 闭环执行入口；现有四方仅是明确命名的两物理态加辅助时钟合同。 | 保留两态 profile 的四方结果及图。 |
| Airplane discrete | 参与者离散转移与控制更新次序的源码或权威记录。 | 可单列实施已明示的 `paper-Euler-controller-first` 四方比较约定。 |
| Balancing 论文五特征 | 五输入控制器文件，或作者明确的五特征到固定四输入模型映射/执行源码。 | 可单列实施 `balancing-fixed-repo-raw4`。 |
| TORA reach 两变体 | 各自选择论文字面或官方可执行激活、输出缩放，以及“5 秒内到达”的判定口径。 | 已核对模型参数与显式构造器；官方文件 sigmoid `u=11f` 仅有 Huan 一期诊断，旧成绩保留为不同合同对照。 |
| NAV standard/robust | 决定新版主表采用论文文字状态/层宽，还是已核实的官方模型加作者可执行顺序。 | 旧 Huan 全程同合同证据已复核，无需重跑；新首周期 smoke 已单列。 |
| Unicycle | 扰动仅加速度还是沿旧代码同时加朝向，以及 `w` 是轨迹常值还是随时间变化；还需 reach 时间量词。 | 固定 ONNX 与旧模型已直接对比，可复用模型文件。 |

这些来源冲突均未用旧实验或自拟默认值填补新版主表。其余未完成格仍需各方法实现、接受/失败记录、宽度和重复计时；这些是当前工程工作，不当作待用户提供的文件。
