# 上次 slides 之后的范围、复现身份与结论边界

本文件供新 TeX 汇报取材。2026-09-29 仅检查本地既有材料；没有 SSH、求解器、GPU 或新实验操作。实验保持暂停。这里负责报告范围与科学口径，完整数值表由本次汇报的独立数据整理提供。

## 1. “自从上次 slides 后”的准确起点

上次交付为 [README](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/README.md)、[slides.tex](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/slides.tex)、[REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/REPORT.md)。共 29 页 slides（23 页主讲、6 页附录）和 17 页报告。发布分支 `codex/progress-report-20260923`，commit `3edae8ce2e5a2b3e7915c645742ececcabc6b8ef`，见 [发布收据](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923_publication.json)。

本次直接从 ZIP 读取并与目录文件比较，`slides.tex`、`README.md`、`REPORT.md` 全部逐字节相同，避免依据后来改写的报告推断旧演示内容。

| 已核对象 | SHA256 |
|---|---|
| `flowstar_progress_review_20260923_latex_bundle.zip` | `edfe5e9bc497d74fd5e3b2fcc76fa5cb4e507a3f992f9cd08397fae5d2702968` |
| 旧 `slides.tex` | `f96380982cd462e1f48121f57b431a00a1715b7caede750767f17254e31f917e` |
| 旧 `REPORT.md` | `8186844672192c8f0bdf00c3f9060bdaa4d4d2830d3c248e61884f3e121a3c0d` |

旧 slides 已经讲过：完整 GPU 推进链；VDP/Brusselator 的 B1 长轨迹与 B32 真实分区四项比较；TORA 线性路径 v2 从部分失败到 2400/2400；单摆和 CartPole；输出后端五次配对改善 15.3%–43.6%；独立 GPU 乘法候选；12 plant / 14 NNCS 的早期筛查；控制器舍入资格未完成。

**旧 slides 没有汇报后续 ARCH-COMP 14 配置四方矩阵、Huan QUAD 75 秒复现、ACC 参照更正、QUAD 连续修复到完整 1024×1000、三角函数复用的约 2 倍完整运行改善。** 这些应是新汇报主线。旧材料与新增 ARCH-COMP 矩阵都标 2026-09-23，因此范围边界按已交付内容确定，不能简单排除同日后续工作。

## 2. 四个仓库及实际被测资产

旧 slides 标“三个仓库、四类实现”；后续 ARCH-COMP 对照明确增加官方 CROWN-Reach 源仓库。四个来源仓库应这样介绍：

| 来源 / 角色 | 固定资产与分支证据 | 入口和复现范围 |
|---|---|---|
| 用户工程 [torch_tm_flowpipe](https://github.com/lsnnnnnnnn/torch_tm_flowpipe) | 工程、适配、实验与报告；旧报告已发布于上述 `codex/progress-report-20260923` | 不把报告分支 commit 当作最新数值引擎 commit |
| Huan [flowstar-gpu](https://github.com/huanzhang12/flowstar-gpu) | `d5f0b68fcd36ba5f582733624f074728fe9720d8`；当时 Huan `main` | 独立 NNCS 入口 `integrations/crown_reach/gpu_driver.py`；后续 SR 分块是隔离变体，不冒称原源未改 |
| Xiangru [CROWN-Reach_Development](https://github.com/xiangruzh/CROWN-Reach_Development) | 来源分支 `2026_experiment`；实际 detached `1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3` | 独立入口 `src/flowstar_gpu/integrations/crown_reach.py`；也作为三方对照的共享驱动 |
| 官方 [CROWN-Reach](https://github.com/Verified-Intelligence/CROWN-Reach) | `7b90f30831212f0c59c44c67aa428932d5307a81`；本地历史 inventory 未保存分支名，不补猜 | `archcomp/*` C++ 程序、同目录 CROWN 服务、模型与原生 Flow* 对照；原入口和固定步数/初始化修复入口分开 |

精确来源：[remote_inventory.json](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/remote_inventory.json)、[ARCH-COMP 报告](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/REPORT.md)、[Xiangru clone 记录](/Users/shengenli/Documents/ChatGPT/verification/planning_sources/conversation_page_2.md:721)、[Huan QUAD 复现报告](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_failure_20260923/REPORT.md:14)。这些是历史固定版本；本轮没有联网核查作者最新 HEAD。

**“我们”是一条派生实现线，不是完全独立发明的第四种数学算法。** 新增 ARCH-COMP 主矩阵中被测我方引擎为 `fff9d0f5742ebfa4e0652bbbf79b31bdd59ab3e3`，服务器工作树 `engine_linear_leaf_v2`；实际 inventory 的 remote 仍为 Huan `flowstar-gpu`。不能把这个 commit 说成已推送到用户仓库或作者仓库。该次工作树分支名未在所读 inventory 中保存，以完整 commit、源码 SHA、实际库身份为准。最新 QUAD 又由冻结数值引擎与独立适配器组合，不能仅用旧 fff9 标签代表其全部代码。

### 独立入口确实跑过

原始 Huan 和 Xiangru 入口各自执行默认 Single Pendulum，均 `completed/rc0`，B1、order2、20 控制周期×5 子步=100 步；实际 `coupling=hybrid`。两份 `ctrl_steps` 的 20 条记录本轮重新以 JSON 相等比较通过，打印的四条最终 HULL 也完全相同。

| 原始入口 | 本次已读历史进程 wall | 原始 stdout 的内部 time cost |
|---|---:|---:|
| Huan `gpu_driver.py` | 8.7958964551799 s | 4.058186 s |
| Xiangru `crown_reach.py` | 8.402115842327476 s | 3.245168 s |

共同物理 HULL：x1 `[0.566926486592, 0.704484162534]`；x2 `[-0.557712860545, -0.418458767625]`。这两次只证明该默认入口可运行，不是五次性能排名，也不证明所有 benchmark、分支或模式已独立复现。

原始 argv、返回值和 metrics 均保存在 [Huan 独立入口](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/huan_standalone_sp_default/process.json) 与 [Xiangru 独立入口](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/xiangru_standalone_sp_default/process.json) 同目录。三方主矩阵另由共享 Xiangru 驱动分别加载各自被冻结引擎；这是隔离 plant 实现差异的共同合同，不应称三套独立完整 NNCS 产品。

## 3. Huan 与 Xiangru 为什么相同

实际 [SOURCE_AUDIT.json](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/SOURCE_AUDIT.json) 校验 58 份冻结源文件：29 个核心 Python 文件中 27 个逐字节相同，内嵌 CUDA 数学源码相同。两个差异文件为：

- `cuda_kernels.py`：Xiangru 增加宿主编译器环境选择与检查；数学 CUDA 源相同。
- `sparse_exec.py`：增加 NVTX 诊断及可选 `FLOWSTAR_COMPOSE_PARENT_ASSEMBLY=batched`；默认 `loop` 路径保持对应算术。不能把默认路径结论扩大到可选 batched 模式。

原始 diff 在 [sparse_exec.py.diff](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/sparse_exec.py.diff) 和 [cuda_kernels.py.diff](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/cuda_kernels.py.diff)。新增主矩阵 13 个可比配置的完整 H/X 范围文件逐字节相同；因此整体 union hull 相同有源码与实际轨迹双重依据，并非只因表格四舍五入。

可用于 slides 的一句话：**这两个被冻结版本在本次默认核心路径高度同源，同输入产生相同范围是预期结果；它不是两种独立算法相互证明正确。** 也不能把 Huan 后来的 parity QUAD 结果自动记成 Xiangru 已重跑的结果。

## 4. 必须分开的数值与控制器合同

| 名称 | 实际区别 | 汇报限制 |
|---|---|---|
| plant `strict` / `parity` | 涉及误差收费和 refinement 尾项语义；parity 对齐原 Flow* 的部分 point-trust 行为 | strict 标签不是完整 NNCS 证明；parity 更窄/更快不能直接当同保证加速 |
| CROWN `box` | 在输入状态盒上建立控制仿射界 | 需要固定 relaxation、输入排列、模型与传输精度 |
| CROWN `hybrid` | 对每个 cell/output，在 same-slope box 与 tm-affine prelayer 耦合候选中选更窄者 | 与 strict/parity 是不同维度；不是“hybrid=parity”或“box=strict” |
| `same-slope` | NN relaxation 策略 | 不能与 two-slope、alpha 迭代或 TM-through-NN 混表 |
| `rpc-float32` | 复现 C++ RPC 的 `asFloat`，再按实际路径传递 | 不是由 ONNX dtype 就能推断的合同 |
| `native-f64` | 作者 GPU 原驱动的系数通路，无上述 float32 RPC 截位 | 历史正式矩阵与后续 QUAD 必须分开标记 |

Huan 文档记载 hybrid 默认自 2026-08-05 才切换；此前原 QUAD 83.048290 秒日志是 box/same-slope，不能把当前默认 hybrid 倒推成旧记录条件。见 [作者冻结 REPRODUCE.md](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_failure_20260923/evidence/source_evidence/REPRODUCE.md:251)。

后续 Huan 完整 QUAD 复现实测是 **parity + box + native-f64 + SR 临时量分块**。分块提交 `f83f6d84f0624608a5bdf85e5af7ac4d394e679e`，分支 `codex/huan-sr-chunk-20260923`，仅在 B 维以 128 分块计算 SR dot 临时量，保留完整 Q 历史和原求和。资格/strict 599 步状态与非计时 metrics 不变。五次完整进程中位 75.250 s、作者内部 70.728 s，全部 1,024,000 接受分区步。作者原日志 83.048290 s 来自不同历史环境，不能据两者算优化倍率。原始日志资产所在配置仓库版本 `c28f0db949b87c475bf58404767b771f42d26dbe` 也不等于当年引擎原始 commit。详见上述 Huan 复现报告。

## 5. 正确性审查的新证据与必须更正的旧解读

“没有通过缩任务、删失败或关误差获得优势”有配置、版本、接受掩码与日志证据；但不能因此承诺三个引擎无错误。新增审查实际找到问题，必须同时汇报。

### A. VAR 截断尾项与 ACC 原生参照

原生/parity refinement replay 遗漏 VAR 截断尾项的精确局部反例：返回 RHS 上界 0.275625，而合法测试值 0.292275390625 在外；三方 strict 路径包含该见证。Huan GOTCHAS #16 已记载此语义差别，本轮为独立复核，不宣称首次发现。

实际原生 ACC 的 t=.1/.2 端点排除合法解析轨迹；直接原库最小例复现，隔离补回 VAR-tail 后原生完整 50 步。关键 step2 a_lead 宽度修复后 0.00224735439，与 H/X strict 0.00224700883 接近。因此旧四方表约 7.94 倍应保留为**旧输出比值**，不能继续作为“GPU 精度差约八倍”的结论。源码、实际执行和修复依据见 [ACC/VAR-tail 报告](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/REPORT.md)。不据此声称所有其他原生输出均已证明错误或修好。

### B. 通用倒数公式的实际反例

分别导入两份未修改的作者 CPU sparse 归档，`1/(2+z/4)`、z∈[-1,1]、k=1–4、零输入余项、cutoff0，实际 `bad=false`，但 z=-1 的精确 4/7 超出返回完整范围。k1 缺口约 0.00309766763848，k4 仍约 4.3889656593e-7。H/X 四组全部保存字段逐字节相同；我方冻结通用倒数亦有相同问题。见 [作者实际 CPU 反例](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/reciprocal_author_counterexamples_cpu_20260928/REPORT.md)。

本次 native matched 构建对应源码含同类额外 1/c 缩放；此项是源码定位，不冒称运行了 native 二进制反例。c=2 的通用见证也没有证明某个已接受 QUAD cosine 分母步实际漏解。新的几何尾项适配器基于有限几何恒等式，须按已检查调用域及路径解释，不覆盖未替换的 dense/replay。见 [源码与精确反例](/Users/shengenli/Documents/ChatGPT/verification/results/quad_native_matched_20260928/RECIPROCAL_TAIL_SOURCE_REVIEW.md) 和 [几何尾项合同](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/RECIPROCAL_GEOMETRIC_ADAPTER_REVIEW_20260928.md)。

### C. 初始盒覆盖和控制边界

原生 full1024 构造器采用已舍入中心 c，却只以 RU(u-c) 定半径；384/16,384 坐标、338 lanes 的实际初始 affine 欠包。首例原下界 -0.30000000000000004 被收成 -0.3，缺口 2^-54。这虽然只有 1 ULP，仍不能忽略或靠放宽 observer 容忍度掩盖。独立新初始化保留 c，取 max(RU(c-l),RU(u-c))；实际原/新 MPFR 系数与 Fraction 审查确认只 384 半径改变、全部 16,384 行覆盖。见 [原生初盒独审](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929/REVIEW.md)。当前 GPU14 原完整运行的 actual-coefficient 初始覆盖 16,384 项已通过，[GPU 独审](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929/GPU_INITIAL_REVIEW.md) 明确同 run 身份。新原生整程结果应标初始化修复变体，不能称未改原版。

早期 plain endpoint 的时间代入采用 RN、未收费；后来的 strict endpoint 才对当前 FULL 支持、时间幂/乘加误差及 pre_rem 交接收费，并保留原完整 SR。旧晚升阶轨迹不能被后来的修复追溯认证。还要单独说明控制器 CROWN 浮点界与 RN 仿射注入仍未取得完整端到端资格。见 [P2/P3 边界独审](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/P2_P3_STRICT_BOUNDARY_REVIEW.md)。

## 6. 阶数、任务完整性与最新优化证据

原 ARCH-COMP QUAD 固定阶为 2，而非“Flow* 始终保留很多阶”。通用 YAML 的 order4 同时带不一致的控制/ODE 步长，不是本次官方 C++ 合同。引擎可以指定阶数；最新 P3 路线明确增加工作多项式总次数，必须标为算法变体。

| 路线 | 工作解总次数 | point RHS code | 验证阶数 |
|---|---:|---:|---:|
| 后续 P2 对照 | 2 | 1 | 3 |
| 后续 P3 | 3 | 2 | 4 |

总次数包括局部时间 τ 和归一化空间变量。例如 τ·z_i·z_j 为三次，P2 无法保留，P3 可以。它与只提高验证展开阶数不同；“P3”也不等同于三次控制器模型。理论完整基 P2=171、P3=1,140、P4=5,985（17 多项式变量），不能当实测活跃项数或显存。来源：[阶数审查](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/ORDER_CONFIGURATION_REVIEW.md)。两叶/32-root 的局部诊断不冒充全部 1024 根盒。

最新三角函数复用并没有再改 P3 数值算法：限定一次 `exec_valid_s` 内相同直接变量、运算和绑定，首次完整执行，重复项复制完整系数/余项/缓存。必须将此实现优化与此前 P2→P3、控制交接、SR 调整等算法改进分开。

最终实际 CPU 终态审计状态为 `full1000_actual_outputs_equal`、`issues=[]`、`partial_logs={}`，核对全部 1000 observer、50 controller/transfer、两最终 plant 和 14 个 SR/host 组件，与原 GPU14 P3 基线逐位相同；数值轨迹/基线字节资格 true。该审计仍明确 `fullbatch_qualification=false`、`end_to_end_strict_certificate=false`、`goal_complete=false`，不能偷换成总体完成。见 [最终原始审计](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/TRIG_GPU14_TERMINAL_REVIEW_20260929/RESULT.json)，SHA `517bd183820d43af794d92b64434d84e05e9fb4fa120927f2e9bf3c41968add2`。

实际完整同配置 watch 时间 3153.449456484→1533.752051520 秒，约 2.056 倍；这是**同基线单次完整资格运行的改善**，不是五次正式计时，也不是对 Huan 75 秒 parity 的同保证胜出。当前完成 1024×1000、50 NN、无数值拒绝；“accepted”不是完整系统安全结论。底层 40 步 phase/CUPTI 用于定位小 kernel 重复，带有测量开销，不能替代这两个完整 wall 数字。

## 7. 新汇报需要固定写出的比较边界

1. **范围定义一致才拼表。** 旧 slides 组合 tmv 范围与新增矩阵局部 tmvPre hull 不同；endpoint 与整步 tube 分开，分区均值与全分区 union hull 分开。NAV standard 的最大单分区比 17.0645 对应原生宽度 3.44169e-15，不能说整体宽了 17 倍；其 union 最大约 1.0096。零参考宽度另列，不加 epsilon。
2. **只比较共同已接受时域。** attempted steps、末周期部分接受、性质早停、数值拒绝、资源守卫和超时是不同状态。不能用失败后旧状态补齐曲线，也不能用前缀时间作完整速度分母。
3. **控制器必须同合同。** 固定实际模型、排列、shape/batch、relaxation、传输精度。旧 QUAD 只做 6 批 endpoint proxy 布局检查不能说成完整真实 RPC 重放；缩成 32 batch 曾有实际 float32 一 ULP 差，必须保留失败证据。
4. **正式计时与诊断分开。** 新增七项四方五次矩阵有 140 计入样本；QUAD 后续多数是带 observer、审计/保存的单次资格运行。内部 elapsed、driver、process/watch、独立导出各有边界，不混取最小者，也不把来自不同运行的中位数相加。
5. **原版/修复版和资源变体分列。** 主存/显存守卫变化会改变能否跑完；提高预算本身不是算术修复。最近 GPU guard14 GiB、allocator13.5 GiB、RSS11.5 GiB；旧 11.5 GiB 守卫失败仍保留。原生提高 RSS 的行也需注明；不能因更高预算而说原算法被改快。
6. **有限见证和字节相同是有范围的证据。** 源码 SHA/import 路径检查支持忠实复现；解析 Fraction 见证查实际包含；CPU 参数检查不是 CUDA 数值资格；旧新字节相同证明受检行为不变，不能证明共享错误不存在。
7. **安全状态不升级。** VERIFIED/Unsafe/FALSIFIED 是该程序在其合同下的输出；控制器/端到端证明未完成。Airplane 原先 78/79 的差别后来由半步性质检查解释，不应记成 200 步完整失败或成功。
8. **文档也保留未知。** native 历史分支名未录、作者当年 dirty source 不可重建、未完成轨迹的全时长宽度等明确留空。局部源反例不推广到所有作者分支；当前文件名中的历史日期不能替代实际运行阶段。

推荐主叙事：先复现并分清模式 → 用更多 benchmark 找出失败/宽度差 → 发现并更正参照中的局部问题 → QUAD 从失败推进到完整接受 → 保持完整输出不变将自身运行约减半 → 完整四方同保证比较及端到端控制器资格仍未闭合。
