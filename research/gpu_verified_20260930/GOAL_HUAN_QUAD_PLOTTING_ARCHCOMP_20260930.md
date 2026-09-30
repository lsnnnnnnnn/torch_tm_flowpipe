# 下一阶段 Goal：解释 Huan QUAD 的速度、接入可达域绘图、完成最新 ARCH-COMP 对照

日期：2026-09-30。用户用途：将本文件挂到新对话，接着执行完整工作。

## 0. 用户要求与执行顺序

先读完当前分支及交接，不从头猜仓库、配置或已经完成的实验。随后按顺序推进：

1. **研究 Huan 的 QUAD 为什么约80秒，写出有源码和实测依据的报告；解释 strict/parity 为什么存在、到底改了什么，判断哪些加速技巧能接入我们的方法，并实施有依据的优化。**
2. **把 flowpipe 绘图做成直接可用的工程功能**：支持 MATLAB 查看，画出可达域、initial box 和论文图里的安全区域；参照最新 ARCH-COMP 报告的图式，不再要求用户手写后处理。
3. **按最新 ARCH-COMP AINNCS 文档，复跑除 VCAS 外的全部 benchmark 及其变体**，对照我们、Huan、Xiangru、原生 Flow*，整理时间、宽度、性质结果和图。
4. **仿照该报告写一份新的完整实验文档**，使不认识本项目的人也能理解做了什么、设置是什么、快在哪里、宽度如何、哪些失败或仍未解决。

用户所说“所有branch”在这里按**benchmark及其控制器/连续离散变体**解释，不是遍历无关 Git 分支。用户已澄清“verified box”指图中那样的**安全区域**；不能擅自改成“验证成功的初始子盒”。具体是全过程安全集、终点目标集还是unsafe集合的补集，逐个依照原文定义，图注标明。

本文件制定时仅完成 GitHub 整理发布和计划，没有启动上述新实验。在新对话中用户要求“执行/恢复这个goal”后持续推进；遇到具体材料或认证阻断才明确说明，继续其它不依赖该阻断的工作。不要反复询问已经授权的常规修复、读仓库、隔离分支开发和实验。

## 1. 唯一接续入口、材料和当前状态

### 1.1 最新 GitHub 分支

- 仓库：<https://github.com/lsnnnnnnnn/torch_tm_flowpipe>
- 新分支：[`codex/gpu-verified-handoff-20260930`](https://github.com/lsnnnnnnnn/torch_tm_flowpipe/tree/codex/gpu-verified-handoff-20260930)
- 最新已验证源/证据快照位于 `research/gpu_verified_20260930/`；先读 `README.md`、`STATUS.md`、`source/SOURCE_MAP.json`、`EXTERNAL_ASSETS.json`。
- 第一份源码与证据发布提交为 `409d91a`；后续文档提交不改变数值源身份。以分支实际 HEAD、SOURCE_MAP 和实验 INPUT/COMMAND 的 SHA 为准。
- 本机发布 checkout：`/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_latest_20260930/repo`。本机研究资料根：`/Users/shengenli/Documents/ChatGPT/verification`。
- 详细历史：本机 `HANDOFF_FLOWSTAR_GPU_20260930.md` 和 `EXECUTION_STATE.md`；GitHub 内复制版在快照的 `handoff/`。它们较早的“native400、终态未知”已被下面9/30的新记录覆盖。

快照包含141个精确来源文件、全部36个基础引擎Python文件、最新入口及依赖适配器、配置/初盒/QUAD模型、报告和小证据。数值源码未改；source snapshot**尚不是新机器一键运行包**：原入口依赖固定服务器路径、干净a3fb引擎checkout、预编译SO与资格PT。先保持这些证据门，后续将路径/运行入口整理为可维护形式，不靠删除检查来假装可复现。

### 1.2 三方和原生身份

| 资产 | 冻结身份 |
|---|---|
| Huan | <https://github.com/huanzhang12/flowstar-gpu>，`d5f0b68fcd36ba5f582733624f074728fe9720d8` |
| Huan SR分块复现 | `codex/huan-sr-chunk-20260923`，`f83f6d84f0624608a5bdf85e5af7ac4d394e679e` |
| Xiangru | <https://github.com/xiangruzh/CROWN-Reach_Development>，来源`2026_experiment`，保存`1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3` |
| 旧ARCH矩阵的我们 | `fff9d0f5742ebfa4e0652bbbf79b31bdd59ab3e3`，engine_linear_leaf_v2 |
| 最新QUAD基础引擎 | `a3fb2e94ba976aaf498c4a9cb3f98165cddcc272`，另加snapshot中记录的严格倒数/边界/SR/P3/trig适配器 |
| 官方CROWN-Reach保存版 | <https://github.com/Verified-Intelligence/CROWN-Reach>，`7b90f30831212f0c59c44c67aa428932d5307a81` |
| 原生Flow*历史库 | `b85a3211748cb77b736fe4ad42ee02d8d2b81148`；具体库/修复变体仍按各run INPUT/SOURCE辨认 |

“我们”是复用作者引擎并加入修复/优化的派生实现，不应描述成完全独立的第四套数学算法。Huan/Xiangru默认被测核心27/29 Python文件相同、CUDA数学相同，13个可比范围文件字节相同；不能据相同hull把它们当独立正确性证明。

### 1.3 当前关键结果

| 方法 | 已有结果 | 解释 |
|---|---|---|
| Huan原作者日志 | 83.048290秒 | 历史内部时间，commit/环境不完全等于本次 |
| Huan parity+box+native-f64+SR B分块 | 1024×1000完整，5次进程中位75.250099秒 | 只改临时存储，原模式数值保持；内部中位70.727803秒 |
| Huan strict同盒分块 | 全分区前缀596，第597拒绝 | 33.764603秒是失败进程，不能当T5时间 |
| 我方修正P2 | 全前缀799，第800拒绝 | 658.477174秒失败watch |
| 我方P3 GPU14 | 完整1024×1000、50NN | 单次3153.449456秒 |
| 我方P3+sin/cos复用 | 同任务完整，所有1000 observer和101其它PT及最终状态一致 | 单次1533.752052秒；对自己旧P3约2.056× |
| 原生新初始覆盖修复变体 | **9/30已读终态：6小时时限超时** | watch21609.612654秒；30完整控制期、1024子盒各600步有记录；无T5终点结果 |

原生不是现在仍卡400步。6小时原作业已自然超时，之前“停止该活跃作业会丢进度”的情形是历史。下一次启动前仍检查原run/进程身份及输出；不得因为连接超时重复启动。24h方案已有，是否采用按新goal下实际需要和资源安排执行，另开目录，不覆盖6小时失败记录。

我方P3设置：B1024、h=.005、T5、1000步、控制周期.1、50NN，work3/point2/validation4，cap±.1、cutoff1e-6、完整SR1000/K20重组；V100 16GB GPU3、CPU14–17，GPU守卫14GiB/allocator13.5GiB/RSS11.5GiB。旧11.5GiB守卫589停止已解决，不再围绕旧停止点反复做同样的内存试验。

目前仍无完整端到端浮点NNCS证明；`fullbatch_qualification=false`、`end_to_end_strict_certificate=false`保留。更窄、accepted、VERIFIED标签不等于独立证明。

## 2. 最新 ARCH-COMP 文档与范围冻结

截至2026-09-30已找到的最新公开官方报告：**ARCH-COMP26 Category Report: Artificial Intelligence and Neural Network Control Systems (AINNCS) for Continuous and Hybrid Systems Plants**，EasyChair EPiC Series in Computing，volume110，pp85–130，2026-07-03；DOI `10.29007/637n`。

- [官方发布页](https://easychair.org/publications/paper/GsKW)
- [官方PDF](https://easychair.org/publications/paper/GsKW/download)
- [ARCH-COMP2026官方benchmark仓库](https://github.com/Kiguli/ARCH-COMP2026)，本次固定 `d55dcc39f6496720adbf8ffdb7ff8c6e04bb8f26`；报告PDF SHA256 `964326d7d418eeb9ecc258c655f358e1764fdb08fdc1d97e0a12432cc3c7fb3e`。
- 本次查阅副本和索引在本机 `output/flowstar_latest_20260930/reference/`；参考文献索引随本goal发布。第三方论文PDF保存在本地，不把它误认为本项目报告。

先冻结：报告版本/下载SHA、benchmark仓库commit、每个模型SHA、动力学源码、初集、扰动、输入布局、控制周期、时域和性质。先写“论文—官方benchmark—CROWN-Reach保存版—当前GPU实现”的对照表，再启动新版suite。

**已发现须先核实的差异**：2026报告的QUAD方程与旧冻结C++表达式存在待核对项。不得仅凭PDF抽取/截图立即宣布代码错误；逐式核原PDF、官方benchmark源码、实际解析表达式，记录坐标、符号及乘积项差异。若任务定义不同，保留“历史作者复现合同”和“2026官方合同”两份，不悄悄改掉旧模型后继续使用旧宽度/时间。历史Huan80秒研究仍以它自己的真实合同为起点。

另一个已见的合同差异：2026官方README的Airplane连续条目写`t∈[0,20]`，报告§3.7则区分连续`t∈[0,2]`和离散`k∈[0,20]`。必须核清采样周期、秒数和离散步数，不把20步自动当20秒。离散实例的ODE阶数/积分余项字段应标不适用，使用其原离散状态更新语义。

### 2.1 必须覆盖的初始清单

报告有12大类；排除VCAS后11类。结合官方仓库连续/离散分支，目前应至少覆盖以下16个配置，再按固定版本README/报告细项核对有无新增控制器或场景：

| ID | Benchmark / 变体 | 已知与旧suite的关系 |
|---|---|---|
| acc | Adaptive Cruise Control | 旧native尾项漏包，修复参照须单列 |
| airplane-continuous | Airplane连续模型 | 旧有性质早停和元数据修复 |
| airplane-discrete | Airplane离散模型 | 新范围；不能用连续积分结果代替 |
| attitude-control | Attitude Control | 旧有完整五次，需新版合同复核 |
| cartpole | CartPole / Balancing | 核实官方名称对应关系及控制器 |
| docking | Spacecraft Docking | 旧14项未覆盖；不能因无旧入口而漏掉 |
| double-pendulum-more | Double Pendulum more robust | 核实际模型，不沿用已知命名/加载错配 |
| double-pendulum-less | Double Pendulum less robust | 旧native超时，不能当完整时间 |
| nav-robust | NAV robust | 旧有加速结果，新版需复核 |
| nav-standard | NAV standard | 旧native超时；小分母宽比单独解释 |
| quad | Quadrotor | 先核2026动力学与历史作者合同差异 |
| single-pendulum | Single Pendulum | 完整新版复跑 |
| tora-homogeneous | TORA remain/safety对应版本 | 核官方命名/时域 |
| tora-relu-tanh | TORA reach ReLU/tanh | 完整新版复跑 |
| tora-sigmoid | TORA reach sigmoid | 完整新版复跑 |
| unicycle | Unicycle | 旧有实质宽差和首步诊断，继续归因 |

“16”是本次已识别的基线清单，不能代替正式manifest：配置语义/数量最终由钉住的最新官方文档和仓库决定。VCAS明确排除。发现新非VCAS条目要补入，不为了沿用旧14项代码删项目。

## 3. 阶段A：Huan QUAD 为什么快，以及 strict / parity 到底是什么

### 3.1 要回答的核心问题

最终报告开头用普通语言回答：同样写QUAD时，到底有多少时间差来自不同数值保证/阶数，有多少来自实现组织、GPU批处理、扩展缓存、SR存储、观测/日志和计时边界？目前不能断言一个单独trick解释全部差距。

逐项核查并用源码位置/实际配置说明：

- Huan83秒作者日志与本次75秒复现的模型、初盒、阶数、验证阶数、NN次数、设备与计时差别。
- `strict` / `parity`每个开关改变的具体路径：保留系数舍入、区间向外舍入、VAR截断尾项、refinement replay、point-trust、倒数余项、端点/控制注入；哪些是作者原文已说明的兼容语义，哪些是后来修复。
- `box/hybrid`、`same-slope`、`RPC32/native-f64`是不同维度，不把box=严格、hybrid=parity。P2/P3也是独立维度。
- 为什么我们最终选择P3/validation4，Huan仍P2/validation1；“阶数不同”对算子数量、显存、收紧/拒绝有什么实际影响。
- SR完整历史保留方式、B-only分块、CPU ledger/K20、CUDA Graph生命周期、临时张量物化、稀疏支持、Python与kernel调度成本。
- 预编译/预加载、JIT回退、观察器、保存/哈希、watchdog和日志的计时边界；用实测证据，不把缓存或回退猜测写成原因。

不能只解释名字或重复“合同不同”。给出一张**模式差异表**：开关→代码路径→数学含义→额外计算→已知反例/保证边界→是否可省→相关实测时间。

### 3.2 实验策略和可移植性

1. 先读取已有5次Huan全程和完整P3结果，核SHA、接受计数/NN/输出；无需无变化地再做全套旧资格。
2. 在同输入/同阶段上比较成本；采用有界profile定位Huan与P3的主要耗时，给出能覆盖整体差距的分类，不停留在一个小kernel的局部倍率。
3. 进行最少但可解释的消融，每次只改变一个因素。模式/阶数切换属于算法变体；失败的运行单列，不用未跑完的时间报全程速度。
4. 列出每个候选trick：来源、作用、正确性条件、工程工作量、实测收益、是否已接入、是否值得继续。保留全部误差和相同语义的实现优化优先。
5. 对可用的候选实际接入我们的隔离实现；先真实输入/输出一致性或独立包含检查，再短程和必要全程。算法/阶数改变不能伪装成逐位等价实现优化。
6. 不把parity默认定为作弊，也不承诺strict名字即正确。已有反例按具体路径/版本陈述；不要推广为“作者全部结果无效”。若某模式缺必要保证，忠实复现与修正版本分列。

**阶段A交付**：`HUAN_QUAD_SPEED_AND_MODES.md`，含面向用户的简明解释、源码地图、成本分解、最小消融表、可移植trick清单、已实施改动及验证、完整时间/宽度结果与仍未回答项。把strict/parity需要保留的理由说清；若能以一个清楚的算法合同替代混乱历史开关，另做有证据的整理，不继续堆模式。

## 4. 阶段B：直接可用的 flowpipe / MATLAB 绘图

原生Flow*功能已有：`Plot_Setting::plot_2D_octagon_MATLAB`，保存的QUAD示例选`t,x3`。实现位于服务器 `flowstar_mainline_causal_b85a321_20260810/flowstar-toolbox/Continuous.cpp:9261`附近，声明`Continuous.h:2856`；QUAD调用示例见旧保存`quad.cpp:286`附近。先复用已有设计和项目plot入口，不另造一套无关可视化系统。

目标是用户从一次实验输出直接得到论文风格图和MATLAB脚本。提供一个稳定的命令/函数入口，能选择benchmark/run、状态坐标、时间—状态或状态—状态、tube/endpoint、叠加方法、输出路径；具体CLI名称按现有工程结构决定。

### 4.1 必须支持

- flowpipe可达域；initial box；用户所说的安全区域（图例可写Safe region），必要时独立显示Target / Unsafe region。
- 明确性质时间语义。例如QUAD `0.94≤x3≤1.06`是终点目标时，应显示目标时刻/区间；不能画成要求整个起飞过程都在该带内的安全结论。
- MATLAB可读取的`.m`及必要小数据文件，另输出可直接查看的PDF/PNG，最好保留矢量图；坐标、单位、图例、时间和模型/方法标签完整。
- 图形数据与原数值结果分离：保存投影几何/方向界和来源索引，重画图不重跑长实验。
- 按论文所选坐标制作图，并允许额外投影；多方法图采用相同轴范围和统计语义。
- 拒绝、缺步、超时、性质早停明确标记；不插值缺失flowpipe，不把幸存子盒画成全部初集成功。

### 4.2 几何与性能要求

原生octagon用八方向的TM支持界构成多边形，保留坐标间相关性；已有逐坐标区间只能得到矩形hull，不能伪装成同精度octagon。实现可先诚实支持box，再用仍有相关性的TM求方向界实现octagon；以实际数据说明每种图的含义。

优先在接受步骤处流式导出少量投影界，避免为绘图保存/常驻全部QUAD巨大TM/SR。求解时间、观察/导出时间、MATLAB/渲染时间分开记录。抽样仅用于显示且标明采样规则；验证仍覆盖原完整任务，不能用视觉抽样代替可达性计算。

**绘图验收**：初盒/安全区域几何和时刻正确；至少QUAD以及一项状态—状态图有实际数据；一张native已有图与新导出做对应核对；导出启用前后数值状态/接受掩码保持；MATLAB脚本实际可运行时记录版本与结果，若环境无MATLAB，明确只完成脚本生成/检查和替代渲染，不声称已在MATLAB实跑。先用已存结果验证绘图，不为美化反复跑长程。

## 5. 阶段C：最新 ARCH-COMP 非VCAS全量四方复跑

先解决最新版模型/入口差异和当前可执行环境：原Unicycle Ninja/编译器加载问题已有诊断，优先还原历史环境并核SO未重建；必要时独立加载适配，不改原失败记录。原生QUAD旧6h已超时，按实际增长及资源安排足够预算的新任务，避免再用旧不够的限制得出性能结论。

每项建立机器可读manifest：官方来源/版本、方程与变量顺序、controllerSHA、初集/分区/boxSHA、扰动、h/阶数/验证阶数、控制周期/NN次数、T、cutoff、SR/余项、strict/parity、控制域/relaxation/dtype/传输、性质与检查策略、硬件/线程/资源/时限、精确命令/源码/SO身份。

四方主比较使用匹配的benchmark合同与可解释的入口。Huan/Xiangru作者默认复现可以单列；共享驱动用于控制变量，明确它不等同三套独立NNCS产品。一个方法不支持离散/模型算子时，要实际定位并接入合理实现或给出具体阻断，不能拿其它算法结果冒名顶替，也不能静默删除该行。

对每项记录：

- 全初集/所有子盒是否完成，接受前缀、首次拒绝、NN次数、性质状态；失败、资源守卫、超时、性质早停、环境错误分开。
- 完整进程wall、求解/NN/观察/输出分项、峰值CPU/GPU内存、计时边界。长任务原始耗时保留，不外推完成时间。
- endpoint和tube的每维绝对上下界/宽度；每分区mean/max与全体union分开；共同有效前缀逐时刻比较，完整终点只在全任务完成时给出。
- 小分母比率必须同时给绝对值；性质检查方法不同造成早停差异时单独解释；原生更窄不自动等于更正确（尤其ACC历史反例）。
- 对可比、完整、稳定的配置做四方轮换、同设备和线程预算的5次正式计时，预热另记；没有足够重复的长任务按实际样本数报告，不把单次写成稳定中位数。不把失败样本删到只剩好看的结果。

已有7项五次和所有旧图表可作回归参考，不能直接复用成“新版本重跑成功”。把旧14项→2026新manifest的差异表加入报告。

## 6. 阶段D：新的完整报告和最终交付

仿照2026 AINNCS报告的组织方式，写中文主体报告（术语/图例可配英文），提供可编辑`.docx`、可读PDF、Markdown及生成图表的数据/脚本。开头清楚写总体目的、方法关系、主要结果、限制；每个benchmark独立小节。

每节至少有：模型与控制器、初始集合/安全或目标性质、完整实验配置、四方完成状态、时间表、绝对宽度表/共同前缀说明、论文风格flowpipe图、对应运行目录与复现命令、失败原因或尚未解决项。

另外包含：

1. Huan约80秒的原因报告和strict/parity解释；哪些技巧已移植、带来多少端到端收益、是否改变数学算法。
2. 绘图功能用户说明与MATLAB例子，解释initial/safe/target图层和tube/endpoint。
3. 全部非VCAS配置覆盖矩阵；新增Docking、Airplane离散、模型变化与未实现边界明确列出。
4. 统一数据包：CSV/JSON、图PDF/PNG、MATLAB文件、版本/配置/SHA、单次时间样本、失败记录、小型证据manifest；大张量只给服务器路径和SHA，不伪称已下载。
5. 新实现整理在独立分支，提交并推送最终代码和报告；避免把冗余诊断/失败候选堆进默认运行入口，但保留历史可追溯性。

### 完成判据

只有当A的原因分析与移植验证、B的实际绘图功能、C的每项实际尝试/可比结果、D的完整报告与发布均完成，才把这个goal标完成。单独找到论文、写出计划、Huan75秒旧复现、某个microbenchmark变快都不构成完成。

若某项数学/软件支持存在真实不可解阻断，保留具体证据、已尝试方案与用户需要决策的选择，不捏造完整成绩；其余可独立工作继续。不要为了追求字面“全部成功”修改时域、缩小初集、删误差、改安全性质或隐藏失败。

## 7. 服务器和第一轮动作

服务器：`shengenli@chicago.huan-zhang.com:62252`，IdentityFile `~/.ssh/id_ed25519`。研究根 `N=/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`。

本次发布读取使用 `/tmp/codex-huan-2252-publish-20260930.sock`；新对话先 `ssh -S <socket> -O check -p 62252 shengenli@chicago.huan-zhang.com`，不能假定长期有效。若需用户交互认证，提供新socket的完整命令，请其在Mac本机运行；不用索取私钥内容。

第一轮执行建议：

1. 读分支README/STATUS及本goal，运行 `python3 research/gpu_verified_20260930/verify_snapshot.py` 验文件完整性。
2. 确认当前用户已经要求开始执行；读取原作业终态/资源，建立新运行目录。
3. 锁住2026报告与官方仓库版本，完成QUAD模型/方程和16项初始清单核对。
4. 复用已归档Huan与P3数据做模式/成本表，确定最有信息量的少量profile/消融；与绘图接口的只读设计可以并行。
5. 按A→B→C→D推进，给用户阶段性“得到了什么结果、下一步解决什么”的更新，不把大量内部小测试当主要进展。

新对话可以直接发：**“请读取此goal和GitHub新分支，恢复并执行这个goal，持续推进；先核对已有状态，不重复启动原实验。缺什么具体告诉我，不要猜。”**
