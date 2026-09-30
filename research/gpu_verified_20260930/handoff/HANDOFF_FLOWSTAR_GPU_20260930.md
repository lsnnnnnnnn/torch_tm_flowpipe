# PyTorch / GPU 可达性加速：新对话完整交接

整理日期：2026-09-30（Asia/Shanghai）。本文件由当前对话根据已有源码记录、实验记录、发布回执和最新汇报包整理。本次只整理本地资料，没有连接 SSH、启动实验、修改数值代码、推送仓库或改变 goal 状态。

## 0. 给接手助手的第一条信息

**最终目标仍是：把 Taylor-model 可达性计算的主要工作移到 PyTorch/GPU，在保持必要数值保证和可用范围质量的前提下，实现完整任务的实际加速，并与 Huan、Xiangru、原生 Flow* 做清楚、可信的时间和宽度比较。整个目标尚未完成。**

**当前实验暂停。** 用户最后一次实验授权的停止边界是“做完这步停一下”。该步（优化版 QUAD 全 1000 步、终态审计与归档）已完成；随后用户只要求汇报、英文 slides、中文逐字稿、解释 Huan 快速结果，以及本次交接文件。这些请求没有自动恢复实验。应用内 `get_goal` 在本次整理时返回 `paused`。新对话若收到用户“恢复/继续实验”，可按第 10 节接续；只上传本文件不等于授权启动新实验。

必须首先区分：**最新完成的我方 P3 GPU 运行**、**历史 Huan parity 快速运行**、**仍未取得终态的原生 Flow* 完整对照**。不能混在同一个速度排名中。

本文件提供可独立阅读的上下文、核心结果、执行身份与详细数值附录。原始大张量、完整源码环境仍在本机工作区和服务器；一个 Markdown 文件不会把这些资产复制进新对话。若新对话能读取同一工作区，即可按路径接续；若不能，明确指出缺失的具体文件/目录，请用户补充，不猜。附录包含已有报告正文，正文里的历史“待办”必须按本节最新暂停状态及第 10 节处理。

## 1. 用户意图、偏好和授权背景

- 用户：shengenli。原始研究对话：<https://chatgpt.com/g/g-p-6989cb31527c8191a7a1423ac417248f/c/6a8d560b-cf14-83ee-a95f-b75636f6dd3a>。
- 三个作者仓库地址已经由用户明确确认，不要因某工具返回 404 就再次怀疑地址；优先用服务器已有保存版本。
- 用户已经交付 Huan 的 `new_crown_reach.pdf`（83 页）和原完整引擎可行性 goal；材料不是缺失状态。原对话正文亦已保存。
- 用户希望持续推进真正加速，不能以“完成一个小 goal/写完报告/局部算子快”代替最终目标。用户曾指出 VDP 一两秒差别意义有限，应重点看 QUAD 及其它 ARCH-COMP benchmark。
- 用户特别要求：我们、Huan、Xiangru 先做相同实验，再分析；核查算法是否正确、是否为效果好而偷偷降任务；宽度和时间都要有实际数字、具体配置、正确停止类别。
- 用户问 Huan QUAD 很快的来源时没有额外命令/日志，已由我们从保存源码、作者记录及实测核实，不必再问同一问题。
- 用户明确允许提高显存预算，认为不应一直围绕人为显存守卫耗费时间。已从 11.5 GiB GPU 守卫提高至 14 GiB，并取得完整 P3 结果。
- 不要覆盖原工作树、原结果或别人的 GPU 作业。SSH 恢复后先查原进程和输出，断线不代表实验失败，更不代表可以再启动一份。
- 原生旧 6 小时作业的主动停止/从零改跑 24 小时曾被自动审批拒绝：没有可恢复完整 TM/SR checkpoint，停止会丢弃已有计算；当时缺少用户知情同意。未执行停止或 24h 重启。新对话先核对它是否自然结束；不要把此前一般“继续”误写成该次具体停止已经获准。
- 用户明确说“不知道/看不到就告诉我，不要瞎猜”。报告必须区分已实测、仅源码分析、仅准备、未下载和未知终态。

## 2. 本机工作区与原始材料

本机主目录：`/Users/shengenli/Documents/ChatGPT/verification`。下文用 **L** 表示这个绝对路径，用 **N** 表示服务器工作根（第 3 节）。代号仅用于本文，不是保证已设好的环境变量。

本机 `L/.git` 在本次只读检查中显示 **main、No commits yet、无 remote，研究文件全为 untracked**。这里是资料/本地开发工作区，不是已经完整克隆并同步的用户主仓库；不要在这里盲目 `git add .` 或声称最新代码都已提交。

| 内容 | 本机文件 |
|---|---|
| 最新运行交接及完整过程历史 | [EXECUTION_STATE.md](/Users/shengenli/Documents/ChatGPT/verification/EXECUTION_STATE.md)；最新状态在最上面，后面的“当前/运行中”多为历史 |
| 原目标与三个仓库核对 | [GOAL_PYTORCH_GPU_MAINLINE.md](/Users/shengenli/Documents/ChatGPT/verification/GOAL_PYTORCH_GPU_MAINLINE.md) |
| 后续总体加速目标 | [GOAL_FLOWSTAR_ACCELERATION.md](/Users/shengenli/Documents/ChatGPT/verification/GOAL_FLOWSTAR_ACCELERATION.md) |
| ARCH 四方实验目标 | [GOAL_ARCHCOMP_FOUR_WAY_COMPARISON.md](/Users/shengenli/Documents/ChatGPT/verification/GOAL_ARCHCOMP_FOUR_WAY_COMPARISON.md) |
| 当前失败优先路线的历史 goal | [GOAL_ARCHCOMP_FAILURE_RECOVERY.md](/Users/shengenli/Documents/ChatGPT/verification/GOAL_ARCHCOMP_FAILURE_RECOVERY.md)；其中早期 632/677 步等状态已被后续更新 |
| 9/24 卡点及用户恢复计划评审 | [CURRENT_BLOCKERS_20260924.md](/Users/shengenli/Documents/ChatGPT/verification/CURRENT_BLOCKERS_20260924.md)、[QUAD_RECOVERY_PLAN_REVIEW_20260925.md](/Users/shengenli/Documents/ChatGPT/verification/QUAD_RECOVERY_PLAN_REVIEW_20260925.md) |
| 原对话保存文本 | `L/planning_sources/conversation_page_1.md` 至 `conversation_page_4.md` |
| 原 9/15 goal | `L/planning_sources/goal_whole_engine_feasibility_decision_20260915.md` |
| Huan PDF 提取文本 | `L/planning_sources/new_crown_reach/full_text.txt`、`page_001.txt` 等逐页文本；原附件 `/Users/shengenli/Downloads/new_crown_reach.pdf` |
| 用户后来提供的恢复计划原文 | `L/results/quad_recovery_plan_review_20260925/USER_PLAN.md`；原附件 `/Users/shengenli/Downloads/QUAD_BLOCKER_RECOVERY_PLAN_20260924.md` |

工作区 `output/anywearg_*` 属于其它任务，不属于此次 Flow*/QUAD 主线，不要混入。

## 3. 仓库、固定版本与远端目录

这些是实际被测/保存版本，**不是 2026-09-30 刚 fetch 的上游最新 HEAD**。

服务器工作根 **N**：`/srv/local/shengenli/flowstar_acceleration_20260921T153643Z`。

| 角色 | URL / 版本 / 路径 |
|---|---|
| 用户工程 | <https://github.com/lsnnnnnnnn/torch_tm_flowpipe>；原整理分支 `codex/research-review-consolidation-20260915T161606Z`，`e4e8d95bf84edda52638a0b87782c606403732a0`；保存目录 `/srv/local/shengenli/research_review_20260915T161606Z/repo`。旧 `/srv/local/shengenli/torch_tm_flowpipe` 曾有未提交修改，不能覆盖。 |
| Huan 原版 | <https://github.com/huanzhang12/flowstar-gpu>；冻结 `d5f0b68fcd36ba5f582733624f074728fe9720d8`；原路径 `/srv/local/shengenli/flowstar-gpu`。保存严格修复资产另有 `/srv/local/shengenli/flowstar-gpu-proof-closure-20260826`，`743f6205e6408072193ad76e940e7f15030e8d3c`，不是作者默认版同义词。 |
| Xiangru | <https://github.com/xiangruzh/CROWN-Reach_Development>；来源分支 `2026_experiment`，被测 detached `1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3`；路径 `/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream`。根目录同名副本曾为更旧 `84184de`，不要误选。 |
| 官方 CROWN-Reach / ARCH 原生入口 | <https://github.com/Verified-Intelligence/CROWN-Reach>；冻结 `7b90f30831212f0c59c44c67aa428932d5307a81`，分支名不猜；官方 C++ `archcomp/*`、NN 服务与模型；服务器 `/srv/local/shengenli/CROWN-Reach`。 |
| 原生 Flow* 库的历史基准 | `b85a3211748cb77b736fe4ad42ee02d8d2b81148`；这是 Flow* 资产身份，不是上行 CROWN-Reach 仓库的 commit。最新实际库及隔离修复以各 run 的 INPUT/SOURCE/manifest 为准；独立 Git remote URL 未核实，不补猜。 |
| QUAD 作者配置/历史日志 | `/srv/local/shengenli/CROWN-Reach-GPU`，保存 `c28f0db949b87c475bf58404767b771f42d26dbe`；此配置仓库 commit 不等于当年运行的引擎 commit。 |
| ARCH 9/23 的“我们” | `N/engine_linear_leaf_v2`，`fff9d0f5742ebfa4e0652bbbf79b31bdd59ab3e3`。这是派生 Huan 引擎，inventory remote 仍是 Huan；不能当成已经推到用户仓库的 commit。 |
| 最新 QUAD 的冻结基础引擎 | `N/engine_quad_normalization_center`，分支 `codex/quad-normalization-center-20260924`，`a3fb2e94ba976aaf498c4a9cb3f98165cddcc272`。实际运行另叠加严格倒数、边界、SR、P3、存储和 trig 适配器；**只 checkout a3fb 不够复现最新运行**。 |
| Huan 快速复现隔离变体 | 分支 `codex/huan-sr-chunk-20260923`，`f83f6d84f0624608a5bdf85e5af7ac4d394e679e`；目录 `N/engine_huan_sr_chunk`；基于 d5f0b68，仅修改 SR 临时计算的 B 维分块。 |

本机精确库存：[remote_inventory.json](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/remote_inventory.json)。源关系与独立入口：[scope_and_contracts.md](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/scope_and_contracts.md)。

**已确认的 Git 发布**：9/23 汇报已推到用户仓库分支 `codex/progress-report-20260923`，commit `3edae8ce2e5a2b3e7915c645742ececcabc6b8ef`。[发布回执](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923_publication.json)。不能由此声称后来的 P3、9/30 英文 slides 或本次 handoff 已推送。当前请求只生成交接文件。

## 4. SSH、环境与恢复方式

用户给定连接信息：

```sshconfig
Host huan-chicago-2252
    HostName chicago.huan-zhang.com
    User shengenli
    Port 62252
    IdentityFile ~/.ssh/id_ed25519
    TCPKeepAlive yes
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

最后成功记录的控制 socket：`/tmp/codex-huan-2252-trig-20260929.sock`；当时 master PID 22732。**本次整理没有检查它现在是否存在或可用，旧 PID 不是当前授权进程身份。**

恢复实验后的第一步可以只检查旧控制连接：

```bash
ssh -S /tmp/codex-huan-2252-trig-20260929.sock -O check -p 62252 shengenli@chicago.huan-zhang.com
```

若不存在/不可用，先尝试已授权的正常连接；若密钥无人值守认证失败，再请用户在 **Mac 本机终端** 完成登录。需要新控制连接时，选一个确实不存在的新 socket，例如：

```bash
ssh -M -S /tmp/codex-huan-2252-handoff-20260930.sock -o ControlPersist=12h -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -N -p 62252 -i ~/.ssh/id_ed25519 shengenli@chicago.huan-zhang.com
```

用户保持该终端开启，并在另一 Mac 终端用相同 socket 跑 `-O check`。若出现 `ControlSocket ... already exists, disabling multiplexing`，先查该 socket 的 master；不要误以为新会话已建立可复用 socket，也不要随手删除活跃 socket。这里没有私钥内容或密码。

运行环境记录：Tesla V100-SXM2 16GB；Python 3.11；PyTorch 2.5.1+cu121；CUDA 扩展工具链路径曾为 `/usr/local/cuda-12.6`（缓存名也含 cu126）；Torch wheel CUDA 与 nvcc 版本是不同字段。Python：`N/nncs_env/bin/python`。原扩展缓存：`N/cache_private_entry/py311_torch251_cu126_gcc13`；Horner 缓存：`N/cache_horner_entry`。

- 最新我方 QUAD：GPU3（UUID `GPU-1ad11bb9-50d4-6b9d-22f8-fc8c33180e56`），CPU14–17，Torch/OMP/OpenBLAS 单线程。
- 新原生 QUAD：GPU2 用于 NN，CPU6–9、4 plant workers，RSS128 GiB/GPU14 GiB。
- 我方完整 P3：GPU 守卫14 GiB、allocator13.5 GiB、RSS11.5 GiB。守卫是本任务 watchdog 的人为停止线，不是显卡物理容量，也不是算法误差参数。
- 服务器共享；曾有 GPU0 等他人作业。恢复后重新检查资源，不能沿用旧“空闲”记录。不擅自停止他人进程。
- SSH 大 stdin/大 JSON 返回超时可能只是链路慢。优先服务器 CPU 就地审计、取回小 JSON/压缩证据；不要重复下载 GB 级 PT，不要将 `.partial` 当完整文件。

## 5. 最新 QUAD：准确进展与用户刚问过的问题

**主成果**：我方从早期 strict 数值失败推进到完整 B1024×1000、50 次 NN；随后在同一 P3 算法内复用重复 sin/cos，将完整单次 watch 时间从 **3153.449456 s → 1533.752052 s**，约 **2.056×**。全部 1000 observer、50 controller、50 transfer、terminal 与最终完整状态对照通过。这不是四方正式五次测速，也不是端到端 NNCS 浮点证明。

完整配置：12 物理状态，加时间/3 控制变量共16维；x1–x6 初盒[-.4,.4]，其它0；8×8×8×2×1×1=1024分区；h=.005，T=5，1000步；控制周期.1，每20步刷新，共50NN；SR容量1000保留完整历史，K20重组；cutoff1e-6，验证余项 cap±.1；box/same-slope/native-f64；冻结8dd初盒。

阶数不是固定不能改：原官方 QUAD C++ 实验 order2；当前我方修正 P2 为 work2/point1/validation3，P3 为 work3/point2/validation4。工作总次数含局部时间与空间变量。单纯加验证阶数和把工作多项式提升到三阶不是同一件事。P3 必须标成显式算法变体。

### 5.1 最新结果简表

| 实现 | 完整全分区前缀 | 耗时 | 状态 |
|---|---:|---:|---|
| Huan parity + SR B分块 | 1000 | 75.250099 s | 5次完整进程中位；内部中位70.727803 s |
| Huan strict + 同分块 | 596 | 33.764603 s | 失败进程总时长，第597首次79分区拒绝，不能算T5耗时 |
| Huan / Xiangru 历史 ARCH strict | 596 / 596 | 58.763553 / 61.563960 s | 两者失败进程，不是全程排名 |
| 我方修正 P2 全B | 799 | 658.477174 s | 第800首次12分区拒绝，813全失败；未触发资源守卫 |
| 我方 P3 旧11.5GiB守卫 | 589 | 1767.419438 s | 资源停止，不是数学拒绝 |
| 我方 P3 GPU14 | 1000 | 3153.449456 s | 完成，单次watch |
| 我方 P3 + trig复用 | 1000 | 1533.752052 s | 完成，单次watch，完整输出与上行相同 |
| Flow* 历史stream/full SR | 160已导出 | 1728.820754 s | RSS守卫，不是完整T5时间 |
| Flow* 新初始覆盖修复入口 | 最后记录400 | 未知 | 9/29 20:19:53的只读快照；终态尚未读取 |

**Huan 为什么很快、我们是否做过同样的事**：确实复现了 Huan 的默认历史 `parity + box + native-f64` 方法，5次完整。我们给他的隔离版本只加了独立 B 维128分块，避免物化巨大的 SR 临时量，Q历史/求和/子盒/时域不删，检查输出一致。那是“我们跑 Huan 的方法”，不是“我们最新 P3 strict 也用了 parity”。parity 在部分 point-trust/refinement 尾项处理上对齐原生语义，和 strict 的误差处理不同；不能靠名字认定完全等于 Flow*，也不能因更窄/更快就认定正确或作弊。通用路径实际有反例，但尚未证明这条 QUAD 快速轨迹具体漏掉真实解。

Huan 原历史日志是83.048290 s（内部、不同日期环境）；不能用83→75当本次分块的严谨加速比。Huan strict 第597失败也不能否定作者 parity 可以成功。我们 P3 的1533.75 s仍明显高于Huan记录，尚未达成总体加速目标；目前没有同误差、同阶数、同入口的完整四方性能排名。

### 5.2 关键宽度和定义

宽度=upper−lower；endpoint为步末，tube为整个时间步。`mean/max`是各子盒宽度均值/最大值；`union/hull`是全部子盒最大上界−最小下界。三者不能混用。

- T5 endpoint整体hull x3：Huan parity **0.0476872323**，我方P3 **0.0689723821**；误差/阶数合同不同，只作绝对值并列。
- 同8dd/native-f64、共同580步 endpoint分区均宽 x6：Huan strict **2.62793953**，我方P2 **0.468764091**，我方P3 **0.215671747**；该时点没有Huan逐步时间，不拿其失败总时长来比较。
- 原生修复与P3共同40步 x3 endpoint均宽：P3 **0.2144798713**，native **0.2165987523**；整体hull **0.8208427869 / 0.8229603592**。这只说明该前缀，不外推T5。
- 所有12维、更多时点、原始来源见附录A；其它benchmark宽度见附录B。

### 5.3 关键结果保存与审计范围

主报告：[TRIG_LONG1000_ACTUAL_20260929.md](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/TRIG_LONG1000_ACTUAL_20260929.md)。

远端 **H=N/runs/quad_fullbatch_p3_20260928**：

- 基线 `H/full1024_p3_gpu14_1000_v1`。
- 优化完成 `H/full1024_p3_trig_gpu14_1000_v1`，watch1660449/solver1660452在最后核查时均已退出，completed/rc0。
- 审计 `H/TRIG_GPU14_TERMINAL_REVIEW_20260929`，CPU18/CUDA不可见，唯一审计PID1711312已结束；实际读4530文件索引，1000+101 PT逐字节相同，两最终plant/14 SR-host组件与进度相同。
- `H/TRIG_GPU14_TERMINAL_REVIEW_20260929.tar.gz`：4,448,924 bytes，SHA256 `9c3ec11af528c60783fb80040e300be1305e21debeb869c7754d864979bdd5e7`。远端完整；本地只有786432B partial，**35成员整包本地未完整验证**。
- 关键run与审计JSON已单独完整下载并核SHA，位于 `L/results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929`。旧基线8个实际observer已在本地重算576宽度；新候选大PT/SR主要在远端，不说全部已下载。
- `fullbatch_qualification=false`、`end_to_end_strict_certificate=false`、总体goal未完成仍保留。“数值完成与轨迹一致”不等于完整系统安全证明。

精确执行以 [quad_trig_INPUT.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_trig_INPUT.json) 与 [quad_trig_COMMAND.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_trig_COMMAND.json) 为准。COMMAND 是启动前的冻结文件，内部 `launched=false` 等准备字段不会自动更新；真实是否执行以 LAUNCH、watch、RESULT 和终态审计为准，不能据静态字段否定已完成实验。

## 6. 正确性审查：已经发现的问题与尚未证明的部分

所有作者实现与Flow*按同一标准审查；“strict”名字、VERIFIED日志、模拟点落在范围内、宽度很接近，都不是完整程序正确性的证明。

1. **VAR截断尾项与ACC**：原生/parity refinement存在遗漏VAR尾项的局部反例。实际旧原生ACC在t=.1/.2排除了合法解析轨迹，独立最小例复现。隔离尾项修复后跑完50步、检查的解析见证通过。旧“GPU宽约7.94倍”不能解释成相对正确Flow*精度差八倍。修复native step2 a_lead宽0.002247354391，H/X strict0.002247008832；其它tube差仍待查。
2. **通用倒数**：未改H/X CPU sparse归档实际算 `1/(2+z/4)`、z∈[-1,1]、阶1–4、零余项/cutoff0，`bad=false`却漏掉z=-1处精确4/7；我方旧通用路径同问题。已做有限几何恒等式严格尾项适配器并检查实际调用路线。不能把CPU通用反例直接当QUAD实轨漏包证明，也不能说所有dense/历史replay都已自动修好。
3. **原生初始覆盖**：8dd盒原native构造器有338分区/384下端点少包1ULP。保留原中心、半径取两侧向外距离max的新变体通过16,384坐标实际Fraction门，只384半径扩大。原生完整对照必须标这个变体。Unicycle旧native首步也发现x2少包1ULP，但不能据此解释最终3.5倍宽差。
4. **严格边界**：后续适配器给端点代入、控制注入等舍入误差收费，并保留完整SR；早期plain/RN边界的局部长程不能由后来的修复追溯认证。
5. **控制器证书**：CROWN/same-slope浮点界、实际接口全链路尚无完整端到端资格。两次native实际NN输入重放得到86,016个T/L/U binary64值逐字节相等，只限这两次输入，不等于所有闭环输入或NN正确性已证明。

来源与反例范围在附录C完整保留。算法变体须单列，不能删除失败记录或用幸存子盒代表完整初集。

## 7. 已走过的主要路线，避免重新绕回旧卡点

| 阶段 / 路线 | 已知结果 / 当前含义 |
|---|---|
| 9/21完整引擎可行性 | 已完成VDP/Bruss、支持入口和配对；只是早期里程碑。极大的旧Python整体实现倍率不是相对原生Flow*或纯CPU→GPU倍率。 |
| 9/22各实现/NNCS接通 | 已做Huan/Xiangru独立入口、共同NN驱动、正式时间、宽度。旧汇报在9/23发布，详见output/progress_review_20260923。 |
| Huan QUAD SR分块 | 已复现parity全程75秒；该任务不用从头核实“能否跑”。strict模式失败另保留。 |
| strict自映射/加权/重定中心/归一化 | 早期631/632提高至676/677/678附近，但不能救所有后期输入；证据在results/quad_*_20260923/24。 |
| 用户9/24恢复计划 | 已审最新输入分母不近零、合法点超cap、x12不变量与结构坐标限制。旧固定表达式求值分域≠子域重建p；不能当已等价试过。9/25结论见QUAD_RECOVERY_PLAN_REVIEW，后来部分方向继续实施。 |
| 两叶/局部高阶/控制交接/倒数 | 实际做过root1/B2等诊断；689/690是局部路线的旧状态，不能冒充全B结果。倒数修复自身未延长该次689前缀；不能无依据重复只换poly取界。 |
| SR完整因子K20重组 | 保留误差和全部历史的严格区间重组，两叶P3跑到1000；host ledger/stream完整恢复后接回B1024。不是删SR或缩历史。 |
| 修正全B P2 | 799全接受；第800十二lane拒绝，第813全失败。已保存真实799完整父快照、cold799→800与trace/decompose，原失败复现。失败后的800状态不能当799父状态。 |
| 第800残差/阶数 | 原P2仅改验证r1/r2/r3不能解；空间二次残差/严格点见证超cap。P3补项局部12/12通过后才接完整P3。 |
| P3存储/图缓存 | 多个32/64/128块、释放未用图、选择性审计和working-eager诊断做过；曾完成1/2/6/9/19步后守卫。最新路线已通过提高GPU预算完成1000，不应仍把那些早期资源停止写成当前主阻塞。 |
| 最新性能定位 | profile40后20步普通+加权区间验证约97.38% advance；CUPTI真实输入见大量小kernel，fill/区间sum/copy成本。局部仪器时间不外推全1000。 |
| trig直接变量复用 | 固定真实输入门→40步完整输出→自身cold20→21→1000→实际CPU终态审计均完成；最新稳定起点。接手后不用再做一遍全部旧准入。 |

历史路径索引：`L/results/quad_targeted_recovery_20260927`，本地入口/适配器 `L/development/quad_fullbatch_p3_20260928`、`quad_fullbatch_sr_20260928`、`quad_sr_reassociate_20260928`、`quad_reciprocal_20260928`。完整旧状态可查 EXECUTION_STATE；按日期和实际 RESULT 判断，不被标题“最新”误导。

## 8. 其它 benchmark 的当前含义

附录B保留完整时间、宽度、配置、停止类别和全维终点表。最重要的边界：**14个ARCH配置矩阵与7个五次正式时间，是9/23冻结旧引擎的结果；最新QUAD P3/trig没有重跑整套。**

- 7项×4方法×5正式样本=140（另28预跑）；时间含新进程初始化，不含首次编译；宽度来自独立观察运行。
- NAV robust：Flow*68.163522s / 我们9.729652s，约7.00575×，整体union接近；TORA ReLU/tanh与sigmoid约1.12×/1.11×；Unicycle约1.55×但宽度明显差；Attitude/单摆完整进程不占优。
- Huan与Xiangru13个可比范围文件字节相同，27/29核心Python相同，CUDA数学相同；共同默认路径同源，不是两套独立算法互证。可选batched路径和作者其它分支未由此认证。
- NAV standard我方单分区最大比17.0645来自约1e-15小分母；整体union最大约1.00959，不能说整体宽17倍。
- Airplane：积分前高维元数据RSS失败已用稀疏元数据路线跨过；后续实际NNCS完成到78步性质早停，并用后半步unsafe见证解释。未做满200步、未恢复H/X四方五次测速；不是积分数值失败。
- ACC：原生基线漏包已纠正；新候选验证策略的宽度收益与速度成本分别记录，别把更窄当更快。
- Unicycle：9/23完整500步已做；9/29原生首步五个内部TM/SR边界已取得。新三GPU首步对照只启动ours，在加载扩展时退出；H/X未启动。Ninja缺失是CPU复现阻断，原三个SO直接导入成功。尚未实施加载修复；不能报新三方数值结论。
- DP less/NAV standard原生300秒超时；CartPole有数值拒绝；DP more性质早停及加载模型命名差异保留；TORA homogeneous我们/native完整、H/X拒绝。VCAS没有同类C++reach入口，不冒充第15项。

## 9. 原生 QUAD 遗留作业：恢复时先读这一节

远端：`N/runs/native_quad_matched_20260929/initial_affine_cover_variant/full1000_v1`，watch目录同名前缀加`_watch`。

最后保存快照 **2026-09-29 20:19:53 中国时间（12:19:53Z）**：1024分区共同400/1000步，409600接受lane-steps，20完整控制期，SR400；没有终态 RESULT/watch。历史PID：watch1359708、supervisor1359709、NN1359714、solver1359789。这仅是身份定位线索，恢复时需核启动时间、cmdline、进程组、run路径；不能只凭PID发信号。

快照：[PAUSE_HANDOFF_READONLY_20260929_v1.json](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/native_full1024_preparation_20260929/evidence_full_cover/PAUSE_HANDOFF_READONLY_20260929_v1.json)。

原任务有21600s native/21660s watch上限，4workers，完整SR1000。保存的控制周期耗时随历史增加；当时四核基本忙，NN调用短，不能简单归因于“别人抢GPU”。不能把我方52.56分钟误说成原生预计完成时间。

没有运行时延期接口，也没有完整TM/SR checkpoint；period/range日志不能直接恢复演化。24h独立方案已准备并上传，但**没有执行停止或新启动**。[NATIVE_24H_RESOURCE_ONLY_PLAN.md](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/native_full1024_preparation_20260929/NATIVE_24H_RESOURCE_ONLY_PLAN.md)保留具体资源、源码、审核与拒绝边界。

恢复后：若自然结束，先读终态/日志判断完成、超时还是数值失败，并保留原前缀；若仍在跑，只读核查，不重复启动。是否需要新24h任务，要依据现场状态和用户新的恢复范围决定；主动丢弃仍在跑的不可恢复进度需处理此前具体审批边界，不能绕过。

## 10. 用户明确恢复后的下一步顺序

这是接续建议，不是声称这些步骤已实施。

1. **恢复实际状态**：读本文→最新EXECUTION_STATE顶部→关键INPUT/RESULT；核SSH，再查原生旧run终态与服务器资源。先拿已经算出的结果，不重新跑已结束GPU1000。
2. **补原生完整QUAD对照**：若原run完成，审计1000×1024、50NN、全分区接受、完整宽度、时间边界与源码；若超时，按已有24h资源方案在新目录处理，保留旧run。先有完整原生结果再谈全程速度。
3. **把Huan快/我方慢的比较设为可回答的问题**：冻结同模型、请求盒、控制器、阶数、误差处理、SR、资源、观察和计时边界；区分作者忠实复现与修正算法变体。若做消融，每次只改可解释的一项，不能为了75秒关掉必要尾项后称严格成功。
4. **推进Unicycle最早差异定位**：先按已保存方案找回历史Ninja/CC/CXX/PATH，验证复用原SO且不重建；若不可行，用独立加载适配器只加载钉SHA库。新目录重新运行原准备好的三GPU首步，和已有原生五边界对照；再追ours相对H/X第4步额外宽差。避免从头跑500步后继续猜。
5. **继续有证据的GPU热点优化**：从已资格P3+trig出发，普通/加权验证及大量小操作是已实测热点。新的等价优化先真实输入检查，再短程/自身恢复/必要完整运行；不重复已通过旧测试或无限审计。只有出现新变化/失败才扩大测试。
6. **扩大最新版本的回归矩阵**：以已冻结7个全程案例优先，单列原生ACC修复参照和Airplane检查策略；时间、宽度、完成性一起验收。最新P3未整套重跑这一缺口必须关闭后才能说通用加速。
7. **代码收敛与交付**：最新运行当前依赖冻结engine和多层独立adapter。先列全source/SO pins和实际入口，再把已验证改动整理为可维护入口，在用户工程隔离分支提交；不能把9/23已发布报告分支当最新实现。后续发布按用户要求处理。

未达验收：完整同口径四方QUAD、最新算法其它benchmark回归、完整CROWN/注入浮点证书、正式隔离重复时间、最新代码整理/发布、剩余宽度差原因。用户目前要求的是交接，以上均未自动启动。

## 11. 已交付汇报和继续编辑的文件

最新英文42页Beamer（27主讲+15参考），中文逐字稿13页Word；依据保存数据翻译与排版，**没有重新实验**。英文PDF、原生LaTeX编辑器编译及页图检查通过；78份来源副本SHA核对通过。

- [英文slides.pdf](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/slides.pdf)
- [可编辑slides.tex](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/slides.tex)
- [中文逐字稿.docx](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/chinese_verbatim_script.docx)
- [中文逐字稿Markdown](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/CHINESE_VERBATIM_SCRIPT.md)
- [英文TeX完整bundle.zip](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english_tex_bundle.zip)
- 该目录的 `SOURCES.json`、`SOURCES.md`、`SHA256SUMS.json`、`data/`、`configs/`、`evidence/` 保存数据源与哈希。
- 旧9/23已推送汇报：`L/output/progress_review_20260923`；中文新增汇报旧版：`L/output/flowstar_progress_since_slides_20260929`。

不要在后续编辑时把中文旧版误当最新英文源，不要把zip当完整可运行solver环境。新对话如改LaTeX，继续编辑现有slides.tex、使用内置LaTeX编译和预览；文档内容更新要同步脚本与证据口径。

## 12. 可以直接复制到新对话的开场文字

> 请先完整阅读我上传的 HANDOFF_FLOWSTAR_GPU_20260930.md，保留里面的目标、已完成结果、未知项和停止边界。本机工作区是 /Users/shengenli/Documents/ChatGPT/verification，服务器研究根目录是 /srv/local/shengenli/flowstar_acceleration_20260921T153643Z。当前实验仍暂停，先确认你理解进展和下一步；我明确说恢复实验后，再先核对原生 QUAD 原作业的实际终态，避免重复启动。仓库和文档已经有记录，看不到哪份就具体告诉我，不要猜。我们的最终目标仍是可靠的 PyTorch/GPU 完整加速，以及与 Huan、Xiangru、Flow* 的时间和宽度比较。

若用户希望新对话立即恢复，可把其中“当前实验仍暂停……”这一句替换为：“现在恢复实验，请按交接第10节推进，首先只读核查原生旧作业，保留所有已有结果。”具体停止不可恢复活跃作业的审批边界仍按第9节处理。

## 13. 关键证据索引与阅读优先级

第一次接手先读第0–5、9–10节；需要科学结论时读附录A–C；执行前读实际INPUT/COMMAND/RESULT，完整历史按需查，别把上百条历史过程当当前待办。

1. `L/EXECUTION_STATE.md`：最新暂停、终态/归档边界、所有旧进度。
2. `L/results/quad_targeted_recovery_20260927/TRIG_LONG1000_ACTUAL_20260929.md`：最新完整优化结果。
3. `L/results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/REPORT.md`：原P3成功、性能定位、原生/Unicycle进展。
4. `L/results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929`：实际最新结果/审计/manifest和本地下载范围。
5. `L/results/quad_targeted_recovery_20260927/native_full1024_preparation_20260929`：原生入口、覆盖修复、short40/NN对照、full旧作业/24h计划。
6. `L/results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929`：初始覆盖独立审查。
7. `L/results/quad_targeted_recovery_20260927/FAIL800_DIAGNOSTIC_REPORT_20260928.md`：P2失败原因、局部验证范围。
8. `L/results/quad_targeted_recovery_20260927/ORDER_CONFIGURATION_REVIEW.md`、`P2_P3_STRICT_BOUNDARY_REVIEW.md`：阶数和严格边界。
9. `L/results/quad_targeted_recovery_20260927/SR_REASSOCIATION_QUAD_20260928.md`、`FULLBATCH_SR_RESULTS_20260928.md`：SR重组与全B P2。
10. `L/results/archcomp_failure_20260923/REPORT.md`：Huan快速结果复现、源码/5次记录与原strict失败；旧状态按后续覆盖。
11. `L/results/archcomp_review_20260923/report`：14配置、时间样本、宽度表；`SOURCE_AUDIT.json`和`remote_inventory.json`位于上一级。
12. `L/results/quad_residual_memory_20260923`：ACC反例、修复和后续宽度。
13. `L/results/airplane_nncs_recovery_20260923`、`airplane_stop_witness_20260924`、`airplane_gpu_stop_20260924`：Airplane真实NNCS/性质停止。
14. `L/results/unicycle_first_gap_20260929`及`L/results/quad_targeted_recovery_20260927/UNICYCLE_FIRST_GAP_REVIEW_20260929.md`：原生首步、GPU加载阻断。
15. `L/output/flowstar_progress_since_slides_20260930_english/SOURCES.json`：78份汇报源文件的原路径、复制路径和SHA映射。

以下附录为交接时嵌入的证据报告正文，保留完整数字以避免新对话只拿到摘要。来源身份与文件完整性摘要附在末尾。报告标题中的旧日期和旧状态须按上文时间界定。


---

## 附录A：QUAD 完整时间、宽度与原始数据来源

以下正文复制自 [quad_evidence.md](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_evidence.md)，SHA256 `5607b7ef6a5304498fca575b6302e1478c66db94fc159da3dde0a7d12cb173db`。未改写实验数值。

## QUAD：相对于2026-09-23 slides的新增证据

范围：仅本地归档提取，不SSH、不启动实验。本材料基线是 `output/progress_review_20260923/slides.tex`（其中原QUAD仍列为规模/配置限制），不是把9月23日前已经展示的VDP/Bruss/TORA成绩再次算作新成果。来源ID、SHA与原始表列完整保存在同目录 `quad_evidence.json`。

### 建议主线（3–4页）

1. **先把任务算完整，再优化速度。** 完整1024初始分区的修正P2由旧共同前缀677推进到799；显式P3变体解决后期拒绝，高预算运行完成1000；最后直接sin/cos复用把完整时间减半且所有输出字节不变。
2. **Huan“很快”已真实复现。** 原作者parity＋只改临时SR存储的分块版，5次全部完整，进程中位75.250秒。它不是strict误差合同；不能用strict失败否定parity，也不能用parity完成追认严格证明。
3. **现在能展示具体同一步宽度，不能展示四方完整strict速度排名。** 第580步有相同8dd盒/native-f64的Huan strict、我们P2/P3均宽/max；第100步有四方历史union参考；T5只有Huan parity和我们P3系列可展示。
4. **性能结果最扎实的部分是P3对P3。** 1000步、50NN、同模型/同阶数/同SR/同资源守卫，watch3153.449→1533.752秒（2.056×），1000个observer和101个其他PT文件逐字节相同。不是全部算法都更快，也不是新的NNCS端到端证书。

### 统一配置与必须保留的差别

名义任务是QUAD 12物理状态＋时间/3控制变量（16维）；x1–x6初始[-.4,.4]，其余0；8×8×8×2×1×1=1024分区；h=.005，T=5，1000步，控制周期.1（每20步），50次NN，SR容量1000，cutoff1e-6、余项cap±.1，冻结网络SHA fabd84e…，box/same-slope。当前我们P2/P3和Huan同盒复现用8dd初始数组/native-f64；历史ARCH Huan/Xiangru/Flow*用f90数组/RPC-float32。4224初盒端点存在末位差（最大8.33e-17），因此后者不是逐位同盒对照。

计时环境记录为Tesla V100 16GB / PyTorch2.5.1+cu121；我方完整P3使用GPU3、CPU14–17，torch线程1；新native使用GPU2、CPU6–9和4植物worker。P3新旧完整版均GPU14GiB守卫/allocator13.5GiB/RSS11.5GiB；原生新full使用RSS128GiB/GPU14GiB，和旧11.5GiB RSS失败不是相同资源条件。

Huan作者P2验证RHS1；我方修正P2为working2/point1/validation3，P3为working3/point2/validation4，另含严格倒数、端点/控制注入舍入收费及完整SR K20重组。不能仅凭“P2”名称判为相同算法。Huan与Xiangru历史同值来自共享默认核心和同一NN驱动；不能当作两套独立算法的交叉证明。一般倒数反例针对所归档版本，不代表已证明该条QUAD轨迹漏包。

### 可直接上slides：完成程度与计时

不同完成范围的秒数分开解释，**不可在此整表计算排名或加速比**。

| 方法/模式 | 全1024共同接受步 | 时间/秒 | 时间口径/状态 |
|---|---:|---:|---|
| Huan parity + B-only SR chunk | 1000 | 75.250099 | 5-repeat full process median [Q1,Q2] |
| Huan strict + B-only SR chunk | 596 | 33.764603 | failure process total, not T5 [Q1] |
| Huan ARCH strict | 596 | 58.763553 | failure process total, not T5 [Q1] |
| Xiangru ARCH strict | 596 | 61.563960 | failure process total, not T5 [Q1] |
| Ours corrected P2 full batch | 799 | 658.477174 | failure watch total, not T5 [Q6,Q18] |
| Ours P3 old GPU11.5GiB guard | 589 | 1767.419438 | resource-stopped watch total, not T5 [Q6] |
| Ours P3 GPU14GiB | 1000 | 3153.449456 | single full watch total [Q8,Q20] |
| Ours P3 + direct trig reuse | 1000 | 1533.752052 | single full watch total [Q11,Q12,Q14,Q23] |
| Flow* historical stream/full SR | 160 | 1728.820754 | RSS-guard process total;160 complete exported steps [Q1] |
| Flow* new initial-affine-cover variant | 400 | — | last local read-only snapshot at2026-09-29T12:19:53Z; terminal unknown [Q15] |

Huan parity另有作者内部中位70.727803秒、driver调用中位72.173857秒；保留三种计时边界。其5次所有非计时指标相同。当前P3完整性能是单次完成性运行，包含日志、检查、初始化/图准备等，另有native在不同GPU/CPU并行，不能替代隔离重复计时。旧P3的589停止是11.5GiB人为显存守卫，GPU预算提高到14GiB后完成；这一步本身不是数学修复或提速。最新native400仅是北京时间2026-09-29 20:19:53的存档快照，不能说现在仍在400或已经结束。

### 四方同一t=0.5：endpoint总体union宽度（描述性参考）

旧100步P3与后来完整P3的observer100 PT SHA均为27a7f006…，表中36个P3 endpoint统计也与完整P3表精确相同。均值不是hull；以下都是最大上界−最小下界。ARCH三方与我们初盒/RPC及数学阶数仍不同，没有匹配的同前缀计时。不绘制把它们视为公平算法排名的柱图。[Q4/Q5]

| 维度 | 我们P3 | 我们P2 | Huan ARCH strict | Xiangru ARCH strict | Flow* ARCH |
|---|---:|---:|---:|---:|---:|
| x1 | 1.2094829 | 1.21096548 | 1.21269684 | 1.21269684 | 1.21501121 |
| x2 | 1.20869062 | 1.21173752 | 1.21619678 | 1.21619678 | 1.21431033 |
| x3 | 0.493475432 | 0.4983923 | 0.501518598 | 0.501518598 | 0.496225632 |
| x4 | 0.825957841 | 0.830255405 | 0.834660163 | 0.834660163 | 0.83485789 |
| x5 | 0.857812185 | 0.874356179 | 0.892896338 | 0.892896338 | 0.867656637 |
| x6 | 1.52604061 | 1.53374833 | 1.53895609 | 1.53895609 | 1.52948766 |
| x7 | 0.0123784314 | 0.0148458469 | 0.0174706922 | 0.0174706922 | 0.0124067843 |
| x8 | 0.0116692321 | 0.014057863 | 0.0165277957 | 0.0165277957 | 0.0117015144 |
| x9 | 0.000141791266 | 0.000168953418 | 0.000215318062 | 0.000215318062 | 0.000129809914 |
| x10 | 0.210014865 | 0.213901938 | 0.218655262 | 0.218655262 | 0.210297889 |
| x11 | 0.197841215 | 0.201549386 | 0.205980906 | 0.205980906 | 0.198103366 |
| x12 | 4.45014772e-308 | 4.45014772e-308 | 0 | 0 | 0 |

### 更严格的同盒t=2.9：每分区endpoint宽度均值

三者均在第580步完整接受1024分区；8dd盒和native-f64一致，算法仍不同。Huan这里只存mean/max，没有总体union或tube；Flow*全1024没有580导出，所以不补值。Huan这一时点没有逐步计时，不拿其599尝试失败总33.765秒当580步耗时。下表适合主slides选x5/x6/x7，其余放附录。[Q6/Q7/Q19]

| 维度 | Huan strict mean | 我们P2 mean | 我们P3 mean |
|---|---:|---:|---:|
| x1 | 2.95059418 | 2.2072176 | 1.822787 |
| x2 | 4.93156584 | 3.75722031 | 3.20070648 |
| x3 | 0.800737104 | 0.19008228 | 0.0922670086 |
| x4 | 2.39121652 | 1.27872459 | 0.838570382 |
| x5 | 4.51963104 | 2.00271607 | 1.4013097 |
| x6 | 2.62793953 | 0.468764091 | 0.215671747 |
| x7 | 0.402382703 | 0.0639981183 | 0.0187245045 |
| x8 | 0.37797573 | 0.059692855 | 0.0178334829 |
| x9 | 0.114547624 | 0.0126888279 | 0.00435001944 |
| x10 | 4.71870951 | 0.676147481 | 0.301811652 |
| x11 | 4.45710235 | 0.606008787 | 0.26108896 |
| x12 | 0 | 4.45014772e-308 | 4.45014772e-308 |

来源历史P3在589守卫后停止，但该580实际observer已本地重算；后来完整高预算P3前589份observer与旧49da运行字节相同，新trig全1000与完整P3相同。这里不能移植旧589运行的1621.385秒prefix580 advance当作新trig的580耗时。P2同前580 advance289.214933秒仅为已存的不同变体诊断分项，不计算速度比。

### 完整t=5的endpoint总体union宽度

相同名义T5和请求盒，但Huan parity与我方P3的余项/阶数/边界合同不同。它们可并列展示真实绝对值，**更窄不代表更严格或更正确**。Huan strict、Xiangru strict、Flow*的完整T5均缺失，不填数字。x12常量舍入底噪，不生成宽度比。[Q3/Q8/Q9/Q12]

| 维度 | Huan parity | 我们P3（含新trig） |
|---|---:|---:|
| x1 | 6.49970944 | 6.62362135 |
| x2 | 7.05301872 | 7.25355368 |
| x3 | 0.0476872323 | 0.0689723821 |
| x4 | 1.42263174 | 1.50395626 |
| x5 | 1.6005968 | 1.72207246 |
| x6 | 0.123350957 | 0.179927098 |
| x7 | 0.00701656253 | 0.0108531397 |
| x8 | 0.00611990429 | 0.00971082782 |
| x9 | 0.00444940679 | 0.00614348764 |
| x10 | 0.0861795769 | 0.15565646 |
| x11 | 0.0710206706 | 0.130841171 |
| x12 | 0 | 4.45014772e-308 |

P2/P3共同第799步另有72项mean/max/union×tube/endpoint数据[Q10]：例如endpoint mean x6从P2的3.869302188到P3的0.196513515；x7从0.535247686到0.014142749。原P2第800开始拒绝，不能将799之后幸存分区hull包装成全初始集覆盖。

### Native Flow*最新修复入口（单独标注变体）

原始8dd/native-f64原库短程40步虽然接受，实际仿射初值有338个分区/384下端点欠包1ULP，故未准入完整运行。独立修复保留输入盒/中心/原库，只用两侧向外距离最大值确定半径；实际16384坐标Fraction包含检查通过，只有384半径扩大。修复后short40的40960范围记录、2NN、SR20/40全部通过；两次同实际NN输入经原driver重放，86016个T/L/U binary64结果全相等，限定为这两次输入。旧库RN边界/VAR尾项等仍未取得全链路证明。[Q16/Q17/Q22]

同40步可示例：endpoint均宽x3，我们P3=0.2144798713、native修复P2=0.2165987523；总体hull分别0.8208427869/0.8229603592。完整72项在Q17。40步外层wall为我们诊断P3 155.451914秒、native修复171.138231秒；观察/保存开销不同，不报公平加速比，更不能外推T5。新native全程原作业仅存档400/1000步，终态与T5时间/宽度未知。

### 审核/归档边界，建议末页一句话

已完成的是“条件性数值轨迹与实现等价核验”：初始化Fraction包含、真实边界冷恢复、完整steps/acceptance、实际PT/SR对照均有保存证据；不是用少跑分区、去掉误差项、抬高余项上限换成功。但控制器CROWN/same-slope浮点证书仍为条件性，`end_to_end_strict_certificate=false`/`fullbatch_qualification=false`保留，原driver的VERIFIED不能扩成完整NNCS证明。[Q12/Q21/Q23]

最新trig远端CPU实际核验4530文件，1000observer+101其他PT字节相同、两个最终plant/14SR组件/进度相同。8关键JSON已本地下齐并SHA核验；完整35成员小包本地仅partial，候选8 PT/SR/plant保持远端，不能写成“所有原始张量已下载本地”。旧完整P3本地8 actual observer已重算576宽度，其表通过全1000实际配对适用于trig。实验保持暂停。

### 原始来源清单

- **Q1** `results/quad_targeted_recovery_20260927/fullbatch_baseline_review_20260928/RESULT.json`；SHA256 `71c122d0ced5a5445a7b043dc3c4c16d77368c3abced7373f645b794432ddfe4`。
- **Q2** `results/archcomp_failure_20260923/evidence/huan_parity_repeats/CAMPAIGN.json`；SHA256 `8651b97943518f6d16e6d73787df624292b3b903ddb9001c5d578cb823144947`。
- **Q3** `results/archcomp_failure_20260923/evidence/huan_box_parity_chunk/metrics.json`；SHA256 `084ce1510d71572dd1d2415ed5a8175a003d2e29dba0cc835b8fff7db92c5186`。
- **Q4** `results/quad_targeted_recovery_20260927/p3_common100_comparison_20260929/RESULT.json`；SHA256 `1ed8863105268f56af41c698d51fc2c398818e8db067da31362559fc45b49dbf`。
- **Q5** `results/quad_targeted_recovery_20260927/p3_common100_comparison_20260929/WIDTHS.csv`；SHA256 `27b454cc97c5d2442d58240b26c2df9ea5940322812f02404bdbeb792bcf5be1`。
- **Q6** `results/quad_targeted_recovery_20260927/p3_common580_comparison_20260929/RESULT.json`；SHA256 `06354feb33ccdd2fbfad22e1eaa80217b221fdc1a9e2a48556371f41f7fd541e`。
- **Q7** `results/quad_targeted_recovery_20260927/p3_common580_comparison_20260929/WIDTHS.csv`；SHA256 `64f846958294058e0f713037b8402a68763a6c866e5b98c90fca61f6254f628f`。
- **Q8** `results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/RESULT.json`；SHA256 `bb9ed40ae642162a9e9376fc0d04bb88ad0f201fb93702bd459229f23d8a84fa`。
- **Q9** `results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/WIDTHS.csv`；SHA256 `f673126d8e8e31b9414dd247dda95e5058e7aa554c754548c6ae0a6913d3bc08`。
- **Q10** `results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/COMMON799_WIDTHS.csv`；SHA256 `8eca27ab682df8cf49cac05ae14901fe0c13af05f73638f8201fa2fd3837b1d5`。
- **Q11** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/full1024_p3_trig_gpu14_1000_v1/RESULT.json`；SHA256 `310bb9acb6780845c2d336e96de344e25bfbb2916cd9b721362296fc75c8ae98`。
- **Q12** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/TRIG_GPU14_TERMINAL_REVIEW_20260929/RESULT.json`；SHA256 `517bd183820d43af794d92b64434d84e05e9fb4fa120927f2e9bf3c41968add2`。
- **Q13** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/LOCAL_VERIFICATION.json`；SHA256 `1b4a6b2d9a0425083e67f3ec5e4dbad68ec22c766aad2f1016422e515a0750de`。
- **Q14** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/TIME_COMPARISON.csv`；SHA256 `614e60f787350946d1684f551a129a09c961d0c2d13fdd8907f92dc3574731f1`。
- **Q15** `results/quad_targeted_recovery_20260927/native_full1024_preparation_20260929/evidence_full_cover/PAUSE_HANDOFF_READONLY_20260929_v1.json`；SHA256 `3f240c731298dce6b2174d496ba052a35a5d1c69b2bce7daf7a4f137636fd865`。
- **Q16** `results/quad_targeted_recovery_20260927/native_cover_P3_common40_20260929/RESULT.json`；SHA256 `47a6e63460c24fcdaafe21c373f97e39497dc03f8bc609d4765bd16aadba6d33`。
- **Q17** `results/quad_targeted_recovery_20260927/native_cover_P3_common40_20260929/WIDTHS.csv`；SHA256 `65ecb45faf3e786e2faf45ef8ae5f3480d0240a3920a41ef05510b7a2f21ad9f`。
- **Q18** `results/quad_targeted_recovery_20260927/evidence_fullbatch_long_logs_20260928/full1024_p2_1000_v1/RESULT.json`；SHA256 `bbe8e4fd243c02202767e2619ed03ac8df0661b29e9f5a7a00a308d317ef4dbd`。
- **Q19** `results/quad_targeted_recovery_20260927/p3_common580_comparison_20260929/INDEPENDENT_REVIEW.md`；SHA256 `d3b7be88fb009e4deb81fee7a4b2efa9fe5bab2a8a0b3ebc7b5d6682142d4777`。
- **Q20** `results/quad_targeted_recovery_20260927/gpu14_full1000_summary_20260929/INDEPENDENT_REVIEW.json`；SHA256 `d854d2f6cd3f467e1f2a12ab3a2735e6c87300ef464cb50d3d93e66dc3337494`。
- **Q21** `results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929/GPU_INITIAL_REVIEW.md`；SHA256 `28a5e6e0ac3487711c3e7600d414eacbc64e1d257dc4c5748ab68bd8de04104d`。
- **Q22** `results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929/REVIEW.md`；SHA256 `0436dca7cca6168b9d494a382d2e2f0fe1d48b12bb311b5af19a2ab20ab5301a`。
- **Q23** `results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/ROOT_AUDIT_REVIEW.json`；SHA256 `5ce9c48388673c495ff47e4bf4dc8dc775a93d00030760899bf3f2182977c0ad`。


---

## 附录B：QUAD 以外完整时间、宽度、配置与停止原因

以下正文复制自 [other_benchmarks_evidence.md](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/other_benchmarks_evidence.md)，SHA256 `4e990e8880ce7e94607330931156297c5270b78da14823188bc939fe922d046e`。未改写实验数值。

## 上次 slides 之后：QUAD 之外的基准证据整理

整理日期：2026-09-29；仅只读本地资料并生成本文，没有 SSH、实验、源码修改或重新计时。实验保持暂停。

### 1. 时间边界与应讲清楚的增量

按 [上次 slides](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/slides.tex) 为起点；虽然新旧目录均含 20260923，ARCH-COMP suite 是上次交付之后补做的更广四方对照。旧 slides 的 VDP/Brusselator、TORA homogeneous、Single Pendulum、CartPole 不是这次新加的算法成果。新 suite 采用官方 C++ 合同重新对齐；不得把旧 slides 的 1.802 倍 TORA 数值与新套件的计时与观察口径拼成同一趋势。

本次最重要的增量是：官方 14 个配置（本文列 QUAD 外 13 个）完成四方筛查；其中 7 个达到完整时域且各方完成 5 次正式计时。Airplane 从积分前元数据内存失败推进到真实 NNCS 的第 78 步性质早停，并解释了停止差异。ACC 找到原生参照真实漏包并在隔离库修复；这撤销了“GPU 精度比正确 Flow* 差约 8 倍”的解释。Unicycle 已找到首个宽差及部分观察器保守性的原因，但三 GPU 同注入单步归因尚未完成。

**当前新 QUAD P3/三角复用引擎并未重跑这些完整基准。** 下述五次时间与主要宽度来自冻结历史 `fff9d0f`/`d5f0b68`/`1c16d4e`；Airplane、ACC 的新候选和 Unicycle 首步诊断分别标记，不能把它们当作一套最新引擎的统一回归成绩。

### 2. 四方比较的合同与限制

- 三 GPU 主比较为共同 box/same-slope、官方输入布局、原 float32 RPC 传输数值、strict 植物模式；这是共享对照入口，不等于作者独立默认 hybrid/parity 入口。Huan/Xiangru 独立默认单摆入口另曾完成，不能用其模式替代主表。
- 原生使用 CROWN-Reach 中冻结的 Flow* 库和原 C++ 植物/NN 服务，固定完整 h 的整数步包装器。批量正式计时用 4 worker，不用早期自动 80 线程结果做性能排名。ACC 为共同且明确的仿射特征适配器。
- 正式计时：V100-SXM2-16GB，GPU3、CPU14–17、Torch 单线程；四方轮换顺序，1 轮排除预跑 + 5 轮正式运行；7 配置 × 4 方 × 6 轮 = 168 进程，其中 140 样本进入统计。共享服务器，非整机独占。
- 表中时间为新进程 wall time，含导入/缓存扩展加载、NN 初始化、求解与总结；不含首次编译，不是单 kernel 时间。宽度来自独立带观察器运行，不能把其单次秒数混入正式中位数。
- Huan/Xiangru 29 核心 Python 文件中 27 相同，CUDA 内嵌数学代码相同；13 个可比配置（含 QUAD、不含 Airplane）的范围文件完全相同。相同结果有源码和字节证据，不是两个独立算法互相证明正确。
- strict 开关和宽度接近不证明 soundness；后续已查到 ACC 原生反例、Unicycle 原生初盒 1 ULP 缺口，以及冻结三 GPU 通用倒数路径反例。通用倒数反例不等于已证某个完整 benchmark 实际漏解；原报告应保留这种范围限定。

来源：[REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/REPORT.md)；[作者倒数路径实际 CPU 反例](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/reciprocal_author_counterexamples_cpu_20260928/REPORT.md)。

### 3. 完整时域的五次正式计时

单位秒；每格为中位数 [最小, 最大]。ACC* 的旧原生结果有后续正确性反例，此行仅保留历史执行时间，不能构成正确参照下的速度/精度验收。

| Benchmark | Flow* | Huan | Xiangru | 我们 | Flow* / 我们 |
|---|---:|---:|---:|---:|---:|
| Attitude | 6.315304 [6.111067, 6.354589] | 6.989571 [6.960294, 7.058767] | 7.028113 [6.839285, 7.375633] | 7.120503 [6.978546, 7.186483] | 0.886918× |
| Unicycle | 10.721960 [10.553988, 10.778262] | 9.795658 [9.622197, 10.058480] | 9.580615 [9.544477, 10.054354] | 6.908679 [6.852378, 7.041852] | 1.551955× |
| TORA ReLU/tanh | 8.632177 [8.617337, 8.676372] | 12.421312 [12.137125, 12.502958] | 12.331677 [12.225945, 12.705246] | 7.684916 [7.659163, 7.874790] | 1.123262× |
| TORA sigmoid | 9.098132 [9.033632, 9.121180] | 13.164734 [12.969238, 13.259016] | 13.273851 [13.188094, 13.515251] | 8.207989 [8.126622, 8.310452] | 1.108448× |
| NAV robust | 68.163522 [66.916274, 78.318447] | 14.646262 [14.070551, 15.074267] | 14.813850 [14.044636, 14.915904] | 9.729652 [9.348544, 9.966918] | 7.005751× |
| Single Pendulum | 4.907801 [4.773311, 4.916538] | 5.596916 [5.527276, 5.624499] | 5.618161 [5.503686, 5.819269] | 5.176120 [5.068480, 5.374686] | 0.948162× |
| ACC* | 8.259421 [8.224243, 8.341850] | 8.034780 [7.875580, 8.078416] | 7.959671 [7.862016, 8.080713] | 7.792836 [7.740355, 7.868385] | 1.059874× |

原始数据：[timing.csv](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/timing.csv)；[168 运行样本](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/timing_samples.csv)。
可用于主讲的具体结论：NAV robust 68.163522 → 9.729652 s（7.005751×，显示可用 GPU 加速）；TORA ReLU/tanh 与 sigmoid 分别约 1.123262× / 1.108448×；Unicycle 约 1.551955×，但最终 x4 宽度为 3.499392 倍，不能只讲速度。Attitude 及单摆完整进程仍慢于 Flow*。

### 4. 未完成四方全时域：原始停止与步数

每格是接受/保存的全分区前缀、终止类别、单次带观察器 wall 秒。它们不是五次正式计时，不能计算完整时域排名。原生“保存前缀”不推断未导出的内部进度。

| Benchmark / 目标 | Flow* | Huan | Xiangru | 我们 |
|---|---|---|---|---|
| DP more robust / 80步 | 21步；性质早停 FALSIFIED；5.107280s | 22步；性质早停 Unsafe；5.630889s | 22步；性质早停 Unsafe；5.755396s | 22步；性质早停 Unsafe；5.359457s |
| DP less robust / 100步 | 25步；300s 求解超时；303.658638s | 100步；完整；8.303232s | 100步；完整；8.506425s | 100步；完整；8.761649s |
| NAV standard / 600步 | 220步；300s 求解超时；303.930472s | 600步；完整；18.117416s | 600步；完整；17.907006s | 600步；完整；14.153986s |
| CartPole / 200步 | 141步；数值拒绝；8.941050s | 156步；数值拒绝；13.765621s | 156步；数值拒绝；13.991427s | 156步；数值拒绝；12.647434s |
| Airplane / 200步 | 78步；性质早停 FALSIFIED；8.464804s | 0（未形成范围）步；积分前 RSS 守卫；35.481602s | 0（未形成范围）步；积分前 RSS 守卫；35.792755s | 0（未形成范围）步；积分前 RSS 守卫；35.673258s |
| TORA homogeneous / 200步 | 200步；完整；8.195928s | 189步；数值拒绝；8.902766s | 189步；数值拒绝；8.818860s | 200步；完整；7.876134s |

来源：[diagnostic_attempts.csv](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/diagnostic_attempts.csv)。DP more robust 的官方 NN 服务器实际加载 less-robust 模型，忠实记录；未将早停分类升级为新的正式安全/不安全证明。H/X homogeneous 2279/2400 lane-steps、全分区前缀 189；不是 200 步全成功。CartPole 已用正确 float32 NN 重跑（GPU 接受156步，原生保存141步），错误初轮不入主表。

### 5. 宽度：共同前缀与两个不同统计量

宽度 = upper − lower；endpoint 是步末、tube 是整步。观测为局部 tmvPre 的区间 hull，不能拼接旧 slides 的组合 tmv 观察量。分区宽度比先对每个坐标/视角在所有共同有效步和分区上取 p95/max，再取最差坐标/视角；union 则先合并全部分区为整体 hull。p95 不是置信区间。以下比值均 GPU/旧 native；Airplane 初轮无 GPU 范围，另列后续数据。

#### 5.1 分区内宽度比（p95 / max）

| Benchmark | 共同步 / 时间 | Huan | Xiangru | 我们 |
|---|---|---:|---:|---:|
| Attitude | 60 / 3s | 1.1195988 / 1.1455754 | 1.1195988 / 1.1455754 | 1.1207865 / 1.1456466 |
| DP more robust | 21 / 0.105s | 1.3446626 / 1.4002776 | 1.3446626 / 1.4002776 | 1.3446381 / 1.4002687 |
| DP less robust | 25 / 0.25s | 1.0945835 / 1.1130817 | 1.0945835 / 1.1130817 | 1.1060886 / 1.1243569 |
| NAV robust | 600 / 6s | 1.0112475 / 1.0387696 | 1.0112475 / 1.0387696 | 1.0298648 / 1.0465015 |
| NAV standard | 220 / 2.2s | 1.0261561 / 1.4863705 | 1.0261561 / 1.4863705 | 9.0921053 / 17.064516 |
| Unicycle | 500 / 10s | 2.8405832 / 3.4434523 | 2.8405832 / 3.4434523 | 2.8756543 / 3.4993923 |
| TORA ReLU/tanh | 500 / 5s | 1.0070272 / 1.0077032 | 1.0070272 / 1.0077032 | 1.0072094 / 1.0081448 |
| TORA sigmoid | 500 / 5s | 1.0091702 / 1.0107409 | 1.0091702 / 1.0107409 | 1.0091702 / 1.0107409 |
| CartPole | 141 / 0.705s | 1.071974 / 1.2030585 | 1.071974 / 1.2030585 | 1.071974 / 1.2030715 |
| Single Pendulum | 100 / 1s | 1.0834202 / 1.0921442 | 1.0834202 / 1.0921442 | 1.0531892 / 1.0581209 |
| TORA homogeneous | 189 / 18.9s | 2.5743687 / 7.8995985 | 2.5743687 / 7.8995985 | 1.4401228 / 1.9658069 |
| ACC* | 50 / 5s | 7.3096151 / 7.9448411 | 7.3096151 / 7.9448411 | 7.3026165 / 7.9397267 |

#### 5.2 整体 union 宽度比（p95 / max）

| Benchmark | Huan | Xiangru | 我们 |
|---|---:|---:|---:|
| Attitude | 1.1195988 / 1.1455754 | 1.1195988 / 1.1455754 | 1.1207865 / 1.1456466 |
| DP more robust | 1.3446626 / 1.4002776 | 1.3446626 / 1.4002776 | 1.3446381 / 1.4002687 |
| DP less robust | 1.0311083 / 1.0394422 | 1.0311083 / 1.0394422 | 1.0317188 / 1.0394415 |
| NAV robust | 1.0047324 / 1.0123134 | 1.0047324 / 1.0123134 | 1.0079598 / 1.012313 |
| NAV standard | 1.0038687 / 1.0097081 | 1.0038687 / 1.0097081 | 1.0048974 / 1.0095881 |
| Unicycle | 2.8405832 / 3.4434523 | 2.8405832 / 3.4434523 | 2.8756543 / 3.4993923 |
| TORA ReLU/tanh | 1.0070272 / 1.0077032 | 1.0070272 / 1.0077032 | 1.0072094 / 1.0081448 |
| TORA sigmoid | 1.0091702 / 1.0107409 | 1.0091702 / 1.0107409 | 1.0091702 / 1.0107409 |
| CartPole | 1.071974 / 1.2030585 | 1.071974 / 1.2030585 | 1.071974 / 1.2030715 |
| Single Pendulum | 1.0834202 / 1.0921442 | 1.0834202 / 1.0921442 | 1.0531892 / 1.0581209 |
| TORA homogeneous | 3.0016734 / 7.8995985 | 3.0016734 / 7.8995985 | 1.4008632 / 1.8235283 |
| ACC* | 7.3096151 / 7.9448411 | 7.3096151 / 7.9448411 | 7.3026165 / 7.9397267 |

精确 CSV：[width_summary.csv](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/width_summary.csv)。NAV standard 的我方 cell 最大 17.064516 倍对应 step59、lane626、x3：native 3.441691376e-15，我方 5.8730798e-14；整体 union 最坏仅 1.00958805。不能把该极小分母比值解说成“整体膨胀17倍”。Unicycle 的 3.499392 倍对应约 0.0352 的实质宽度，不属于此类末位现象。ACC* 旧分母漏包，因此旧 7.94 倍只描述输出、不支持精度结论。

#### 5.3 共同前缀最后一步的绝对 endpoint union 宽度

下面直接从归档 `width_trajectory_*_matched.csv.gz` 抽取，物理坐标按官方配置次序。完整案例是最终时刻；其余只是表中共同前缀最后时刻，不能冒充完整终点。数值显示10位有效数字，原 gzip CSV 保留完整二进制64文本。Huan/Xiangru 分列但值相同。

| Benchmark / 步 | Flow* | Huan | Xiangru | 我们 |
|---|---|---|---|---|
| Attitude / 60 | [0.004194134463, 0.005980153698, 0.006193976913, 0.03229213136, 0.01763594283, 0.01863740395] | [0.004174214013, 0.005935658769, 0.006172913343, 0.03207032616, 0.01736151581, 0.01847936361] | [0.004174214013, 0.005935658769, 0.006172913343, 0.03207032616, 0.01736151581, 0.01847936361] | [0.004398304598, 0.006299372384, 0.006555204266, 0.03384847706, 0.01848375419, 0.01971918617] |
| DP more robust / 21 | [3.80609534e-06, 7.522141192e-06, 4.407068939e-05, 9.565700683e-05] | [3.255760261e-08, 4.029821099e-08, 5.513193027e-07, 7.481526322e-07] | [3.255760261e-08, 4.029821099e-08, 5.513193027e-07, 7.481526322e-07] | [2.737195159e-08, 3.416114414e-08, 5.486477652e-07, 7.437120154e-07] |
| DP less robust / 25 | [0.3989302374, 0.326588365, 0.6066063165, 0.5372534131] | [0.3986209495, 0.3262640348, 0.6040297561, 0.5347722464] | [0.3986209495, 0.3262640348, 0.6040297561, 0.5347722464] | [0.3992846159, 0.327082814, 0.6071061661, 0.5376248335] |
| NAV robust / 600 | [0.09099849755, 0.01507648988, 0.02489177998, 0.2941088597] | [0.09030302286, 0.01480298716, 0.02461457813, 0.2924140666] | [0.09030302286, 0.01480298716, 0.02461457813, 0.2924140666] | [0.09134371892, 0.01511633265, 0.02495126959, 0.2947860326] |
| NAV standard / 220 | [0.4330085412, 0.1660777786, 0.3054833759, 0.6188299386] | [0.4329277545, 0.1660152493, 0.3054564813, 0.6187841293] | [0.4329277545, 0.1660152493, 0.3054564813, 0.6187841293] | [0.4329757561, 0.1660973374, 0.3054856481, 0.6187799965] |
| Unicycle / 500 | [0.0403853379, 0.06082287733, 0.04165892139, 0.03520954223] | [0.0954093678, 0.1409441537, 0.1313936842, 0.1212423802] | [0.0954093678, 0.1409441537, 0.1313936842, 0.1212423802] | [0.09669186312, 0.1428876202, 0.1337684916, 0.1232120025] |
| TORA ReLU/tanh / 500 | [0.0251280695, 0.02745753526, 0.02203482393, 0.02161695002] | [0.0250795086, 0.02741051979, 0.02199292419, 0.02159111089] | [0.0250795086, 0.02741051979, 0.02199292419, 0.02159111089] | [0.02509965709, 0.02739114704, 0.02201384029, 0.02165234232] |
| TORA sigmoid / 500 | [0.02487466578, 0.02635338883, 0.02610609014, 0.02799617606] | [0.02480363422, 0.02628227864, 0.02602510721, 0.02794208213] | [0.02480363422, 0.02628227864, 0.02602510721, 0.02794208213] | [0.02489772349, 0.0262875855, 0.02608527732, 0.02804372738] |
| CartPole / 141 | [0.3692052792, 1.830964122, 2.116055089, 15.83408371] | [0.3684962987, 1.821561803, 2.064329903, 14.39007037] | [0.3684962987, 1.821561803, 2.064329903, 14.39007037] | [0.3684543433, 1.821221842, 2.063535278, 14.38119426] |
| Single Pendulum / 100 | [0.1391393524, 0.1444355487] | [0.1482616762, 0.1551736948] | [0.1482616762, 0.1551736948] | [0.1438742238, 0.1502647624] |
| TORA homogeneous / 189 | [0.1416534512, 0.1358758305, 0.5933792014, 0.5906356258] | [0.2870019596, 0.3714916143, 3.124087479, 4.148934951] | [0.2870019596, 0.3714916143, 3.124087479, 4.148934951] | [0.1674881631, 0.1657311502, 0.9090997401, 1.000535148] |
| ACC* / 50 | [20.99393312, 0.1975723307, 0.0006846514465, 5.028368774, 1.79563214, 1.063699852] | [21.01204096, 0.2017538906, 0.0007410763492, 5.497975787, 2.01551073, 1.181106183] | [21.01204096, 0.2017538906, 0.0007410763492, 5.497975787, 2.01551073, 1.181106183] | [21.01045505, 0.2016286635, 0.0007376997796, 5.333353383, 1.948655935, 1.159430455] |

### 6. ACC 后续纠错：需要替换旧解读

旧 native 的合法初始点 v(0)=32、a(0)=0 在 t=.1/.2 有独立解析包络，而保存端点与它们不相交。直接 ODE.reach / 匹配包装器、2维/8维最小例复现，问题定位于 refinement 的 VAR 截断尾项缓存/重放；不能用“Flow* 更窄”当真值。只在隔离库补回尾项，原作者库和原结果保留。修复库 `73a23102…` 完整 ACC 50步、早期6个解析见证通过，尚未通用回归。

| 方法 | step2 a_lead endpoint 绝对宽度 | 计时口径/秒 |
|---|---:|---|
| 旧 native（漏包） | 0.00028282614978181186 | 旧正式五次见上表；不作正确参照 |
| VAR-tail 修复 native 实验版 | 0.002247354391217149 | 8.296836939，新进程单次诊断 |
| Huan strict 历史 | 0.0022470088317284587 | 5.659108131，历史 driver |
| Xiangru strict 历史 | 0.0022470088317284587 | 5.573999452，历史 driver |
| 我方本轮控制组 | 0.002245562328254458 | 5.157350691 driver；8.3066进程 |
| 我方 solution_order 候选 | 0.0007721378347164132 | 5.448104681 driver；8.4740进程 |

候选保持解阶3、h=.1、T5/B1/SR50，仅将 RHS 验证阶数升至解阶3，无重试；关键 endpoint 宽度比控制组减约65.6%，driver单次慢约5.6%，不是加速结论。与修复 native 的50×6 endpoint比，H/X最大约1.0000000000035；tube最大仍约1.42045，需另查。候选和修复参照都未成为覆盖全基准的默认验收版本。
原始来源：[REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/REPORT.md)；[ACC_REPAIRED_ANALYSIS.json](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_REPAIRED_ANALYSIS.json)；[ACC_ANALYSIS_V2.json](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_ANALYSIS_V2.json)；[ACC_REPAIRED_WIDTHS.csv](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_REPAIRED_WIDTHS.csv)。

修复参照下的逐项最大宽度比（覆盖50步×6维，endpoint与tube分开；小于1仍不是正确性证明）：

| 方法 | endpoint最大比 | tube最大比 |
|---|---:|---:|
| 修复native | 1 | 1 |
| Huan | 1.0000000000034808 | 1.4204541210857169 |
| Xiangru | 1.0000000000034808 | 1.4204541210857169 |
| 我方控制组 | 0.99999382236969947 | 1.4204438330975624 |
| 我方solution_order候选 | 0.99999356475164225 | 1.4099253717240527 |

### 7. Airplane 后续：已越过内存阻塞，未完成200步

原任务 B1、19总变量/12物理维、P6/h=.01、T2/200步不变。将稠密元数据改为按实际单项式支持集构建，并接入稀疏范围与带舍入的endpoint转换；显式候选 b049d443，不是偷偷替换H/X原入口。模型、初盒、NN/RPC32合同保持。原H/X仍无积分范围。

| 阶段 | 实际接受步 | 原因 | 完整进程秒（单次，非正式排名） |
|---|---:|---|---:|
| 原四方 suite Huan / Xiangru / ours | 0 / 0 / 0 | 元数据建表RSS守卫 | 35.481602 / 35.792755 / 35.673258 |
| 我方稀疏元数据 NNCS | 79 | 整步 hull Unsafe 早停 | 46.628120 |
| 仅额外采集 GPU 见证 | 79 | 同上；79步范围与前次完全同字节 | 45.689562 |
| 隔离时间二分检查 | 78 | 后半步违反约束，Unsafe 早停 | 45.312770 |
| 原生 Flow* 历史 | 78 | 递归时间划分 FALSIFIED 早停 | 8.464804 |

已取回并检查原生和实际 GPU 的 step78 y−1 多项式：整步跨0，后半步下界严格为正；GPU独立Fraction后半步界约[.004196709424782507,.009004833661608235]。有界两分检查在本例让停止步一致；不声称与原生递归检查在任意系统等价。提前停止没有改善前78步宽度，范围与原候选前缀完全同字节。

共同78步：tube最坏比1.0272159799109835；step78 y tube native0.009514599684767755、ours0.009582362199862193。endpoint含极小分母，最坏比223，不可脱离绝对值：step78 y endpoint native4.1096015479524795e-12、ours8.713696431073004e-12。所有12维完整表见来源。此病例初始点箱使端点接近浮点误差尺度，跨实现中心亦不同；不能由tube接近宣称完整NNCS已证明。

首次稀疏NNCS peak RSS1.872482GiB / GPU2.480469GiB；最终二分诊断GPU采样峰2,663,383,040B，无资源中止。通用runner/watch因没到200步记incomplete/error，但实际是接受后的性质早停，不是数值拒绝。仍没有完整200步、H/X恢复、四方正式五次时间。
来源：[REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_nncs_recovery_20260923/REPORT.md) → [REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_stop_witness_20260924/REPORT.md) → [REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_gpu_stop_20260924/REPORT.md)；[WIDTHS_COMMON_PREFIX.csv](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_nncs_recovery_20260923/evidence/WIDTHS_COMMON_PREFIX.csv)。

### 8. Unicycle 后续：不要把9/29首步诊断写成新500步成功

9/23 suite 四方确已完整500步及五次计时。9/29新工作是归因，不是重跑新QUAD引擎的500步。初盒/扰动/变量置换核对正确；native真实50次输入的独立NN replay在同输入时f64与RPC32均相等，但不能据此说闭环两边每周期输入盒相同。

- 首步 native→三GPU 已出现共同宽差。x3/x4 tube可在同一个GPU多项式上解释为逐项区间化丢失共享时间相关性；额外下界宽度分别0.0010869132244709406和0.00013452625651671978。这不是删误差项，也不是整条轨迹原因已找到。
- 我方相对H/X首3步仅末位差，以1e-12绝对阈值观察到step4 x1额外宽1.0968519248422126e-6；尚在第一次控制期。没有step3/4内部证据，不能把cutoff猜测当结论。
- 原生CPU首步已在原库/原RPC响应asFloat模式复现，16bounds/8width与旧首步逐位一致；保存initial、before/after injection、after advance、after endpoint五个实际TM/SR边界。一步0.114133202秒不含NN重算，不是全程计时。
- 实际旧native初始化x2下端点少覆盖1ULP（8.881784197001252e-16）。本次保留该问题；不能拿同RPC域重放替代初始集合严格覆盖，也不能直接把这一末位缺口当作最终3.5倍宽差解释。
- 三GPU对齐首步计划只启动ours一次，在植物计算前库加载失败；Huan/Xiangru没有启动。watch3.132135秒是失败启动时间，不是单步数值性能。缺Ninja为CPU实证前置阻断；原三SO直接导入成功、SHA不变。没有修加载/新v2/数值重跑。

来源：[UNICYCLE_FIRST_GAP_REVIEW_20260929.md](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/UNICYCLE_FIRST_GAP_REVIEW_20260929.md)；[REVIEW.md](/Users/shengenli/Documents/ChatGPT/verification/results/unicycle_first_gap_20260929/native_v1_local/REVIEW.md)；[LOADER_FAILURE_AND_MINIMAL_PROPOSAL.md](/Users/shengenli/Documents/ChatGPT/verification/results/unicycle_first_gap_20260929/gpu_preparation_v1/LOADER_FAILURE_AND_MINIMAL_PROPOSAL.md)。

### 9. 剩余其他基准的真实进展

NAV robust、异构TORA、Attitude、单摆已有上面冻结suite结果；本次本地资料未找到它们在新QUAD候选上的全程复验。NAV standard与DP less robust三GPU完成，但原生4worker仍在300s求解时限停止，不能将其截断时间当完整Flow*时间。DP more robust仍为早停、模型命名/实际加载差异保留。CartPole仅完成原配置/NN dtype纠正后的失败对照，尚未解释所有拒绝差异。TORA homogeneous我方与native完整而H/X拒绝，仍无完整四方速度结论。VCAS只有NN服务/随机模拟，没有同类C++可达性入口，未冒充第15个实验。


### 10. 各任务精确配置索引

共同 cutoff=1e-6；SR默认1000、ACC50。余项估计按各配置逐项保留，不统一写成±.1。下面是最终执行配置，初盒、分区、模型/盒SHA见原CSV。

| Benchmark | B | 物理维 | h / 解阶 | 控制周期 × 次数 | T / ODE步 | 余项估计 |
|---|---:|---:|---|---|---|---|
| Attitude | 1 | 6 | 0.05 / 3 | 0.1 × 30 | 3.0 / 60 | [-0.01, 0.01] |
| DP more robust | 1 | 4 | 0.005 / 4 | 0.02 × 20 | 0.4 / 80 | [-0.01, 0.01] |
| DP less robust | 225 | 4 | 0.01 / 4 | 0.05 × 20 | 1.0 / 100 | [-0.01, 0.01] |
| NAV robust | 25 | 4 | 0.01 / 4 | 0.2 × 30 | 6.0 / 600 | [-0.1, 0.1] |
| NAV standard | 640 | 4 | 0.01 / 4 | 0.2 × 30 | 6.0 / 600 | [-0.1, 0.1] |
| Unicycle | 1 | 4 | 0.02 / 2 | 0.2 × 50 | 10.0 / 500 | [-0.01, 0.01] |
| TORA ReLU/tanh | 1 | 4 | 0.01 / 6 | 0.5 × 10 | 5.0 / 500 | [-0.01, 0.01] |
| TORA sigmoid | 1 | 4 | 0.01 / 6 | 0.5 × 10 | 5.0 / 500 | [-0.01, 0.01] |
| CartPole | 1 | 4 | 0.005 / 6 | 0.02 × 50 | 1.0 / 200 | [-0.1, 0.1] |
| Airplane | 1 | 12 | 0.01 / 6 | 0.1 × 20 | 2.0 / 200 | [-0.01, 0.01] |
| Single Pendulum | 1 | 2 | 0.01 / 2 | 0.05 × 20 | 1.0 / 100 | [-0.01, 0.01] |
| TORA homogeneous | 12 | 4 | 0.1 / 3 | 1.0 × 20 | 20.0 / 200 | [-0.01, 0.01] |
| ACC* | 1 | 6 | 0.1 / 3 | 0.1 × 50 | 5.0 / 50 | [-0.1, 0.1] |

[configurations.csv](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/configurations.csv) 同时给出各初盒、模型SHA、boxesSHA和实际YAML路径。Unicycle额外w∈[-1e-4,1e-4]。ACC候选仍是解阶3，只改变验证阶策略，不与其它配置的解阶混为一谈。

### 11. 可直接用于slides的主线与数据边界

1. 从少数案例扩展到官方14配置的四方筛查，7项有正式五次时间；最亮眼是NAV robust约7×且整体范围接近。
2. 异构TORA时间略快、范围接近；Unicycle速度更快但范围宽，Attitude/单摆启动开销下仍不占优。
3. 发现“原生更窄不一定更正确”：ACC原生尾项缺失，修复后关键宽度与H/X接近；这是一项正确性纠错，不包装成GPU更好看的结果。
4. Airplane由积分前内存失败变成真实NNCS接受并可解释早停，未完成全程和四方性能。
5. Unicycle完成历史复核、首步原生状态采集及环境定位，三GPU内部归因未完成；当前新QUAD优化尚未全套回归。

以下源SHA仅绑定本次读取的本地文件，不声称重新下载/审计服务器全部大范围。终点表重读12份现存CSV并检查三GPU重复native列一致、H/X终点一致。原始二进制范围多数保留远端；可复算的CSV/JSON与manifest本地已有。

| 本地来源 | SHA256 |
|---|---|
| [oldslides](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/slides.tex) | `f96380982cd462e1f48121f57b431a00a1715b7caede750767f17254e31f917e` |
| [main](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/REPORT.md) | `ea322d459fa53284fdf4992fe393d36544dc08c909f04ad5b7a4bddb2599887f` |
| [timing](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/timing.csv) | `0075fa5557e72bc9633aa2e4ba3950b0764bcdca38253e1b4220b883c26adfcf` |
| [samples](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/timing_samples.csv) | `a37013ba290f5601f6a691a5305e349020a3382219a195630d24816471accc51` |
| [width](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/width_summary.csv) | `63b08b7f6c16e76b26a12588d51db60ee97a48d6f5ca34494c09e71e8004ba76` |
| [diag](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/diagnostic_attempts.csv) | `35cecb669a47b2daf5990098c23810919b2178076b7e1b3bbb955fa3f90ba19b` |
| [config](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/configurations.csv) | `7abe1a1e1545dccc717b65825161921f9ae73956c87d55417df73f3d16f98220` |
| [accfix](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_REPAIRED_ANALYSIS.json) | `f3cdfd7fc856c12a2e203581feea13a60596750840463feb691aeebfa316a057` |
| [accnew](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/ACC_ANALYSIS_V2.json) | `aa2e12ff5ec27029c9db28e078556bf8d80906e8dc013a3442521c7ef6f52a27` |
| [airwidth](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_nncs_recovery_20260923/evidence/WIDTHS_COMMON_PREFIX.csv) | `dc29535672696c3b02516e56ebf0ace1fb4dd9c0031f3641302546e6dde5b1d9` |
| [airgpu](/Users/shengenli/Documents/ChatGPT/verification/results/airplane_gpu_stop_20260924/REPORT.md) | `9d5504a410e65baa28e623511ddeac0d2a98f1a24fc3d1598f3a7cb549161349` |
| [unigap](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/UNICYCLE_FIRST_GAP_REVIEW_20260929.md) | `8045690da759d37ecb11fa6e75fa6b6b4fe6fe87399198bf323393dbf669a680` |
| [unicpu](/Users/shengenli/Documents/ChatGPT/verification/results/unicycle_first_gap_20260929/native_v1_local/REVIEW.md) | `5475d55afb6bdbc8f70339f3de75df273a8e82aeb88f2021f2f67c92c91796a3` |
| [unienv](/Users/shengenli/Documents/ChatGPT/verification/results/unicycle_first_gap_20260929/gpu_preparation_v1/LOADER_FAILURE_AND_MINIMAL_PROPOSAL.md) | `8c1648b062dd84b9b741d1f3c7cd45448a43806564bd64b95596dfcec09390d7` |

完整轨迹CSV（endpoint/tube、每步整体上下界/宽度，保留原全精度）：

- [width_trajectory_attitude_control_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_attitude_control_matched.csv.gz)；SHA `edc8dd2732ddea1242a6721076326104683096dbce85629d60209b80a8753841`。
- [width_trajectory_double_pendulum_more_robust_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_double_pendulum_more_robust_matched.csv.gz)；SHA `36b104f4a940d5f06b44d35ba8e25253499bb1602c41f61ecaa034cfd7873cac`。
- [width_trajectory_double_pendulum_less_robust_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_double_pendulum_less_robust_matched.csv.gz)；SHA `cac4bd3a63d09619f16f4b6e19d3fdbc9f9744e99f75cc834bca6de82e96531f`。
- [width_trajectory_nav_robust_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_nav_robust_matched.csv.gz)；SHA `bc2316cd3817cd4575c5417a6131bdffad8c533068cf53a2e7faf307b3929808`。
- [width_trajectory_nav_standard_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_nav_standard_matched.csv.gz)；SHA `1f950a2c1cd7d7cfddfae4fc2cf080499aa0a1f0861fee81b927305fa26d4cdc`。
- [width_trajectory_unicycle_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_unicycle_matched.csv.gz)；SHA `faf582bcbd28286aeee8ef413268b7de4f28f21564a71618df23bbc2cb864751`。
- [width_trajectory_tora_relu_tanh_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_tora_relu_tanh_matched.csv.gz)；SHA `67b0087b23f659bdd75d7c5c8aaca96220828b81c2e8226c63e154905165781f`。
- [width_trajectory_tora_sigmoid_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_tora_sigmoid_matched.csv.gz)；SHA `1db4abcdd6e7ac5c87709cf5331976e86341618f7e7d772fafd9691923c59b2b`。
- [width_trajectory_cartpole_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_cartpole_matched.csv.gz)；SHA `b572693f1981bedb6b5a47fa73d4e3b2399787f5a55d7c54596eecf353a3dd3f`。
- [width_trajectory_single_pendulum_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_single_pendulum_matched.csv.gz)；SHA `7407b0cc59e878a926cfae85b40cd42ff8580da56d3c329efdb87fb4deea2dc0`。
- [width_trajectory_tora_homogeneous_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_tora_homogeneous_matched.csv.gz)；SHA `3643f8e0bdb8c82d00662d16570f9e9ca37446614e2f288c701c53bc716e64d7`。
- [width_trajectory_acc_matched.csv.gz](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v2/width_trajectory_acc_matched.csv.gz)；SHA `d520c3f0e3a9c900375ef2cd594dc8988a228f9588b10658d8a3f1b4ce079726`。


---

## 附录C：仓库复现身份、比较合同与正确性边界

以下正文复制自 [scope_and_contracts.md](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/scope_and_contracts.md)，SHA256 `82fae398fb2eb6816d583c85ec9a70c9cceb5fb0e66d21092a26de2010944226`。未改写实验数值。

## 上次 slides 之后的范围、复现身份与结论边界

本文件供新 TeX 汇报取材。2026-09-29 仅检查本地既有材料；没有 SSH、求解器、GPU 或新实验操作。实验保持暂停。这里负责报告范围与科学口径，完整数值表由本次汇报的独立数据整理提供。

### 1. “自从上次 slides 后”的准确起点

上次交付为 [README](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/README.md)、[slides.tex](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/slides.tex)、[REPORT.md](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923/REPORT.md)。共 29 页 slides（23 页主讲、6 页附录）和 17 页报告。发布分支 `codex/progress-report-20260923`，commit `3edae8ce2e5a2b3e7915c645742ececcabc6b8ef`，见 [发布收据](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923_publication.json)。

本次直接从 ZIP 读取并与目录文件比较，`slides.tex`、`README.md`、`REPORT.md` 全部逐字节相同，避免依据后来改写的报告推断旧演示内容。

| 已核对象 | SHA256 |
|---|---|
| `flowstar_progress_review_20260923_latex_bundle.zip` | `edfe5e9bc497d74fd5e3b2fcc76fa5cb4e507a3f992f9cd08397fae5d2702968` |
| 旧 `slides.tex` | `f96380982cd462e1f48121f57b431a00a1715b7caede750767f17254e31f917e` |
| 旧 `REPORT.md` | `8186844672192c8f0bdf00c3f9060bdaa4d4d2830d3c248e61884f3e121a3c0d` |

旧 slides 已经讲过：完整 GPU 推进链；VDP/Brusselator 的 B1 长轨迹与 B32 真实分区四项比较；TORA 线性路径 v2 从部分失败到 2400/2400；单摆和 CartPole；输出后端五次配对改善 15.3%–43.6%；独立 GPU 乘法候选；12 plant / 14 NNCS 的早期筛查；控制器舍入资格未完成。

**旧 slides 没有汇报后续 ARCH-COMP 14 配置四方矩阵、Huan QUAD 75 秒复现、ACC 参照更正、QUAD 连续修复到完整 1024×1000、三角函数复用的约 2 倍完整运行改善。** 这些应是新汇报主线。旧材料与新增 ARCH-COMP 矩阵都标 2026-09-23，因此范围边界按已交付内容确定，不能简单排除同日后续工作。

### 2. 四个仓库及实际被测资产

旧 slides 标“三个仓库、四类实现”；后续 ARCH-COMP 对照明确增加官方 CROWN-Reach 源仓库。四个来源仓库应这样介绍：

| 来源 / 角色 | 固定资产与分支证据 | 入口和复现范围 |
|---|---|---|
| 用户工程 [torch_tm_flowpipe](https://github.com/lsnnnnnnnn/torch_tm_flowpipe) | 工程、适配、实验与报告；旧报告已发布于上述 `codex/progress-report-20260923` | 不把报告分支 commit 当作最新数值引擎 commit |
| Huan [flowstar-gpu](https://github.com/huanzhang12/flowstar-gpu) | `d5f0b68fcd36ba5f582733624f074728fe9720d8`；当时 Huan `main` | 独立 NNCS 入口 `integrations/crown_reach/gpu_driver.py`；后续 SR 分块是隔离变体，不冒称原源未改 |
| Xiangru [CROWN-Reach_Development](https://github.com/xiangruzh/CROWN-Reach_Development) | 来源分支 `2026_experiment`；实际 detached `1c16d4ef2cb91cc94b1c784f7383e1eba135d8d3` | 独立入口 `src/flowstar_gpu/integrations/crown_reach.py`；也作为三方对照的共享驱动 |
| 官方 [CROWN-Reach](https://github.com/Verified-Intelligence/CROWN-Reach) | `7b90f30831212f0c59c44c67aa428932d5307a81`；本地历史 inventory 未保存分支名，不补猜 | `archcomp/*` C++ 程序、同目录 CROWN 服务、模型与原生 Flow* 对照；原入口和固定步数/初始化修复入口分开 |

精确来源：[remote_inventory.json](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/remote_inventory.json)、[ARCH-COMP 报告](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/report/REPORT.md)、[Xiangru clone 记录](/Users/shengenli/Documents/ChatGPT/verification/planning_sources/conversation_page_2.md:721)、[Huan QUAD 复现报告](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_failure_20260923/REPORT.md:14)。这些是历史固定版本；本轮没有联网核查作者最新 HEAD。

**“我们”是一条派生实现线，不是完全独立发明的第四种数学算法。** 新增 ARCH-COMP 主矩阵中被测我方引擎为 `fff9d0f5742ebfa4e0652bbbf79b31bdd59ab3e3`，服务器工作树 `engine_linear_leaf_v2`；实际 inventory 的 remote 仍为 Huan `flowstar-gpu`。不能把这个 commit 说成已推送到用户仓库或作者仓库。该次工作树分支名未在所读 inventory 中保存，以完整 commit、源码 SHA、实际库身份为准。最新 QUAD 又由冻结数值引擎与独立适配器组合，不能仅用旧 fff9 标签代表其全部代码。

#### 独立入口确实跑过

原始 Huan 和 Xiangru 入口各自执行默认 Single Pendulum，均 `completed/rc0`，B1、order2、20 控制周期×5 子步=100 步；实际 `coupling=hybrid`。两份 `ctrl_steps` 的 20 条记录本轮重新以 JSON 相等比较通过，打印的四条最终 HULL 也完全相同。

| 原始入口 | 本次已读历史进程 wall | 原始 stdout 的内部 time cost |
|---|---:|---:|
| Huan `gpu_driver.py` | 8.7958964551799 s | 4.058186 s |
| Xiangru `crown_reach.py` | 8.402115842327476 s | 3.245168 s |

共同物理 HULL：x1 `[0.566926486592, 0.704484162534]`；x2 `[-0.557712860545, -0.418458767625]`。这两次只证明该默认入口可运行，不是五次性能排名，也不证明所有 benchmark、分支或模式已独立复现。

原始 argv、返回值和 metrics 均保存在 [Huan 独立入口](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/huan_standalone_sp_default/process.json) 与 [Xiangru 独立入口](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/evidence_v1/xiangru_standalone_sp_default/process.json) 同目录。三方主矩阵另由共享 Xiangru 驱动分别加载各自被冻结引擎；这是隔离 plant 实现差异的共同合同，不应称三套独立完整 NNCS 产品。

### 3. Huan 与 Xiangru 为什么相同

实际 [SOURCE_AUDIT.json](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/SOURCE_AUDIT.json) 校验 58 份冻结源文件：29 个核心 Python 文件中 27 个逐字节相同，内嵌 CUDA 数学源码相同。两个差异文件为：

- `cuda_kernels.py`：Xiangru 增加宿主编译器环境选择与检查；数学 CUDA 源相同。
- `sparse_exec.py`：增加 NVTX 诊断及可选 `FLOWSTAR_COMPOSE_PARENT_ASSEMBLY=batched`；默认 `loop` 路径保持对应算术。不能把默认路径结论扩大到可选 batched 模式。

原始 diff 在 [sparse_exec.py.diff](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/sparse_exec.py.diff) 和 [cuda_kernels.py.diff](/Users/shengenli/Documents/ChatGPT/verification/results/archcomp_review_20260923/cuda_kernels.py.diff)。新增主矩阵 13 个可比配置的完整 H/X 范围文件逐字节相同；因此整体 union hull 相同有源码与实际轨迹双重依据，并非只因表格四舍五入。

可用于 slides 的一句话：**这两个被冻结版本在本次默认核心路径高度同源，同输入产生相同范围是预期结果；它不是两种独立算法相互证明正确。** 也不能把 Huan 后来的 parity QUAD 结果自动记成 Xiangru 已重跑的结果。

### 4. 必须分开的数值与控制器合同

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

### 5. 正确性审查的新证据与必须更正的旧解读

“没有通过缩任务、删失败或关误差获得优势”有配置、版本、接受掩码与日志证据；但不能因此承诺三个引擎无错误。新增审查实际找到问题，必须同时汇报。

#### A. VAR 截断尾项与 ACC 原生参照

原生/parity refinement replay 遗漏 VAR 截断尾项的精确局部反例：返回 RHS 上界 0.275625，而合法测试值 0.292275390625 在外；三方 strict 路径包含该见证。Huan GOTCHAS #16 已记载此语义差别，本轮为独立复核，不宣称首次发现。

实际原生 ACC 的 t=.1/.2 端点排除合法解析轨迹；直接原库最小例复现，隔离补回 VAR-tail 后原生完整 50 步。关键 step2 a_lead 宽度修复后 0.00224735439，与 H/X strict 0.00224700883 接近。因此旧四方表约 7.94 倍应保留为**旧输出比值**，不能继续作为“GPU 精度差约八倍”的结论。源码、实际执行和修复依据见 [ACC/VAR-tail 报告](/Users/shengenli/Documents/ChatGPT/verification/results/quad_residual_memory_20260923/REPORT.md)。不据此声称所有其他原生输出均已证明错误或修好。

#### B. 通用倒数公式的实际反例

分别导入两份未修改的作者 CPU sparse 归档，`1/(2+z/4)`、z∈[-1,1]、k=1–4、零输入余项、cutoff0，实际 `bad=false`，但 z=-1 的精确 4/7 超出返回完整范围。k1 缺口约 0.00309766763848，k4 仍约 4.3889656593e-7。H/X 四组全部保存字段逐字节相同；我方冻结通用倒数亦有相同问题。见 [作者实际 CPU 反例](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/reciprocal_author_counterexamples_cpu_20260928/REPORT.md)。

本次 native matched 构建对应源码含同类额外 1/c 缩放；此项是源码定位，不冒称运行了 native 二进制反例。c=2 的通用见证也没有证明某个已接受 QUAD cosine 分母步实际漏解。新的几何尾项适配器基于有限几何恒等式，须按已检查调用域及路径解释，不覆盖未替换的 dense/replay。见 [源码与精确反例](/Users/shengenli/Documents/ChatGPT/verification/results/quad_native_matched_20260928/RECIPROCAL_TAIL_SOURCE_REVIEW.md) 和 [几何尾项合同](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/RECIPROCAL_GEOMETRIC_ADAPTER_REVIEW_20260928.md)。

#### C. 初始盒覆盖和控制边界

原生 full1024 构造器采用已舍入中心 c，却只以 RU(u-c) 定半径；384/16,384 坐标、338 lanes 的实际初始 affine 欠包。首例原下界 -0.30000000000000004 被收成 -0.3，缺口 2^-54。这虽然只有 1 ULP，仍不能忽略或靠放宽 observer 容忍度掩盖。独立新初始化保留 c，取 max(RU(c-l),RU(u-c))；实际原/新 MPFR 系数与 Fraction 审查确认只 384 半径改变、全部 16,384 行覆盖。见 [原生初盒独审](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929/REVIEW.md)。当前 GPU14 原完整运行的 actual-coefficient 初始覆盖 16,384 项已通过，[GPU 独审](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/native_initial_coverage_review_20260929/GPU_INITIAL_REVIEW.md) 明确同 run 身份。新原生整程结果应标初始化修复变体，不能称未改原版。

早期 plain endpoint 的时间代入采用 RN、未收费；后来的 strict endpoint 才对当前 FULL 支持、时间幂/乘加误差及 pre_rem 交接收费，并保留原完整 SR。旧晚升阶轨迹不能被后来的修复追溯认证。还要单独说明控制器 CROWN 浮点界与 RN 仿射注入仍未取得完整端到端资格。见 [P2/P3 边界独审](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/P2_P3_STRICT_BOUNDARY_REVIEW.md)。

### 6. 阶数、任务完整性与最新优化证据

原 ARCH-COMP QUAD 固定阶为 2，而非“Flow* 始终保留很多阶”。通用 YAML 的 order4 同时带不一致的控制/ODE 步长，不是本次官方 C++ 合同。引擎可以指定阶数；最新 P3 路线明确增加工作多项式总次数，必须标为算法变体。

| 路线 | 工作解总次数 | point RHS code | 验证阶数 |
|---|---:|---:|---:|
| 后续 P2 对照 | 2 | 1 | 3 |
| 后续 P3 | 3 | 2 | 4 |

总次数包括局部时间 τ 和归一化空间变量。例如 τ·z_i·z_j 为三次，P2 无法保留，P3 可以。它与只提高验证展开阶数不同；“P3”也不等同于三次控制器模型。理论完整基 P2=171、P3=1,140、P4=5,985（17 多项式变量），不能当实测活跃项数或显存。来源：[阶数审查](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/ORDER_CONFIGURATION_REVIEW.md)。两叶/32-root 的局部诊断不冒充全部 1024 根盒。

最新三角函数复用并没有再改 P3 数值算法：限定一次 `exec_valid_s` 内相同直接变量、运算和绑定，首次完整执行，重复项复制完整系数/余项/缓存。必须将此实现优化与此前 P2→P3、控制交接、SR 调整等算法改进分开。

最终实际 CPU 终态审计状态为 `full1000_actual_outputs_equal`、`issues=[]`、`partial_logs={}`，核对全部 1000 observer、50 controller/transfer、两最终 plant 和 14 个 SR/host 组件，与原 GPU14 P3 基线逐位相同；数值轨迹/基线字节资格 true。该审计仍明确 `fullbatch_qualification=false`、`end_to_end_strict_certificate=false`、`goal_complete=false`，不能偷换成总体完成。见 [最终原始审计](/Users/shengenli/Documents/ChatGPT/verification/results/quad_targeted_recovery_20260927/evidence_trig_terminal_20260929/TRIG_GPU14_TERMINAL_REVIEW_20260929/RESULT.json)，SHA `517bd183820d43af794d92b64434d84e05e9fb4fa120927f2e9bf3c41968add2`。

实际完整同配置 watch 时间 3153.449456484→1533.752051520 秒，约 2.056 倍；这是**同基线单次完整资格运行的改善**，不是五次正式计时，也不是对 Huan 75 秒 parity 的同保证胜出。当前完成 1024×1000、50 NN、无数值拒绝；“accepted”不是完整系统安全结论。底层 40 步 phase/CUPTI 用于定位小 kernel 重复，带有测量开销，不能替代这两个完整 wall 数字。

### 7. 新汇报需要固定写出的比较边界

1. **范围定义一致才拼表。** 旧 slides 组合 tmv 范围与新增矩阵局部 tmvPre hull 不同；endpoint 与整步 tube 分开，分区均值与全分区 union hull 分开。NAV standard 的最大单分区比 17.0645 对应原生宽度 3.44169e-15，不能说整体宽了 17 倍；其 union 最大约 1.0096。零参考宽度另列，不加 epsilon。
2. **只比较共同已接受时域。** attempted steps、末周期部分接受、性质早停、数值拒绝、资源守卫和超时是不同状态。不能用失败后旧状态补齐曲线，也不能用前缀时间作完整速度分母。
3. **控制器必须同合同。** 固定实际模型、排列、shape/batch、relaxation、传输精度。旧 QUAD 只做 6 批 endpoint proxy 布局检查不能说成完整真实 RPC 重放；缩成 32 batch 曾有实际 float32 一 ULP 差，必须保留失败证据。
4. **正式计时与诊断分开。** 新增七项四方五次矩阵有 140 计入样本；QUAD 后续多数是带 observer、审计/保存的单次资格运行。内部 elapsed、driver、process/watch、独立导出各有边界，不混取最小者，也不把来自不同运行的中位数相加。
5. **原版/修复版和资源变体分列。** 主存/显存守卫变化会改变能否跑完；提高预算本身不是算术修复。最近 GPU guard14 GiB、allocator13.5 GiB、RSS11.5 GiB；旧 11.5 GiB 守卫失败仍保留。原生提高 RSS 的行也需注明；不能因更高预算而说原算法被改快。
6. **有限见证和字节相同是有范围的证据。** 源码 SHA/import 路径检查支持忠实复现；解析 Fraction 见证查实际包含；CPU 参数检查不是 CUDA 数值资格；旧新字节相同证明受检行为不变，不能证明共享错误不存在。
7. **安全状态不升级。** VERIFIED/Unsafe/FALSIFIED 是该程序在其合同下的输出；控制器/端到端证明未完成。Airplane 原先 78/79 的差别后来由半步性质检查解释，不应记成 200 步完整失败或成功。
8. **文档也保留未知。** native 历史分支名未录、作者当年 dirty source 不可重建、未完成轨迹的全时长宽度等明确留空。局部源反例不推广到所有作者分支；当前文件名中的历史日期不能替代实际运行阶段。

推荐主叙事：先复现并分清模式 → 用更多 benchmark 找出失败/宽度差 → 发现并更正参照中的局部问题 → QUAD 从失败推进到完整接受 → 保持完整输出不变将自身运行约减半 → 完整四方同保证比较及端到端控制器资格仍未闭合。


---

## 附录D：最新完整 P3 运行身份与冻结命令

下面是历史已成功运行的来源与命令，**用于审查/复现定位，不表示应立即再次执行**。其中静态的 `launched` 等字段是准备时值，真实执行状态以终态 RESULT 为准。全部依赖指纹仍需读取完整 INPUT。

### 实际 INPUT 身份摘要

```json
{
  "script_sha256": "fa3b227110e53f56d53d4c1ab4f8422f48cccd1f667722d332d586b5d24a6287",
  "common_script_sha256": "1c85e04988028f7ac95d945eeff9ef3e0936ee306c1f0a80650bb06d8073d985",
  "config_path": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp_failure_20260923/quad_author_resolved.yaml",
  "config_sha256": "940cb0b6188c3b127eee28df5d34fbda6d072f9e8b09cee8f251bd6b098f9144",
  "model_path": "/srv/local/shengenli/CROWN-Reach-GPU/ARCH-COMP2024/benchmarks/QUAD/quad_controller_3_64_torch.onnx",
  "model_sha256": "fabd84e411f4b0ebe0d6b996be6e4bd2adc48cf1118ab99c82180563b95532dd",
  "driver_path": "/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src/flowstar_gpu/integrations/crown_reach.py",
  "driver_sha256": "1bc3aeea0e9216fe8432ae78f18c9592d19658c23cdde02b90a54d0b14c08a31",
  "boxes_path": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp_failure_20260923/huan_box_parity_preload/boxes.json",
  "boxes_sha256": "8dd18155d7d63b311fece5732251758a4984c0abb428377c135e68c244cedd30",
  "reciprocal_policy": "finite_geometric_reciprocal_tail_all_five_routes_v1",
  "engine_root": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/engine_quad_normalization_center",
  "engine_head": "a3fb2e94ba976aaf498c4a9cb3f98165cddcc272",
  "driver_argv": [
    "/srv/local/shengenli/xiangru_adoption_20260907T032448Z/xiangru_upstream/src/flowstar_gpu/integrations/crown_reach.py",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp_failure_20260923/quad_author_resolved.yaml",
    "--device",
    "cuda:0",
    "--strict",
    "--engine",
    "sparse",
    "--crown-domain",
    "box",
    "--crown-relax",
    "same-slope",
    "--crown-transport",
    "native-f64",
    "--crown-input-layout",
    "native",
    "--nn-mode",
    "crown",
    "--order",
    "3"
  ]
}
```

### 原 COMMAND 全文

```json
{
  "environment": {
    "FLOWSTAR_WEIGHTED_VALIDATION": "0",
    "FLOWSTAR_RECENTER_VALIDATION": "1",
    "FLOWSTAR_SELF_MAP_RETRIES": "8",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "FLOWSTAR_GLUE": "graph",
    "FLOWSTAR_INJECTIVE_MAPS": "1",
    "FLOWSTAR_HORNER_EDGE_CACHE": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/cache_horner_entry",
    "FLOWSTAR_CENTER_NORMALIZATION": "1",
    "TORCH_EXTENSIONS_DIR": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/cache_private_entry/py311_torch251_cu126_gcc13",
    "FLOWSTAR_VALIDATION_POLICY": "solution_plus_one",
    "OMP_NUM_THREADS": "1",
    "FLOWSTAR_COMPOSITION": "horner",
    "FLOWSTAR_EARLY_WEIGHTED": "1",
    "FLOWSTAR_INJECTIVE_GLUE": "1",
    "CUDA_VISIBLE_DEVICES": "GPU-1ad11bb9-50d4-6b9d-22f8-fc8c33180e56",
    "FLOWSTAR_SUPPORT_POLICY": "structural",
    "OPENBLAS_NUM_THREADS": "1",
    "CUDA_HOME": "/usr/local/cuda-12.6"
  },
  "source_sha256": {
    "run_fullbatch_p3_observer64.py": "171dd167496895b15fd1373de16986d30118b3338c4140a4f5c91b98eee353c9",
    "observer_endpoint64.py": "d8d87a51a7db600685a79ff19193a56986dab6e84b4825a30570caaf157c8872",
    "check_observer_endpoint64.py": "42d1eb24c1a59f03ad208eb44a5615b87e2ccd456b8af1c9f59c0a823740de61",
    "run_fullbatch_p3_working_eager.py": "e16c197e8338c7798a98509ade44804ca119278a26334aaa47d1a2ac6d38316a",
    "working_eager_segments.py": "c4fb7b098a10ab98d9e9250fc88460ade130225ef710ea5734522df03ca0453b",
    "probe_working_eager_segments.py": "01e454d3e88f2f4dfc09e27be513c153f947eb5a690f27919bb3d8eecce9da82",
    "run_fullbatch_p3_selective_audit.py": "660c75cf22346e71d5044a60b5bbd62320ed38e542727cbcb6805fad909095e0",
    "selective_prune_audit.py": "3658f1fced8b47a601b06e261b9a86fa193339dd1df55be64db9a418c558c9e7",
    "check_selective_prune_audit_cpu.py": "ad78926c489429d44ef74391b1cd4411ab842d85c077778064f9404f5fa22fe1",
    "phase_profile.py": "0398dc1bb032fe85c7ef0a192e5667a12890934d85a71d5ab717adf360023166",
    "check_phase_profile_cpu.py": "86a21df970bc7bf29c3816b52f9a7edd593a6bfa662faa6dcf8c16b3f29bc025",
    "run_fullbatch_p3_phase_profile.py": "b4d9927d89b61b2829de0733c00344b70282645c6254d6d2db61fde65927498d",
    "run_fullbatch_p3_phase_profile_gpu14.py": "780240cbaf08f02fc7f0a6967bfa7f00218395e5419a630ab075e78121775e7e",
    "check_phase_profile_gpu14_cpu.py": "5911aeb6917f0094702fa1282b023cd646539fba12d0a2218f3716dbb958a505",
    "nncs_watchdog_gpu14.py": "4298689de9ccf8a97c0007334dcdbe6aedeb246b8acead81a4b1873941a56350",
    "run_fullbatch_p3_selective_audit_long.py": "49da2756eb7637e1d3feeae27927bacd78f93511aa61bc802b863d2b3e1fec46",
    "run_fullbatch_p3_working_eager_long.py": "3efc0a39166856042ad22a4d1085ee780f11ba06c84889cf0a80ac4fd9dc7366",
    "run_fullbatch_p3_gpu14.py": "9d39ffdc78e37b108fa6518a39ed38ee58ad722ced84fffcc086f719aa6a38e9",
    "run_fullbatch_p3_trig_continue.py": "fa3b227110e53f56d53d4c1ab4f8422f48cccd1f667722d332d586b5d24a6287",
    "check_trig_continue_cpu.py": "4bdc24bf2adf8ad3d79ddb9efe1f9e015a607bbf40c118b4ed6444174042bbee",
    "trig_continue_cold_sequence.diff": "7fe06153d11931e9008af69a12e3fa2e4aa301bb0363adb2d38d973d999be214",
    "trig_direct_reuse.py": "304ebc9c783fde32c8c1b4f6117ba83548198e9d38820398928e25533a1f0f07",
    "run_fullbatch_p3_trig_reuse40.py": "1bbf4e7a8a81b2efb453da6cf9e3f6f8b3bc56a08eda1958088998361d2c4ee9",
    "check_trig_direct_reuse.py": "9a9429011384b3138f2c80c619152efb099221b9de41c61bdb659cb1dc65e808",
    "full1024_p3_phase_profile_gpu14_40_v1_COMMAND.json": "3705e932fbe1c1c54a47f10694fd0dcf5281f682ee4ba657c3295b6f51137e01",
    "full1024_p3_gpu14_1000_v1_COMMAND.json": "959a4254cabcf5bc1cb2dd35841210ab2b3576881fdd64a76da511efaa4b11c9"
  },
  "CPU_preparation": {
    "local_path": "results/quad_targeted_recovery_20260927/trig_continue_cpu_v2_20260929/RESULT.json",
    "remote_path": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/trig_continue_cpu_v2/RESULT.json",
    "sha256": "295fb2f23f3a9322f48fd020e262db3ae25c09705b22687c9810f0e49fde5eec",
    "status": "passed_CPU_actual40_protocol_argument_and_cold_source_checks",
    "CUDA_executed": false
  },
  "candidate40": {
    "path": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/full1024_p3_trig_reuse40_v1",
    "RESULT_sha256": "49dcd81db47e65a9d18c7e66917c86363148fced1e4238a912e251687815e523",
    "INPUT_sha256": "38e00dcfc2fb830c9152cff52f1a6424b57e91eaf249bada6116104ccc5717ca",
    "TRIG_REUSE_RUN_sha256": "f94e7db1392327cc6ed35bc9fcf333e495d76f1c4114278aa7cb4cead8272337",
    "REFERENCE40_COMPARISON_sha256": "beca30ebcb4a6fa47d740394af1a349756af4244d3ab19c13126e698a8c0fba3"
  },
  "preflight_requirements": {
    "all_pins_match": true,
    "prior_GPU3_diagnostics_terminal": true,
    "GPU3_compute_processes_empty": true,
    "GPU3_free_MiB_at_least": 15360,
    "output_watch_LAUNCH_absent": true,
    "do_not_touch_GPU2": true,
    "root_scheduling_release_required": true,
    "prior_cold_PIDs_and_groups_absent": [
      1654094,
      1654095
    ]
  },
  "uploaded": false,
  "launched": false,
  "cold_numerical_qualified": true,
  "long_numerical_qualified": false,
  "executable": true,
  "base_command_sha256": "959a4254cabcf5bc1cb2dd35841210ab2b3576881fdd64a76da511efaa4b11c9",
  "resource_budget": {
    "gpu_guard_bytes": 15032385536,
    "torch_allocator_bytes": 14495514624,
    "rss_guard_bytes": 12348030976,
    "same_gpu_uuid": "GPU-1ad11bb9-50d4-6b9d-22f8-fc8c33180e56",
    "reason": "User requested higher budget; do not relabel previous11.5GiB failure."
  },
  "scope": "Own candidate40 and actual cold20->21 qualified; original GPU14 initialize historical prerequisites then actual candidate gates before install. Original LongRuntime/main,1000steps/50NN/SR1000 and failure semantics unchanged. Root review and fresh GPU3 admission required before one launch.",
  "source40_original_actual_file_gate": "selective.reference_pair_gate: original40 observer PT/sidecar/reference links and manifests",
  "argv": [
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/nncs_env/bin/python",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/nncs_watchdog_gpu14.py",
    "--output",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/full1024_p3_trig_gpu14_1000_v1_watch",
    "--timeout",
    "36000",
    "--",
    "taskset",
    "-c",
    "14-17",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/nncs_env/bin/python",
    "-W",
    "error",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/run_fullbatch_p3_trig_continue.py",
    "--mode",
    "long1000",
    "--base-command",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/full1024_p3_gpu14_1000_v1_COMMAND.json",
    "--candidate40",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/full1024_p3_trig_reuse40_v1",
    "--candidate-cold",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/full1024_p3_trig_cold20to21_v1",
    "--output",
    "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/full1024_p3_trig_gpu14_1000_v1",
    "--candidate-cold-result-sha256",
    "69f104fd9ffe080d452ddcf4e44e38a8a2b398bbd6b2ef5247c211641a4ae49e",
    "--candidate-cold-receipt-sha256",
    "6f252643203395718bf33fcfe5d3a70c89f25abcf3f369b19914d7ed8d9b0af1"
  ],
  "candidate_cold": {
    "path": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/quad_fullbatch_p3_20260928/full1024_p3_trig_cold20to21_v1",
    "RESULT_sha256": "69f104fd9ffe080d452ddcf4e44e38a8a2b398bbd6b2ef5247c211641a4ae49e",
    "TRIG_CONTINUE_RUN_sha256": "6f252643203395718bf33fcfe5d3a70c89f25abcf3f369b19914d7ed8d9b0af1",
    "INPUT_sha256": "7d91744ffd1966e5860cde78c842ba2dc6fc1234a83c3e03dd3c8b661593885e",
    "status": "candidate_cold_passed",
    "completed_step": 21,
    "accepted_lane_steps": 1024,
    "four_snapshots_and_four_artifacts_passed": true,
    "execution_identity_sha256": "3fa122eac709815769ff1ac34bc0bc4707a1516dadddee0dc36ccfcebd960285",
    "source_identity_sha256": "c3b4f6904d3d31c7483970a58c44fe209164c332f0eda18d4fce605b79dc0471"
  },
  "postrun_requirements": [
    "Preserve original per-lane failure classification; never equate visited horizon with all lanes accepted.",
    "If completed_all_lanes:1000steps/1024000accepted/NN50 at0..980/SR reset epoch1000 and final endpoint.",
    "Report exact source/run identity and source40/cold links; original main script SHA kept truthful.",
    "Inspect actual candidate counters; no Python counter as replay count or five-repeat performance claim."
  ]
}
```


## 附录E：交接整理时的本地文件校验

下列 SHA256 由本次交接直接读取本机文件计算，绑定本地版本；这不是重新访问服务器或重做数值审计。新对话若发现这些文件变化，先确认是否为后续工作更新。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| [EXECUTION_STATE.md](/Users/shengenli/Documents/ChatGPT/verification/EXECUTION_STATE.md) | 325831 | `078c922c6957445f3086010d779af89d88c1fdbb07da74321645d5be2bd14827` |
| [GOAL_PYTORCH_GPU_MAINLINE.md](/Users/shengenli/Documents/ChatGPT/verification/GOAL_PYTORCH_GPU_MAINLINE.md) | 24153 | `9b3839507fe8fd209fc48b8a923a15591f0ea0764ceccc23c7f48c953378aded` |
| [GOAL_ARCHCOMP_FAILURE_RECOVERY.md](/Users/shengenli/Documents/ChatGPT/verification/GOAL_ARCHCOMP_FAILURE_RECOVERY.md) | 20129 | `fc9a02534003f83b0b03112101217091503ef2bc57de2f4f649e3e04d0f6bb6f` |
| [progress_review_20260923_publication.json](/Users/shengenli/Documents/ChatGPT/verification/output/progress_review_20260923_publication.json) | 1089 | `3275bd0dc30b849427714cfebfc87507b10448c066e54f14f303803ae4fe4bec` |
| [quad_trig_INPUT.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_trig_INPUT.json) | 2759983 | `45b7ba4babb6129067b51e23092383574a943dfc3d05e25e21fd4bd188f05546` |
| [quad_trig_COMMAND.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_trig_COMMAND.json) | 8888 | `9d44c14d9c9d40fcd45f8282aac70e7998853f1d499d31b4eebb4b5ad7b9d4f8` |
| [quad_trig_RESULT.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_trig_RESULT.json) | 3157594 | `310bb9acb6780845c2d336e96de344e25bfbb2916cd9b721362296fc75c8ae98` |
| [quad_trig_audit.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_trig_audit.json) | 1092849 | `517bd183820d43af794d92b64434d84e05e9fb4fa120927f2e9bf3c41968add2` |
| [quad_trig_root_review.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_trig_root_review.json) | 1677 | `5ce9c48388673c495ff47e4bf4dc8dc775a93d00030760899bf3f2182977c0ad` |
| [quad_trig_local_verification.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/quad_trig_local_verification.json) | 3327 | `1b4a6b2d9a0425083e67f3ec5e4dbad68ec22c766aad2f1016422e515a0750de` |
| [native_pause_snapshot.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/native_pause_snapshot.json) | 4999 | `3f240c731298dce6b2174d496ba052a35a5d1c69b2bce7daf7a4f137636fd865` |
| [huan_parity_campaign.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/evidence/huan_parity_campaign.json) | 1798 | `8651b97943518f6d16e6d73787df624292b3b903ddb9001c5d578cb823144947` |
| [SOURCES.json](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/SOURCES.json) | 26088 | `08ce5ba433e79f1ea3c5b2db0ed62c2b93918fca7a2c55a305448134b27485af` |
| [VALIDATION.md](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english/VALIDATION.md) | 794 | `342abce53ce73ed6efba96d4338576d32c764ee86a00b932082e5b46a8392a64` |
| [flowstar_progress_since_slides_20260930_english_tex_bundle.zip](/Users/shengenli/Documents/ChatGPT/verification/output/flowstar_progress_since_slides_20260930_english_tex_bundle.zip) | 1180035 | `73087a6255c42e6b3e6e75d5825f5f177246b79e9b89a059e9d98ebc6a220775` |


### 可访问性与未知项

- 本次没有联网刷新上游仓库，也没有检查当前SSH/服务器PID/GPU占用。
- 原生QUAD截至最后归档仅400步；现在终态未知。
- 本地整包partial不能作为完整终态张量，模型/大PT/SR路径在远端manifest。
- 最新代码并未由一个新commit完整代表，也无最新统一push回执。
- 旧goal历史中的active/blocked/未执行字段不覆盖本文件开头的最新暂停状态及实际结果。
- 接手后优先取得缺失的终态或输入材料；不能把未观察事实补成确定结论。
