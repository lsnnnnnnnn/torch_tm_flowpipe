# Codex Goal：减少在线 GPU 的碎请求，并补齐原始两系统的 GPU 长时域验证

日期：2026-09-10  
仓库：`https://github.com/lsnnnnnnnn/torch_tm_flowpipe`

## 0. 本轮要交付什么

**在已有在线求解器上，把当前已经就绪的多个独立范围请求作为一个设备工作包传输和执行，减少按小组反复分配、拷贝、启动和等待的成本；随后做真实在线前缀配对，并从原始未分区初始盒新跑 VDP T10 与 Brusselator T20。**

这不是再做离线请求回放，不是继续搜索等待时间参数，也不是直接重写完整 GPU 求解器。GPU 仍只负责范围运算；本轮可以改变请求组织、设备描述符、传输和内存生命周期，但不新增第二类数学算子。

大方向不变：在保证范围可靠的前提下，使 PyTorch 求解器获得真正可用的批量/GPU 执行能力。不能把通过一个约 10% 的前缀门槛说成已经追平 Flow*，也不能因为 GPU 算术与 CPU 不逐位相同就直接否定。

本轮有两个独立结果轴：

1. **工程效果**：新工作包是否比现在的在线 GPU 路线真正更快，而不只是小 kernel 更快。
2. **长时域效果**：使用已检查 GPU 范围算术的真实求解链，从原始初始盒能否完成两套 1000 步目标，整个过程宽度怎样。

性能未达目标不自动否定正确实现；局部算子正确也不自动获得完整时域结论。分别报告，不合并成一个含糊的 PASS。

## 1. 已经知道的状态，禁止重新做成一轮审计

| 角色 | 固定提交 |
|---|---|
| 本轮父交付 | `e4d920aa710bd407666c60285632a4c294edb9da` |
| 已在线运行的科学源码 | `ef4e2f0c17518a4c4989071a596cd8631235c5ea` |
| 现有范围请求及 CUDA 数学核心 | `440714afc3acc250903c5ec98fb0011905671944` |
| 已有张量范围版本 | `5f37cbe0427c480ef0ebbbcaba292143bc8f4ede` |
| 已有 prepared replay | `1551ab57aef7324f91882beeba9d368f36b3cdd5` |
| 端点误差修复运行时 | `0714e475ed9e73bec31619c9c690d1fd63de3d36` |
| 历史原生 Flow* 对照 | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` |

父分支：`codex/live-range-batching-solver-integration-20260909T053007Z`。

父结果已经完成：真实任务到达范围调用后提交当前输入，等待真实返回再继续；取消、提交、旧返回、恢复、同后端等价和有限长依赖检查。共 110 个诊断案例、60 次正式计时；独立副本又实际完成 32 个接受 lane-steps。不是只有离线回放。

父实验主要数字：

| 指标 | VDP | Brusselator |
|---|---:|---:|
| B32，20步，每组 min(S,Q)/G | 1.105566 / 1.156779 / 1.034097 | 1.167667 / 1.113319 / 0.974747 |
| 三组配对倍率中位数 | 1.105566 | 1.113319 |
| 正式 G 完整 wall 中位数（秒） | 304.436 | 430.507 |
| 在线平均范围组大小 | 约 2.72 | 约 3.64 |
| 请求等待 P95（毫秒） | 约 138.378 | 约 121.010 |

两个中位数按原预注册规则通过，但接近门槛；Brusselator 一组 GPU 更慢。它们只属于该资源、分区和20步前缀。

名义20步分别只有 VDP 时间0.2、Brusselator时间0.4；不要写成 T10/T20 完整GPU性能。两个不同子任务各120步的检查、从旧 checkpoint 恢复到清空附近，也不等于从原始盒新跑1000步。

父报告里的“20ms”是请求获得派发资格的等待上限，不是保证20ms内返回。已有组正在执行、单核调度等仍会延迟服务。不得把它宣传为硬实时期限。

已知数学边界：原 `Interval.pow_int` 的三次幂问题只在新严格请求入口校正；旧默认L不能当无条件真值标准。GPU和CPU各自有包含依据，二者正确区间不必相互包含。整个ODE求解器仍未形式化证明。

## 2. 工作树、环境与读取范围

优先使用已知目录：
`/srv/local/shengenli/live_range_solver_20260909T053007Z/repo`。

```bash
set -euo pipefail
SRC=/srv/local/shengenli/live_range_solver_20260909T053007Z/repo
BASE=e4d920aa710bd407666c60285632a4c294edb9da
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/srv/local/shengenli/live_gpu_packets_${RUN_ID}"
BRANCH="codex/live-gpu-range-packets-and-long-horizon-${RUN_ID}"
git -C "$SRC" status --short --branch
git -C "$SRC" cat-file -e "$BASE^{commit}"
mkdir -p "$ROOT"
git -C "$SRC" worktree add -b "$BRANCH" "$ROOT/repo" "$BASE"
```

目录不存在时只检查已知同轮 package 与自己的主仓库，不全盘寻找十个旧 clone。不 reset/clean 旧修改，不强推，不合并 main。

继续使用 py11、现有 PyTorch/NVRTC/驱动和实际 V100。CPU主性能保持与父实验相同的CPU2、intra/inter-op=1和GPU0；预先统一设置线程与设备，不在worker里竞争改全局配置。保存实际版本与导入路径，不升级依赖。

必须读：

```text
docs/live_range_solver/REPORT_PLAIN_CHINESE.md
docs/live_range_solver/SCHEDULING_CONTRACT.md
experiments/live_range_solver/README.md
experiments/live_range_solver/runner.py
experiments/live_range_solver/verify.py
experiments/live_range_solver/verify_package.py
src/torch_tm_flowpipe/live_range_service.py
src/torch_tm_flowpipe/live_range_checkpoint.py
src/torch_tm_flowpipe/range_requests.py
src/torch_tm_flowpipe/range_cuda.py
src/torch_tm_flowpipe/range_cuda_kernel.cu
docs/range_batch_device/OPERATOR_CONTRACT.md
```

读取父包的 `PERFORMANCE_RESULT.json`、`paired_speedups.csv`、`time_partition.csv`、`actual_grouping.csv`、`coverage/coverage_summary.json`。已完成的父核验回执只读引用。开始时只跑受影响的局部基线，不重新执行全部110诊断/60计时；新核验器不要强行要求旧源身份验证器在新工作树通过。

## 3. 数学与比较合同固定

两系统仍是原方程、原输入和原表达式顺序。以父冻结runner与合同为准，不重新录入一套近似参数：

- VDP：原方程，order4，固定 binary64 h=0.01，原余项预算、cutoff、validation epsilon及SR100。
- Brusselator：原方程，order6，固定 binary64 h=0.02，原余项预算、cutoff、validation epsilon及SR1000。
- 原端点E修复、接受后的收紧规则、491次上限、0.99停止规则、左右映射、历史owner与已接受状态规则不变。
- 两个已有执行优化在全部主要路线开启；新工作包开关独立，默认关闭。
- 不改阶数、时间步、初始分区、历史容量、精度或误差预算来取得成功。

前缀性能复用固定8×4的32个不同子盒；B8仍为原指定子集。文件是：
`artifacts/runs/range_batch_device_20260909T030609Z/PARTITION_PLAN.json`。

本轮完整1000步使用**原始未分区初始盒**，不能误用 task0 子盒。原始B1、B8/B32子盒、恢复窗口分别列出，不能横向拼出性能或tightness结论。

不接CROWN/NN controller，不加第三系统，不改Flow*，不重新审计Huan/Xiangru。已有第三方架构仍可作只读参考；本轮结果不决定永久弃用任何后端，也不要求CPU先追平C++才能继续GPU。

## 4. 先判断在线请求为什么碎：有限测量，不再开展全项目profile

先利用父保存的诊断生命周期和组记录；必要时每系统新增一个B8短窗口、一个B32短窗口的轻量记录。不要用保存的请求/答案来推进任务。

回答四件事：

1. 派发瞬间共有多少已就绪请求？其中多少因数学key不同被留在其他队列？
2. 主要是所有任务尚未到达调用，还是已有请求因key分裂没能同次处理？
3. 实际时间用于：CPU打包/校验、设备分配、结构传输、数值传输、kernel、完成同步、回传包装、锁与Future调度各多少？
4. 一次派发能合并多个语义组时，可以减少多少提交、传输和内存操作？这只是机会估计，必须标为PROJECTED，不是线上速度。

必须区别：设备纯kernel时间与host观察到的kernel+sync时间；等待时间的并集与各任务等待总和；线程CPU时间与墙钟。不同任务等待不能相加充当总运行时间。

若trace没有足够信息，增加最少记录补齐，不猜测PCIe带宽、GPU占用或GIL是根因。若主要瓶颈是非范围数学，明确量化本轮方案的整体上限，不自动迁移第二数学阶段。

## 5. 唯一新机制：多个独立语义组装进一个设备工作包

### 5.1 “一起传输”不等于“改变它们的数学表示”

现服务一次选择一个结构key。新模式可以在满足既有派发条件时，从其他已经就绪的key取出有限数量请求，一起提交设备。

每条请求仍完整保留：

```text
run / task / epoch / generation / attempt / counter
有序支持及显式零项
系数、domain、输出个数
standard / normal / interval-coefficient
时间/状态变量角色、外供幂表规则
依赖前驱、状态及独立结果
```

**禁止把不同支持补零、重排、做union后套一个公共求和次序。**允许使用不同长度的连续存储加offset描述符：每条请求仍有自己的项序和运算描述，只在物理传输/执行提交上合并。

外供幂表仍走原安全回退，明确计为CPU工作。空支持保持原精确零。不可为凑包执行未来请求、重放已有答案或让任务多算一步。

### 5.2 设备实现建议，但以真实成本为准

实现一个有界 `RangeWorkPacket` 或等价对象：

- 系数/domain/操作描述连续存放，各请求用checked offset/length定位。
- kernel可按全包的power条目、term条目和output条目执行，各条目明确归属请求。
- 单个请求内部的directed幂、项乘法和顺序累加保持原计算链。
- 优先使一包使用同一组4阶段kernel，而非每个key各启动4个；不能只把Python列表改名成packet就称完成合并。
- 如果不同形状的单次执行成本过高，可以保留多次kernel，但必须如实计数，并实际证明其他成本下降；不预先保证速度。
- 实际上减少了多少device launch、拷贝次数、metadata字节和临时分配，必须从执行计数确认。

只迁移这一类范围运算。不要在同轮增加端点代入GPU版、Picard GPU版、历史矩阵GPU版或完整稠密求解器重写。

### 5.3 常驻什么、不常驻什么

允许常驻或复用：

- 有界不可变支持/操作描述；
- 有界scratch buffer与pinned staging buffer；
- 已编译模块。

数值输入每次来自当前任务；不隐式缓存上一步系数/domain/幂/最终答案。buffer复用只是内存复用，不能假装数据没变。若实际没有跨多个数学操作保留状态，不得写成“完整边界状态常驻GPU”。

stream/event必须管理缓冲生命周期：一包完成且最后消费者已取走数据前，不能覆盖其输入/输出。取消只让结果失效，不能提前把正在被kernel读取的存储给下一包。状态与回执每包重新初始化；不得使用上一包的成功状态。

说明每条输入、输出、描述符归谁所有、何时可复用；不通过关闭检查、删除必要copy或暴露可写别名换速度。scratch容量、包最大请求数/项数/字节数都有固定上限；超限安全切包，不OOM后偷偷改实验。

### 5.4 在线派发与公平性

保留父请求身份和取消/commit锁的规则。每任务一条未完成请求仍是允许的约束。

不重新搜索2/20/50ms；主比较固定原20ms资格规则。新机制减少派发次数和碎片，不靠额外等待凑满组。最老过期请求优先；包里其他请求只能是当时已经就绪的请求。包不能无限扩张导致另一任务饥饿。

任何任务失败、提前完成、重试或取消，其他任务仍继续。旧generation、迟到返回与重复响应被丢弃。返回分派仍检查完整身份，不只校验在一张GPU输出表中的行号。

## 6. 不改变算术时，应提供什么证据

数学primitive和运算顺序原则上与父CUDA一致，因此：

1. 相同单请求经父GPU与packetGPU，状态和上下界预期逐位一致。
2. 同GPU后端，独立执行、父服务、新服务，按task逐步比较完整segment/下一状态，不以CPU/GPU本来不同掩盖同GPU回归。
3. 对新kernel所有主要layout、offset、mask、长度路径，用独立Fraction检查幂、项和总和。不要只调用同一份实现当oracle。
4. CPU/GPU允许不同区间，各自检查包含；不得取两者较窄端点、盲目求交或用抽样替代。
5. 原端点E、已知三次幂、次正规数、相消、溢出与非法输入检查继续通过。

必须新增的针对性测试：

- 同包不同支持、同支持不同项序、不同输出数量、变量角色和domain；
- 显式零/符号零、空项、最大合法幂、极小/极大混合；
- 错offset、越界长度、乘积索引溢出与错请求ID；
- 一个请求非法、一个数值溢出，健康请求的状态和结果不被污染；
- input在提交后被调用者修改，owned版本仍正确；output在交付后被修改不能影响其他任务或下一包；
- 连续多包交替大小、scratch复用、取消在排队/在途/完成未消费三个时刻；
- 新epoch恢复后旧packet回传无效；
- 不同合法派发/切包顺序，按任务结果一致；
- GPU硬件失败的显式停止/CPU回退保持原合同，并单列成本。

结构损坏不能变成成功区间。新路径局部可修复问题在本轮修复，不因“又发现一个测试失败”机械停止整个项目；独立数学缺陷必须保留反例并限制主张。

## 7. 实际接入与有限正确性矩阵

路线命名：

| 路线 | 含义 |
|---|---|
| S | 已校正的新CPU请求接口，独立逐任务求解 |
| Q | 父在线CPU服务 |
| G0 | 父在线GPU服务，当前最快已验收GPU前缀基线 |
| Gp | 新工作包GPU服务 |
| L | 原packed最快路线，仅有限历史速度参照，已知幂限制 |

Gp默认关闭；S/Q/G0不能顺手重构成更慢分母。新功能用独立开关。prepared、端点保护和新请求算术始终保留。

按次序：

1. B1/B2，两系统各2–5步：Gp对G0与独立GPU。
2. 原32子盒的B8、B32各20步：实际在线生成与消费，诊断范围及状态。不读取父保存答案推进任务。
3. 每系统两个不同子任务连续120步；VDP跨SR100。未变父G0数据可作REUSED数值锚点，但新Gp必须实际连续运行。改变输入/配置则不能复用。
4. 只对受影响的取消、失败重试、checkpoint、新epoch、异构请求数量场景重跑，不重复整套110个诊断。
5. 根据检查需要复用安全的历史清空附近状态，清楚标为RESUMED_LOCAL_WINDOW，不能称从初始集新跑到该时间。

正式计时不做Fraction检查或重型序列化。诊断遍历保存原始对象与请求，在离线核验器中复算；新操作布局和失败邻域必须充分覆盖。未逐条重算的请求说明使用哪条已验证数学合同。

## 8. 真正的长时域：原始未分区盒，两个1000步新运行

### 8.1 本轮把此前缺少的能力补出来

局部与连续前缀正确性通过后，新跑：

- 原始未分区VDP，固定1000步，名义T10；
- 原始未分区Brusselator，固定1000步，名义T20。

优先用经过验收的Gp；若Gp仅性能未达目标，但数值完全等价，可照常做正确性长跑并保留默认关闭。若Gp无法通过数值接入，则用冻结G0完成这一独立验证轴，明确标为G0，不冒充Gp成果。安全本身失败时不强行发布证书。

**一个任务在GPU范围服务中跑1000步，不是整套数学全在GPU，也不是B32完整T20。**报告使用“GPU范围参与的完整求解链”，不得改写为全GPU引擎。

不要求本轮32×1000步，不重新跑旧自适应性能矩阵。旧自适应能力不修改，针对受影响路径做回归即可。

### 8.2 结果必须包含完整宽度

保存每一步真实的：

```text
端点x/y上下界
整个小时间段x/y上下界
端点E及普通余项、必要误差类别
实际h与精确累计时间、调度时钟
接受/拒绝、收紧次数与停止原因
历史队列大小、清空、关键完整状态
求解/请求服务/初始化/导出时间，分别标注
```

每步的下一输入必须来自本步实际接受的完整状态，不能用CPU发布盒或CPU检查点周期性纠偏。

与现有CPU和stock Flow*完整对象做共同observer比较：P50/P95/max、最差时间和上下界绝对差，近零宽度另列。旧对象及时间标为REUSED。共同observer采用新严格范围检查，不能拿旧已知缺陷结果作无条件oracle。

GPU/CPU在有效非零宽度处超过1.10时列出具体位置并解释；宽度比不是安全证明。到不了目标则记录第一拒绝原因与已完成前缀，不改h/order/预算救活。

如需要同GPU父版本完整数值锚点，而父版本没有新1000步，不得伪称已有。可以用相同请求算术的局部/连续证据加完整CPU/Flow*对照声明有限结论，或实际新增一次所需父GPU长跑并如实标记；不能把恢复窗口冒充全程。

## 9. 正式性能：基线已经是在线GPU，不重复计算之前收益

### 9.1 主矩阵与资源

主性能仍使用两系统B32、每任务20步、固定不同子盒。每条路线完成同一640个成功lane-steps。

S/Q/G0/Gp 使用同一CPU2、Torch线程1和同一V100。GPU需要服务线程不代表可以另给CPU核心。后续多核实验不属于本主表。

冻结五个配对区块和交替/平衡先后顺序，然后一次执行。不要只为刚好超过阈值追加样本。每个区块都有S/Q/G0/Gp；同时报告Gp对G0、S、Q以及同区块较快CPU。原L可只作有限样本，绝不宣称它获得新的普遍严格保证。

B1/B8诊断与少量时间另列即可，不再把此前60个正式计时全部复制。主要新问题是Gp是否胜过G0。

### 9.2 计时边界

完整wall包括初始状态、worker/服务启动、所有必要数学、安全检查、请求复制、等待、描述符准备、结构/数值H2D、kernel、D2H、scatter、清理。

编译和一次性设备自检可预热单列；首个buffer准备、必要的本轮数值上传不能移到分母外。明确冷启动、热请求、初始化加求解和导出差别。

绝不把parent约304/431秒或更早1455/1122秒直接作为新鲜分母。每个正式倍率必须由同轮匹配测量得到；旧数据只提供背景。

GPU事件可测kernel/传输；host同步span与纯设备时间分开。CPU等待采用区间并集分析，不能把重叠任务时长相加当总时间。

### 9.3 目标与如实分级

主目标：

- 两系统Gp/G0的速度倍率中位数至少1.15，至少4/5配对Gp更快；
- 同时没有相对较快严格CPU的稳定回退；报告每组min(S,Q)/Gp，而非只选更慢CPU；
- 未因取消/失败任务减少工作量；
- 内存使用合理且无持续增长，公开包大小和scratch上限。

这些是本轮工程目标，不是理论承诺。

若只有约1.05–1.15，报告有用但未达目标；若波动很大，标为边界证据。不得用单个kernel倍率替代完整结果。

若完整无收益，保留正确算子与测量，解释究竟是：可合并请求不足、等待/线程成本、非范围数学主导、或布局/传输成本未下降。下一轮不再开一轮等待参数搜索，而应转向一个更大、自然连续的边界数学阶段或评估已有设备架构的复用成本。

## 10. 范围和外部参考的边界

不得通过子盒分区更细，让GPU宽度看起来比未分区Flow*更好。前缀批量与原始B1长跑分别分析。

不得声称修复了全仓 `pow_int`。本轮参与的范围服务走已有校正/设备严格路径，其他数学函数保持原合同和既有局限。

CPU/GPU和第三方版本都按同样标准；我们的实现不因作为reference就免检，Huan/Xiangru也不因曾有反例就永久不能复用。本轮不再次研究他们的历史patch/论文，但最终路线文档可以说明哪些已知资产值得未来采用。

不要用“整个求解器已形式化证明”。可声明的最大范围是本轮操作链、调度契约、实际完整实验与独立局部检查。

## 11. 文件、测试与独立复核

建议新增：

```text
docs/live_gpu_packets/REPORT_PLAIN_CHINESE.md
docs/live_gpu_packets/PACKET_AND_OWNERSHIP_CONTRACT.md
experiments/live_gpu_packets/              # capture/compare/timing/verify，复用父工具
artifacts/runs/live_gpu_packets_<RUN_ID>/
  SOURCE_MAP.json
  PLAN_FROZEN.json
  ready_packet_opportunity.csv
  packet_cost_breakdown.csv
  actual_launch_transfer_counts.csv
  same_backend_state_equivalence.json
  packet_lifetime_fault_checks.json
  timings_raw.csv
  paired_speedups.csv
  full_horizon_matrix.csv
  full_width_comparison.csv
  RESULT.json
  tests/commands.json
  tests/*.xml
  tests/*.log
  raw_minimal/
  SHA256SUMS
```

不要给每次普通PASS新增一个孤立大审计框架。复用现有对象格式、checkpoint、比较器和测试工具，新增文件围绕本轮改变。

测试覆盖：实际kernel布局、分包/重排、buffer生命周期、请求身份和原有数学反例；完整测试按身份去重。历史源锁定测试在旧快照执行，明确REUSED，不删断言伪造全绿。

独立clone：验证源码/配置/原始结果与派生表，重算局部请求和状态连续性，并真实运行两系统B2两步G0/Gp。完整1000步是否重跑必须实说，不能把核验器成功说成重跑了长实验。

加入少量针对性篡改：packetoffset、request身份、generation、bufferepoch、结果端点、实际GPU回执、计时分母、成功lane-step数、完整时域标记。改外层hash后仍应被语义检查拒绝。

## 12. 最终交付必须是白话结论，不只是机器状态

中文报告首页先回答：

1. 一次派发实际从多少小组变成多大一个包？是否真正减少launch/拷贝？
2. 为什么不改变每条请求的误差处理？
3. 五组完整前缀时间分别是多少，相对旧在线GPU和更快CPU各怎样？
4. 从原始初始盒的新GPU范围链有没有跑到T10/T20？全程宽度怎样？
5. 哪些是REUSED、哪些新跑、哪些仍未完成？
6. 下一步是扩大连续边界块、扩大状态常驻，还是停止碎粒度GPU路线？依据是什么？

结果建议分开字段：

```text
packet_correctness: pass/fail
packet_runtime_effect: useful/near_threshold/no_gain/regression
complete_horizon_vdp: achieved/not_achieved/not_run
complete_horizon_brusselator: achieved/not_achieved/not_run
width_relation: measured/not_established
whole_solver_formal_proof: false
full_gpu_engine: false
```

版本分开：父交付、实际科学源码、正式计时源码、长跑源码、最终封装。科学提交之后若改源码，须明确哪些重跑/衔接证据支持最终版本，不能把报告提交冒充运行来源。

最终push新分支并核对local/remote/独立副本，不合并main。不要输出token或带凭据的remote地址。

## 13. 本轮为什么值得做

我们已经证明局部GPU范围算子可用，也证明在线任务真的能够消费它，但当前通常只有2–4条请求一组，完整前缀优势只有约10%。

下一步不是把10%的结果包装成最终成功，也不是抛弃已有实现重来，而是完成一次有边界的工程推进：

> **把独立小请求合成更粗的设备工作包，测清实际收益；同时把GPU范围参与的原始长时域跑出来。**

如果这两件事完成，我们才有依据决定是否进入更完整的连续边界计算和设备常驻。若收益仍小，不能无休止重复等待/缓冲区微调；应把已有可靠算子与独立参考作为资产，转向更大粒度的计算结构。

### 本Goal依据

仓库父交付中的在线报告、调度合同、源码与原始配对记录为主要依据：

- `docs/live_range_solver/REPORT_PLAIN_CHINESE.md`
- `docs/live_range_solver/SCHEDULING_CONTRACT.md`
- `artifacts/runs/live_range_solver_20260909T053007Z/paired_speedups.csv`
- `src/torch_tm_flowpipe/live_range_service.py`
- `src/torch_tm_flowpipe/range_cuda.py`

外部工程参考只说明一般原则，不替代本项目实测：NVIDIA CUDA C++ Best Practices Guide，Data Transfer Between Host and Device（尽量减少往返、合并小传输、明确中间数据驻留）。
`https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device`
