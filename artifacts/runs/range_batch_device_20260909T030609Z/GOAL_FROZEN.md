# Codex Goal：真实批量取范围与单算子 CUDA 原型

日期：2026-09-09  
仓库：`https://github.com/lsnnnnnnnn/torch_tm_flowpipe`

## 0. 本轮主线与交付目标

**把已经通过完整对照的“有序张量取范围”，变成能处理真实不同任务的批量接口，并在同一个数学操作上完成一个有严格误差处理的 CUDA 原型。测清分组、打包、计算、回传的全部成本，再决定它适不适合后续 GPU 求解器。**

不是再追求一次 CPU B1 的 1.5×，不是重写整套 GPU Flow*，也不是重新审计 Xiangru/Huan。CPU 不需要先追平 C++ 才能研究 GPU。本轮只开发“给定多项式及变量范围，求包含范围”这一项操作，不扩张到历史队列重写、控制器或其他系统。

必须产生可运行的局部批量接口、正确性测试、真实输入测量和清楚的设备选择结论。不能只重复上一轮 profile 后写“将来可以 batch”。同样不能把离线请求回放、单算子速度或数据常驻 GPU 的数字写成整个求解器已经加速。

### 上轮结果作为既知事实，不重新研究

- 端点代入误差修复始终保留。
- prepared replay：同一次接受后收紧中复用不变的多项式工作，动态余项与安全判断照常执行。
- 新范围操作：独立项相乘张量化，变量幂在一次调用内复用，最后按原项顺序逐项向外相加。
- 两系统各 1000 步的数值载荷、四项上下界、端点误差及关键状态逐位一致。
- VDP 自适应仍为 246 接受 / 35 拒绝。
- 完整 fresh 配对：Brusselator 1440.102287 → 1121.579599 秒，1.283995×；VDP 696.164242 → 575.061501 秒，1.210591×。
- 每系统只有一对完整测量；三对短实验支持方向，不能称稳定全程倍率。
- 正式状态 `BOUNDARY_EXECUTION_PRESERVED__USEFUL_BELOW_TARGET`，优化默认关闭。
- 上轮当前测试口径 1076 passed / 2 optional skipped；不是前一轮 1108/11 的完全相同测试集合。重复局部测试不再加总。
- 整体求解器尚未形式化证明；CPU 对照不是不可质疑的真值。

## 1. 固定源码与工作树

| 身份 | 提交 |
|---|---|
| 起始证据 / 仓库提交 | `ec6abec2bed30f0b7b52acb6b5b56723cea6918e` |
| 当前范围操作数值源码、上轮正式运行 | `5f37cbe0427c480ef0ebbbcaba292143bc8f4ede` |
| prepared replay 数值源码 | `1551ab57aef7324f91882beeba9d368f36b3cdd5` |
| 最终端点修复数值源码 | `0714e475ed9e73bec31619c9c690d1fd63de3d36` |
| 历史同合同 Flow* | `b85a3211748cb77b736fe4ad42ee02d8d2b81148` |

已知目录：`/srv/local/shengenli/boundary_execution_20260908T172756Z/package`。若没有 `.git`，依次检查同级 `repo` 和已知自己的主仓库，不盘点整个服务器或十个第三方 clone。

```bash
set -euo pipefail
SRC=/srv/local/shengenli/boundary_execution_20260908T172756Z/package
BASE=ec6abec2bed30f0b7b52acb6b5b56723cea6918e
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
ROOT="/srv/local/shengenli/range_batch_device_${RUN_ID}"
BRANCH="codex/rangeplan-real-batch-cuda-pilot-${RUN_ID}"
git -C "$SRC" status --short --branch
git -C "$SRC" cat-file -e "$BASE^{commit}"
mkdir -p "$ROOT"
git -C "$SRC" worktree add -b "$BRANCH" "$ROOT/repo" "$BASE"
```

保留旧工作树和原始证据。不得 `reset --hard`、`clean`、强推或合并 main。使用现有 py11；记录真实 Python、torch、CPU affinity、线程数、CUDA/驱动/设备。不要顺手升级依赖。

**本轮所有基线必须同时包含：**

```python
with prepared_remainder_replay(True), packed_boundary_execution(True):
    ...
```

1121/575 秒是上轮记录而不是新 fresh 分母；1455、1992 秒更不能成为新优化分母。公开默认开关不在本轮偷偷改变。

## 2. 快速起始检查，不再恢复整个历史项目

阅读：

```text
docs/boundary_execution/REPORT_PLAIN_CHINESE.md
docs/boundary_execution/NUMERICAL_AND_LIFETIME_CONTRACT.md
src/torch_tm_flowpipe/packed_boundary_range.py
src/torch_tm_flowpipe/interval.py
src/torch_tm_flowpipe/polynomial.py
src/torch_tm_flowpipe/accepted_boundary_sr.py
experiments/boundary_execution/README.md
experiments/boundary_execution/profile.py
experiments/boundary_execution/remaining_profile.py
experiments/boundary_execution/oracles.py
experiments/boundary_execution/verify.py
```

在父版本运行一次 README 指定验证器和局部回归。未来数值源码改变后，旧验证器仍只在父快照执行，不能修改旧证据让它接受新 SHA。

现有端点、历史传递、失败回滚、队列清空测试是正常应通过的回归，不重新构造一个“发现已知问题就停止”的审计任务。新发现的本算子错误应在本轮范围内修复并保留反例。真正与本算子无关的安全阻碍单列，不能隐瞒，也不要展开第四方引擎审计。

## 3. 明确现有能力与本轮不做的事

现有 `RangePlan.evaluate` 已接收 `[B, output, terms]`，但：

- 变量幂仍有逐 batch 的标量 Interval 路径；
- 项累加仍有 Python 顺序循环；
- 每次稀疏输入需要打包；
- 不同请求的单项式集合和顺序可能不同；
- 完整 flowpipe、历史队列、调度器仍是 B1。

因此本轮可以研究局部请求批量与 GPU 算子，但不能称为“完整 batch solver 已完成”。

禁止：更改 ODE、主实验初始盒子、阶数、步长、cutoff、误差预算、收紧规则、历史容量；新 tightness 方法；第三个系统；删除端点 E；以抽样代替包含检查；关闭向外舍入；原样复制整套第三方引擎；新全流程 CUDA/NN controller。

允许只读参考仓库内已有张量工具与已经保存的 Huan 严格算术机制，复用代码须保留来源和许可；不能把别的版本的测试通过自动继承到本算子。没有授权重新克隆候选或重做三类历史反例的整套接纳任务。

## 4. 先准备两种明确区分的真实输入集

### 4.1 真实单任务请求集：用于数学覆盖与局部成本

优先复用上轮安全 JSON 的实际完整状态、已导出的取范围输入及 capture 工具。选择两系统早/中/晚与历史清空前后，不再重跑两条完整 1000 步来获取输入。

每个请求保存：

```text
来源系统、数值 SHA、完整状态身份、实际 step、实际调用位置
请求类型：standard / normal / interval-coefficient
原始且有序的 exponent tuple（包括原显式零项）
实际浮点系数上下界、domain 上下界
state variable indices、time index、power-table 是否外供
输出维度、dtype、标量 CPU 结果
```

用二进制十六进制或无损格式保存，不把实际 binary64 时间换成十进制理想值。取不同时间步骤的请求一起运行，只能标记 `OFFLINE_REQUEST_REPLAY`，不能暗示实际求解时这些请求同时可用。

### 4.2 独立任务请求集：用于现实的分组规模

仅使用 VDP 与 Brusselator 的既有方程和算术设置。为局部批量测试预先规定不同初始子盒子：例如把原盒子按 x 8 份、y 4 份形成 32 个子盒子，端点分割要向外包含且检查覆盖。固定方案后再计时，不调分区来选择漂亮结果。

这是测试独立任务的诊断输入集，不改写原 B1 benchmark；不能用更细子盒子的宽度宣称算法比 Flow* 更紧，也不能拿它与未切分 B1 的不同工作量计时直接比较。

对相同 32 个任务，基线与候选完成完全相同的请求。先各取 1–2 个真实步骤即可；B8 取固定子集，B32 取全部。重复同一任务只允许作一致性 smoke，不能作为唯一性能输入。

在相同名义步骤、相同依赖阶段记录可同时准备的请求，报告有序支持结构分组后的组数、有效组大小和 singleton 比例。不能为了凑 B32 把未来步骤或有依赖关系的操作提前拿来算。若没有实际调度接入，称“并发任务请求采集与回放”，不称 integrated flowpipe throughput。

## 5. 实现可运行的分组批量 CPU 接口

建议 API 形态由代码实际决定，例如 `evaluate_range_requests(requests, backend=...)`，返回每个 request_id 的区间与状态。

### 5.1 分组键必须足以确定语义

至少包括：有序支持、变量数、standard/normal 类型、时间/状态变量角色、外供幂表语义、输出结构、dtype/device。不要只按项数分组。

- 相同项但不同排列仍是不同执行顺序。
- 不准给缺少的项补零后无条件参加顺序加法；向外舍入使“加零”也可能改变结果。
- 不准把 normal 的奇偶规则替换为一般域乘积，或者反过来。
- normal 路径保证域是归一化状态范围；输入记录与合同显式说明，不把假设藏起来。超出该保证范围的请求应拒绝或走有说明的一般域路线。
- 外供 step-power table 未实现时明确走原 evaluator，不能忽略其中数值。

实现真正按组执行的局部张量操作，不允许在新接口内部只是逐请求再次调用完整 B1 range，然后称为 batch 提速。打包/分组可以有必要循环；核心独立任务乘法必须共享张量或 kernel。

### 5.2 所有权与失败隔离

每个请求有独立 ID 和状态；新 API 不修改原系数/domain 或保存的完整状态。输出不与输入及其他请求共享可写存储。

正常有限请求必须成功；非法区间、NaN、无法有限包住的溢出按请求明确失败，不能把无穷区间提升为有效证书。一个非法请求不应污染其他请求。掩码关闭的行不参与计算，不把 padding、无效 lane 或失败 lane 计入已验证吞吐。

结构缓存只保存不可变指数/索引。系数、domain、h、余项、变量幂和最终区间不跨调用缓存。

## 6. CPU 数学与行为验收

对 CPU 新分组入口，要求逐请求与当前 RangePlan B1 路径逐位一致，同时对有限代表输入做独立精确有理数检查。参考程序和批量程序相等，不能单独替代数学包含性。

覆盖：

- B1/B2/B8/B32，1/2/3 变量，order4/6，多个输出；
- 不同系数、不同非对称 domain、不同 time/state 角色；
- 项顺序不同、空支持、显式零项、正负零、严重抵消、极小非零数；
- 点系数和区间系数；normal 与普通路径；
- B32 一次、4×B8、16×B2、32×B1 逐请求一致；
- 请求重排后按 ID 复原一致；两个线程/上下文不共享可变数值；
- 输入修改后下一次调用重算，先前输出不变；
- 正常请求与无效请求混放、mask、fallback、取消/失败无跨请求污染。

幂、每项区间及总和都要有独立精确检查。若发现 CPU 既有 pow 在某合法输入不能满足严格包含，不用“和旧版一致”掩盖；把问题范围限定在此算子，保留失败与修复，并区分新严格行为与旧逐位行为。

## 7. 先测批量 CPU 的实际成本，不设 CPU 必须先赢 1.5× 的前提

至少 B1/B8/B32；B128 仅用于明确标注的局部扩展诊断，不重复制造同一任务当成真实输入。

同一请求列表比较：

```text
S：当前最快 B1 RangePlan，逐请求调用
B：新分组批量 CPU 接口
```

比较单位用请求数/秒及整批 wall time。分组、打包、幂计算、所有检查、scatter 回 ID、输出包装都必须计入主表。另列结构已缓存、数据已是张量时的纯计算表，不混在一起。

预热后至少 5 个测量区块，交替执行顺序；报告 median、min/max 或 IQR。无 profiler 的时间才作性能分母。调用数、对象数等用独立遍历采集。

记录“有效分组大小→耗时”的关系。CPU batch 不快，不等于 GPU 没价值；不以 CPU 1.5× 为 CUDA 原型门槛。若现实组全为单例，仍可做一次有界设备实验，但不得基于理想 B32 推导当前应用收益。

## 8. 同一个数学操作的有界 CUDA 原型

CPU 合同与分组隔离通过后，明确授权实现这一个范围操作的 CUDA 原型，不再另外申请一轮只写计划。只保留一种清楚可验证的算术方案，不做多后端大搜索。

### 8.1 硬件与源码

记录实际 GPU、驱动、CUDA、编译器和构建参数。只修改新独立设备模块及必要局部接口；保留 CPU 原路径。使用已有环境，不假定是 V100 或 5090。无设备/授权时保存真实错误，不用 mock 产生 GPU 通过结果。

可优先复用项目内合适的已验证算术助手。需要 custom CUDA 时允许一个数学算子的若干内部 kernel，但不迁移历史队列或全步验证器。

### 8.2 不可照搬的假设

当前 CPU 保持标量 `Interval.pow_int`，不代表 CUDA 的 `torch.pow` 有相同误差边界。不能简单 `.cuda()` 再容差比较。

建议采用固定运算顺序与逐操作向外界：

- 用明确舍入的加、乘原语，或固定 round-to-nearest 后向外扩展；
- 整数幂用有证明的有限基本运算，正确处理奇偶、负数、跨零、0 次、1 次；
- 每项变量次序及总项加法次序固定；不引入未说明的并行 sum/FMA；
- no-FTZ/极小值行为、非有限中间值、溢出条件有实际设备检查；
- 算术误差界本身也必须向外覆盖；不能统一加经验 epsilon。

NVIDIA device intrinsic 支持显式舍入的双精度加乘；根据本机 CUDA 文档确认可用接口，不作“GPU 不支持 directed rounding”的笼统判断。不得把 CUDA 标准文档当成已测试本机 kernel 的证据。

### 8.3 CPU 相等与 GPU 包含性分开

CPU 分组模式优先要求逐位等于当前参考。GPU 必须满足已声明的数学包含合同及局部精确 oracle；不要求 CPU/GPU 或两个正确区间彼此相互包含。

若 GPU 采用不同但保守的整数幂边界，可能出现少量宽度差异，必须显式报告。不能为追逐 bitwise 删除必要误差，也不能仅靠 tolerance 宣称 sound。CPU 当前基线及公开默认行为保持不变。

至少检查每项、总和、B1/B2/B8/B32、请求重排/分块、异构输入、无效 mask、相消、subnormal 和溢出。记录真实 kernel invocation>0，不能把 extension 可导入当成生产执行通过。

## 9. GPU 计时必须区分三种现实

对完全相同的有限请求、同样顺序与有效项数，报告：

1. **数据本来就在 GPU 的算子耗时**：输入、结构驻留，含 kernel 内部必要检查。
2. **当前 CPU 调用者的完整请求耗时**：分组、打包、H2D、kernel、D2H、校验及拆包全计入。
3. **分组 CPU 耗时**：新 CPU 接口，作为 GPU 最相关的比较对象。

另列首次扩展编译/加载成本和常驻复用次数。GPU 计时预热、同步；记录峰值显存，不能以异步启动时间冒充完成时间。

固定 B1/B8/B32，在实际非重复请求集上至少 5 个测量区块。B1 更慢可以接受，但必须展示，不选择性隐藏。公开的额外 B128 理想规模只作扩展诊断。

工程判断（不是安全证明）：

- B32 resident 相比已经分组的 CPU ≥2×：具有设备常驻研究价值；
- 包含所有 CPU↔GPU 往返后仍稳定快于分组 CPU ≥1.2×：当前请求 offload 有实用价值；
- resident 快、往返慢：明确 `RESIDENCY_REQUIRED`，不建议逐次 B1 调用跨设备；
- 没明显收益或现实组太碎：保留数学接口和负结果，不迁移第二个算子凑速度。

不要求单算子使完整 solver 再快 1.5×。上轮候选代表步骤中 D 份额约 17.7%、29.0%、25.5%；即使将其完全消除，对那些同一工作量样本的整体上限也只约 1.22、1.41、1.34。它们不是全程精确份额，更不能忽略其他工作预测整个 T20。

给出同一批请求真实调用频率下的 Amdahl 估计，分组/传输开销另加。估计必须标 `PROJECTED`，未执行实际并发 flowpipe 不得标 `MEASURED_SOLVER_SPEEDUP`。

## 10. 与真实求解器的有限接入检查

本轮必须保留实际调用适配能力，但不要求写完整多任务调度器。

CPU 分组 API：用真实完整状态和正常调用链，核对 B1 回退及至少两个独立任务的结果，不通过最终发布盒子重建下一步状态。

若 GPU 原型通过局部数学与设备测试，可做一个可选 B1 range offload 的 1–2 步及 20 步接入诊断，CPU 继续执行其他部分。其用途是确认范围消费者、端点 E、历史余项及失败路径没有断开；很慢也如实记录。不能把它包装成完整 GPU solver。

至少在 VDP99/100/101、Brusselator999/1000/1001 的已存完整状态附近做小型 CPU 回归；实际 range 输出变化时，还须检查下一步消费完整区间，不应仅复算静态 CSV。

本轮默认不重复两套完整 1000 步计时，因为主任务是局部 batch/设备可行性。若修改了主 CPU B1 数值语义或正式宣称新完整 horizon/宽度，则必须新鲜长跑；否则旧长跑严格标 REUSED，不改原档案、不编造新全程结果。

## 11. 结束时给出明确的下一步，而不是继续无尽 B1 小修

必须将结论分成三层：

- `correctness`：CPU 分组保持语义？GPU 算子满足自身的包含合同？
- `local_throughput`：真实不同请求、含打包/传输是否快？需要什么组大小？
- `solver_integration`：已实测到哪里？哪些只是常驻/离线假设？

允许状态：

```text
RANGE_BATCH_CPU_CLOSED__CUDA_LOCAL_PILOT_USEFUL
RANGE_BATCH_CPU_CLOSED__CUDA_RESIDENCY_REQUIRED
RANGE_BATCH_CORRECT__NO_MATERIAL_DEVICE_GAIN
RANGE_BATCH_CORRECT__REAL_GROUPS_TOO_FRAGMENTED
RANGE_BATCH_PARTIAL__LOCAL_NUMERICAL_BLOCKER
RANGE_BATCH_CPU_CLOSED__CUDA_HARDWARE_UNAVAILABLE
```

不要仅因为局部算子通过就授权“完整 GPU 迁移完成”。

若 resident GPU 明显有价值而传输主导：下一轮要决策的是一个更完整边界状态在设备上常驻，或复用现成严格 GPU 架构；不能接着写第四轮 CPU B1 1.5× 小修。

若现实分组不足：说明当前表示/调度与可批量数学工作之间的缺口，不增加第三系统寻找正结果。

若发现全步剩余时间主要在历史状态，记录现有 evidence 支持的比例，不本轮重写 queue；下一轮只在确有数据时选择边界/历史常驻的连贯改造。

Huan/Xiangru 的架构与已有严格修复仍是可选资产。本轮算子检查不等于采用或否定整套后端，也不要求自研 CPU 先比 Flow* 快。

## 12. 文档、测试与证据保持精简

建议输出：

```text
docs/range_batch_device/REPORT_PLAIN_CHINESE.md
docs/range_batch_device/OPERATOR_CONTRACT.md
artifacts/runs/range_batch_device_<UTC>/
  SOURCE_MAP.json
  REQUEST_CORPUS_MANIFEST.json
  grouping_statistics.csv
  cpu_batch_equivalence.json
  cuda_arithmetic_checks.json
  cuda_actual_invocations.json
  timing_samples.csv
  timing_summary.csv
  transfer_and_packing_costs.csv
  integration_checks.json
  RESULT.json
  tests/commands.json
  tests/*.xml / *.log / *.exit
  SHA256SUMS
```

父完整状态和长跑引用现有路径与摘要，不重复复制几千行大档案。原始请求保存足够复算，禁止不可信 pickle 作为证据读取格式。

新 verifier 从原输入/输出与计时事件重算分组、包含检查、倍率和最终状态。少量针对性篡改测试覆盖请求 ID、支持顺序、系数、mask、误差、设备调用数、时间分类与结论。测试总数按身份去重；父历史测试记录单列 REUSED，候选-on 复查不再次加总。

运行当前根测试与新局部测试，记录原有可选跳过。外部历史 source-locked verifier 在自己的旧快照上执行，不修改它适应新代码。不要再把整个历史测试矩阵的恢复变成本轮主要任务。

最终在独立副本运行新 verifier 和相关局部测试。明确这不是重新长跑。保留数值源码 SHA、测试源码 SHA 与证据封装 SHA 的分别记录；push 本轮新分支，不强推、不动 main。

## 13. 中文报告必须能直接用于汇报

不靠代号讲故事，按以下顺序回答：

1. 原来是一份多项式内许多项一起算，现在是否真的能让不同任务一起算？
2. 不同任务的项结构有多相似，实际能组成多大的组？
3. 怎么保证不会漏误差，也不会把任务 A 的输入/输出串给 B？
4. CPU 批量、GPU 已驻留、CPU↔GPU 往返三种时间各是多少？
5. 快的是整个请求还是只是 kernel？实际完整 solver 测到了哪里？
6. 为什么下一步值得做状态常驻/调度，或为什么应该停止这一设备路径？
7. 哪些旧成果原样保留，哪些没有新测？

## 14. 查证依据

仓库内上轮报告、实际代码及证据是本轮起点。设备方案属于本轮新研究，不是上轮已经支持的结论。

可查官方资料（按本机已安装版本核对，不因此升级环境）：

- PyTorch Numerical accuracy：批量/逐片以及 CPU/GPU 不能默认逐位相同。
  https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- NVIDIA CUDA Math API，Double Precision Intrinsics：`__dadd_rd/ru`、`__dmul_rd/ru` 等显式舍入原语及其条件。
  https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html

**本轮成功是：得到一个数学合同清楚、真实不同请求可批量调用、设备成本已测量的范围算子，并据此作出下一阶段架构选择。不是继续凑某个 B1 倍率，也不是再发表一轮只有“发现问题、未实施”的报告。**
