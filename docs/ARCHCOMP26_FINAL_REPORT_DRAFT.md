# ARCH-COMP26 非 VCAS 四方完整实验报告：草稿（不可作为最终成绩）

> 本文只使用当前 2026 manifest、哈希绑定实例合同和哈希绑定结果记录。
> 历史 Huan、旧 PyTorch 与旧 Flow* 数字不进入本报告的当前四方结果表。

## 摘要与边界

目标是在同一冻结 benchmark 合同下比较 PyTorch/GPU、Huan、Xiangru 和 Flow* native 的完整性、进程时间与绝对 flowpipe 宽度。共享驱动仅用于控制变量，不等同于三套独立 NNCS 产品。

- 报告状态：**not final-ready**。
- 实验暂停：**是**。
- 覆盖：16 个实例 × 4 个方法 = 64 个 cell。
- 当前 run 计数：`not_started=64`。
- 最终发布门：`experiments_paused, unresolved_contracts=16, matrix_not_terminal=not_started, unassessed_support=64, nonterminal_cells=64`。

## 比较规则

- 正式目标为 1 次冷启动和 5 次独立进程 steady；冷启动不进入 steady 中位数。长任务可预先声明较少 steady 次数并写明原因，但不得据此取得稳定排名资格。
- 只报告完整请求时域且明确允许性能测量的时间；失败前缀不外推完成时间。
- 宽度始终给绝对上下界、union width 和每分区 mean/max；本报告不计算宽度比。
- 不同共同前缀、domain、变量顺序或单位不会合并为同一比较域。
- `failed`、`timeout`、`interrupted` 与有证据的 `unsupported/skipped` 都保留。

## 全部非 VCAS 配置覆盖矩阵

| 实例 | PyTorch/GPU | Huan | Xiangru | Flow* native |
|---|---|---|---|---|
| `acc-safe-distance` | `not_started` | `not_started` | `not_started` | `not_started` |
| `airplane-continuous` | `not_started` | `not_started` | `not_started` | `not_started` |
| `airplane-discrete` | `not_started` | `not_started` | `not_started` | `not_started` |
| `attitude-control-avoid` | `not_started` | `not_started` | `not_started` | `not_started` |
| `balancing-reach` | `not_started` | `not_started` | `not_started` | `not_started` |
| `docking-constraint` | `not_started` | `not_started` | `not_started` | `not_started` |
| `double-pendulum-less-robust` | `not_started` | `not_started` | `not_started` | `not_started` |
| `double-pendulum-more-robust` | `not_started` | `not_started` | `not_started` | `not_started` |
| `nav-standard` | `not_started` | `not_started` | `not_started` | `not_started` |
| `nav-robust` | `not_started` | `not_started` | `not_started` | `not_started` |
| `quad-reach` | `not_started` | `not_started` | `not_started` | `not_started` |
| `single-pendulum-reach` | `not_started` | `not_started` | `not_started` | `not_started` |
| `tora-remain` | `not_started` | `not_started` | `not_started` | `not_started` |
| `tora-reach-sigmoid` | `not_started` | `not_started` | `not_started` | `not_started` |
| `tora-reach-tanh` | `not_started` | `not_started` | `not_started` | `not_started` |
| `unicycle-reach` | `not_started` | `not_started` | `not_started` | `not_started` |

## 旧 14 项到 2026 manifest 的差异索引

旧结果仅作回归线索；下表不会把旧完成状态或时间提升为 2026 重跑成绩。

| 2026 实例 | 旧配置候选 | 映射状态 | 合同状态 | 已知差异/阻断数 |
|---|---|---|---|---:|
| `acc-safe-distance` | `acc` | `candidate_only` | `unresolved` | 3 |
| `airplane-continuous` | `airplane` | `candidate_only` | `unresolved` | 1 |
| `airplane-discrete` | `—` | `missing_from_legacy_14` | `unresolved` | 1 |
| `attitude-control-avoid` | `attitude_control` | `candidate_only` | `unresolved` | 3 |
| `balancing-reach` | `cartpole` | `candidate_name_mapping_only` | `unresolved` | 3 |
| `docking-constraint` | `—` | `missing_from_legacy_14` | `unresolved` | 3 |
| `double-pendulum-less-robust` | `double_pendulum_less_robust` | `candidate_only` | `unresolved` | 1 |
| `double-pendulum-more-robust` | `double_pendulum_more_robust` | `wrong_controller_and_initial_set` | `unresolved` | 1 |
| `nav-standard` | `nav_standard` | `candidate_only` | `unresolved` | 2 |
| `nav-robust` | `nav_robust` | `candidate_only` | `unresolved` | 2 |
| `quad-reach` | `quad` | `dynamics_mismatch_unresolved` | `unresolved` | 2 |
| `single-pendulum-reach` | `single_pendulum` | `candidate_only` | `unresolved` | 1 |
| `tora-remain` | `tora_homogeneous` | `candidate_name_mapping_only` | `unresolved` | 1 |
| `tora-reach-sigmoid` | `tora_sigmoid` | `candidate_only` | `unresolved` | 1 |
| `tora-reach-tanh` | `tora_relu_tanh` | `candidate_name_mapping_only` | `unresolved` | 2 |
| `unicycle-reach` | `unicycle` | `candidate_only` | `unresolved` | 2 |

## 1. ACC — safe-distance (`acc-safe-distance`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：distance over time。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：distance over time；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：Historical native result required the VAR-tail correction; old completion/width cannot be promoted without the corrected latest contract.
- 未决：The report names v_rel as a controller input but does not define its sign convention; the input transform must be frozen from authoritative execution code.
- 未决：The safety property is a linear halfspace, not an axis-aligned box.

## 2. Airplane — continuous (`airplane-continuous`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 2 and 7。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 2 and 7；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The repository top-level table says continuous [0,20], but the report and instance specification agree on a 2-second continuous horizon. Method-specific integration settings remain unresolved.

## 3. Airplane — discrete (`airplane-discrete`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`discrete_execution_contract_v1`。
- 计划可视化：states 2 and 7。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 2 and 7；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The report defines forward Euler with delta_t=0.1 and k=0..20, but the pinned repository has no selected executable discrete transition or exact control-application ordering. A continuous ODE result is not a substitute.

## 4. Attitude Control — avoid (`attitude-control-avoid`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 1 and 2。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 1 and 2；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The legacy property verdict was not independently audited.
- 未决：Two distinct official ONNX candidates are present without a repository-level selection.
- 未决：The report says both that the unsafe set should be avoided and that the goal is to show the specification does not hold; freeze checker polarity before execution.

## 5. Balancing — reach (`balancing-reach`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 1 and 3。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 1 and 3；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：Balancing is the top-level name for the CartPole folder.
- 未决：The report writes a five-feature controller expression using sin/cos of the pole angle, while the repository dynamics comment and ONNX graph use four raw state inputs.
- 未决：The report uses the closed property interval [8,10], while the repository specification says t > 8.

## 6. Docking — constraint (`docking-constraint`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：state 1 over time。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：state 1 over time；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：No legacy four-way configuration is present.
- 未决：The instance file omits the 40-second horizon supplied by the report.
- 未决：The coupled nonlinear safety inequality cannot be represented as an axis-aligned property box.

## 7. Double Pendulum — less-robust (`double-pendulum-less-robust`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 3 and 4。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 3 and 4；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The old native run timed out; timeout is not a complete runtime.

## 8. Double Pendulum — more-robust (`double-pendulum-more-robust`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 3 and 4。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 3 and 4；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The legacy more-robust config uses the official less-robust controller SHA and the singleton corner {1.3}^4 instead of the official more-robust controller and full [1,1.3]^4 initial set; its time, widths, and verdict are not promotable.

## 9. NAV — standard (`nav-standard`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 1 and 2。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 1 and 2；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The old native run timed out; small-denominator width ratios require absolute widths.
- 未决：The report orders the physical state as [x,y,theta,nu], while the pinned repository dynamics evaluates x3*cos(x4) and x3*sin(x4), implying [x,y,nu,theta] and swapped control-output semantics. Resolve the controller input/output order before execution.

## 10. NAV — robust (`nav-robust`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 1 and 2。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 1 and 2；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：Legacy acceleration results are regression references only, not latest-version reruns.
- 未决：The report orders the physical state as [x,y,theta,nu], while the pinned repository dynamics evaluates x3*cos(x4) and x3*sin(x4), implying [x,y,nu,theta] and swapped control-output semantics. Resolve the controller input/output order before execution.

## 11. QUAD — reach (`quad-reach`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：state 3 over time。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：state 3 over time；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The 2026 report equations and the saved CROWN-Reach quad.cpp differ in x2 signs, x4 multiplication versus subtraction, and the x5 formula.
- 未决：The original native 1024-lane job ended by six-hour timeout after 600 complete steps; it has no T=5 time or width.

## 12. Single Pendulum — reach (`single-pendulum-reach`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：state 1 over time。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：state 1 over time；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The report and specification define two physical states, but the repository dynamics also returns dx(3)=1 without defining that clock state's initial value or controller/property role.

## 13. TORA — remain (`tora-remain`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 1 and 2; states 3 and 4。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 1 and 2; states 3 and 4；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The report places u=f(x)-10 at the controller boundary, while the repository dynamics subtracts 10 inside dx4. Freeze whether the plant function receives raw or post-processed controller output so the offset is applied exactly once.

## 14. TORA — reach-sigmoid (`tora-reach-sigmoid`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 1 and 2。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 1 and 2；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The report's grouped activation prose conflicts with the pinned sigmoid MAT metadata, and 'within 5 s' does not by itself freeze the reach-property checker semantics.

## 15. TORA — reach-tanh (`tora-reach-tanh`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 1 and 2。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 1 and 2；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：Confirm whether the saved legacy 'relu_tanh' controller is the 2026 reach-tanh controller.
- 未决：The report's grouped activation prose conflicts with the pinned ReLU/tanh MAT metadata, and 'within 5 s' does not by itself freeze the reach-property checker semantics.

## 16. Unicycle — reach (`unicycle-reach`)

### 模型、控制器、初始集合与性质

- 执行合同：**未冻结**；本节不得据此启动作业或填入成绩。
- 待解决字段配置：`full_execution_contract_v1`。
- 计划可视化：states 1 and 2。

### 完整配置、状态与复现入口

| 方法 | support / run | h / work / point / validation | cutoff / cap / SR | updates / NN | arithmetic | hardware / runtime | checker / early-stop | 命令 / cwd | source / binary identity | 结果记录 |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch/GPU | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Huan | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Xiangru | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |
| Flow* native | `unassessed` / `not_started` | `h=unresolved; work=unresolved; point=unresolved; validation=unresolved; semantics=—` | `cutoff=unresolved; cap=unresolved; SR=unresolved; semantics=—` | `updates=—; NN=unresolved; semantics=—` | `{"controller_domain":null,"dtype":null,"mode":null,"relaxation":null,"transport":null}` | `{"cpu_threads":null,"gpu":null,"hardware":null,"resource_limits":null,"timeout_s":null}` | `mode=unresolved; id=—; early-stop=—; semantics=—; certificate=—` | — / `—` | `{"binary":{"path":null,"sha256":null},"source":{"kind":null,"locator":null,"revision":null,"sha256":null}}` | `—` |

### 完整性、性质与结果资格

| 方法 | requested / validated | 完整时域 | accepted / rejected / NN | 性质 / 证书 | soundness / scope | formal / performance / ranking |
|---|---|---|---:|---|---|---|
| PyTorch/GPU | — | — | — | — | — | 无结果记录 |
| Huan | — | — | — | — | — | 无结果记录 |
| Xiangru | — | — | — | — | — | 无结果记录 |
| Flow* native | — | — | — | — | — | 无结果记录 |

### 时间

| 方法 | 冷启动 process (s) | steady n | process median/min/max (s) | driver / solver / validation / observer / plot median (s) | 排名资格 |
|---|---:|---:|---:|---:|---|
| — | — | — | — | — | 当前无合格的完整时域时间样本 |

### 绝对宽度与共同前缀

| 方法 | view | domain | 坐标 | lo | hi | union width | partition mean | partition max | 排名资格 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | 当前无完整宽度记录 |

### Flowpipe 图、失败与未决项

- 目标图：states 1 and 2；只接受结果记录中哈希绑定的图/轨迹。
- 当前无哈希绑定的图或轨迹记录。
- 未决：The latest GPU diagnostic stopped at extension loading before plant advance; frozen shared objects load directly, while Ninja is absent. Restore/qualify the loader without rewriting the failed record.
- 未决：The report includes w in [-1e-4,1e-4] in velocity_dot, while the repository dynamics omits w and its comment gives a malformed range. Disturbance temporal semantics and the shared reach checker remain unresolved.

## Huan QUAD 原因分析与 strict/parity 边界

原因、模式、移植技巧与历史证据见 [`docs/HUAN_QUAD_SPEED_AND_MODES.md`](HUAN_QUAD_SPEED_AND_MODES.md)。该文档不向本报告的 2026 结果表注入历史时间。

## 绘图功能与 MATLAB 示例

图层、tube/endpoint 语义与 MATLAB/替代渲染边界见 [`docs/flowpipe_plotting.md`](flowpipe_plotting.md) 和 [`docs/FLOWPIPE_PLOT_VALIDATION_20261001.md`](FLOWPIPE_PLOT_VALIDATION_20261001.md)。

## 统一数据包与发布

每个终态 cell 的 result record 是索引入口，必须绑定原始单次样本、失败记录、结果/日志/命令、轨迹与宽度 artifact。大张量不嵌入报告；其位置与 SHA 只从已验证记录读取。Markdown 通过最终门后，才可据同一数据生成 DOCX/PDF。

## 最终发布门

当前阻断：
- `experiments_paused`
- `unresolved_contracts=16`
- `matrix_not_terminal=not_started`
- `unassessed_support=64`
- `nonterminal_cells=64`
