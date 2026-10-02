# 旧作者 QUAD P3：加权细化一轮 40 步限长变体

日期：2026-10-03（上海）。本目录是独立的新 ID 诊断，不是旧 1,000 步实验的重启，也不是 2026 论文方程 QUAD 的结果。

## 为什么需要新求解

[既存两轮门禁](../weighted_round_gate_v1/README.md)保存了各步第一轮和第二轮的余项，因此能够说明第二轮并非逐位无效；但第 2 步以后的原运行均从**两轮后的状态**继续。只读重算中间余项无法判定一轮变体后续 40 步的接受、控制输入或宽度。为此另立服务器目录 `p3_quad_old_weighted_one_round40_nohash_20261003_001/run_001`，只运行一次。

## 冻结合同与唯一数值改动

使用既存 [无内容摘要、无 JIT 的 40 步 runner](../observer_pair_v1/runner.py) 与原 `quad_40.yaml`，完整 1,024 初盒、旧作者 `x2/x4/x5` 方程、`h=0.005`、两个控制周期、P3 工作阶/P2 点值阶/P4 验证阶、strict 边界、完整 host K20 SR、trig 复用及逐步 observer 均保留。生成的 `data/config.yaml` 与源 YAML 在解析字段上仅有原 runner 已规定的 `ode_order: 2→3`、`sr_queue: absent→1000` 两项差异；动力学表达式逐项相同。服务器原 runner 和源 YAML 分别与本地保存副本直接逐字节相等。服务器控制器与本仓库既存 `quad_controller_3_64_torch.onnx` 直接逐字节相等；没有计算内容摘要。

[新探针](one_round_probe.py)在既有 runner 准备完成后，只把 `weighted_validation.refine_accepted` 的 `rounds=2` 改为 `rounds=1`。它在首次数值拒绝时立即停止并保存拒绝盒编号；没有放宽自映射条件、改变其他求解参数或修改原 runner/旧 run。所有 40 步的一轮调用均有 1,024/1,024 eligible、attempted、recovered，且每盒只进行一次加权 map 评价。启动前确认 GPU 3 空闲、新 ID 未占用；实际使用 GPU 3、CPU 14–17、120 秒外层限时。结束后 GPU 3 释放，无新进程残留。

## 原始结果与直接读回

| 指标 | 一轮新变体 | 两轮保存参考 |
| --- | ---: | ---: |
| 数值时域 | 40/40 步、`t=0.2 s` | 40/40 步、`t=0.2 s` |
| 已接受盒步 | 40,960/40,960 | 40,960/40,960 |
| 控制器调用 | 2 | 2 |
| 终步状态 | 1,024 盒均为原状态码 0 | 相同 |
| 终步每盒 12 坐标 tube 宽度总和 | `2697.8759514297453` | `2697.875935884544` |
| 终步每盒 tube 端点变化 | 22,528 个；新变体均向外扩大 | 参考 |
| 保存并集的最大端点绝对差 | `3.7443330103137384e-09` | 参考 |

外层监控 `completed/exit0`、未超时，wall `45.207353 s`；内层 `completed`、driver 时间 `38.407367 s`，`broken=0`。40 行保存的 tube/endpoint 并集均与两轮参考不同；[独立保存读回](run_001/SAVED_AUDIT.json)检查每行有限、有序、全部接受、SR 长度与参考相同，并检查最终 `1,024×12×2` tube/endpoint 和状态数组。终步 tube 和 endpoint 的 22,528 个变化端点全部向外扩大，向内变化 0；终步两个数组的最大端点差均为 `3.836081063912644e-09`。两轮参考的第二轮确实改变结果，因此一轮变体**不是逐位等价优化**。

原 runner 的目标检查在缩短的 `t=0.2 s` 打印 `FALSIFIED`，不得解释为原 `T=5 s` 终点性质失败。一轮与两轮的时间来自不同日期的单次进程，且两轮 round-gate run 额外插入每轮事件、同步及数组复制；现有 `45.207353 s` 不能直接当成净节省，也不能推出 1,000 步速度、收缩接受或最终宽度。当前只是一个数值更宽、40 步仍全接受的单因素算法变体；任何全程使用仍需另做完整数值与性质资格，且现有 `end_to_end_strict_certificate=false` 未改变。

原始 [外层 START](run_001/START.json)、[外层 RESULT](run_001/RESULT.json)、[探针 RESULT](run_001/ONE_ROUND_RESULT.json)、[求解器 RESULT](run_001/data/RESULT.json)、[实际配置](run_001/data/config.yaml)、[逐步观察](run_001/data/observations.jsonl)、[stdout](run_001/stdout.log)、[stderr](run_001/stderr.log)、最终数组和 [独立审计脚本](audit_saved.py)均在本目录。关键原始 JSON、观察和两份最终区间数组已与服务器文件直接逐字节核对；未做任何哈希校验。
