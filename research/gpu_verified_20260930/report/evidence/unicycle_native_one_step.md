# Unicycle：实际原生 CPU 一步诊断（2026-09-29）

独立原生入口已编译并完成一步，五个真实 TM/SR 边界均已保存。新观察器得到的 16 个 tube/endpoint 边界值、8 个宽度，与历史原生 first-step CSV 的 binary64 值逐位相同。它补齐了原生内部状态证据；尚未运行相应三个 GPU 边界诊断，不能声称已经解释宽度差的原因。

另一个必须分开记录的发现：原生旧初始表示在 x2 下端点少覆盖 **1 ULP**。本次保持原库和原初盒不变，因此也保留这个问题。诊断完成不等于请求初盒的严格可达性证明。

## 实际执行及身份

- CPU18，`CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。未启动 Torch、GPU、NN server，也没有重新计算 CROWN。
- 唯一 runner PID 1532213；归档时 `/proc/1532213` 已不存在。上传启动 SSH session 23761、终态归档 session 90773 均返回 0。没有重复启动。
- 原 `cpu_v2/prepared` 十个文件原样上传；`CPU_PREPARATION.json` 的九个成员 SHA 全部通过。实际编译命令只把旧 recipe 的 affinity 10 改为 18，另存 `evidence/run/ACTUAL_COMMAND.json`。
- 原库 `libflowstar.a` SHA `274e9bd11c9de21e95d72e935d9e45fd4fe9479ebb986751dd7f90522e416e3e` 保持不变。实际 toolbox headers、matched_reach、arch_ranges、编译器及链接动态库身份分别存于 `SOURCE_AND_DEPENDENCIES.json` / `LINKED_DYNAMIC_LIBRARIES.json`。
- C++ 编译 rc=0，2.921146846 s；原生一步 rc=0，0.114133202 s。后者含启动和详细诊断输出，仅作作业记录，不用于全程速度比较。原库 TaylorModel.h:900 的编译器提示保留于 `compile.stderr`，没有修改源库。
- 二进制 SHA `ca6e8530c3ad5d75112c597edf87dd4ba2731b6b5edd16b2116036d6d35299b1`；运行时 120 s 超时，必要时另给 5 s TERM 清理。实际正常退出，无清理触发。

配置保持 B=1、8 总变量、h=.02、P2、cutoff=1e-6、余项估计 ±.01、SR=1000。冻结历史首条 RPC 原始 response，再走原 C++ `asFloat()` 和控制注入算术。实际 NN 输入域 8 个端点逐位等于历史 RPC 输入域，四维 containment 全 true。没有把冻结 response 用于一个未经核实的新输入域。

## 实际第一步范围复现

`ranges.bin` 只有一条 152 B 记录：lane=0、step=1、h=.02。直接按 `<QQd` 加 16 个 `<d` 解码，与本地历史 `width_trajectory_unicycle_matched.csv.gz`（SHA `faf582bcbd28286aeee8ef413268b7de4f28f21564a71618df23bbc2cb864751`）核对。三个历史 arm 行中重复存储的原生 bounds/width 共 72 项检查全部逐位相等；独立数值为 16 个端点、8 个宽度。

以下 Huan/Xiangru/我方列是**历史同一步观测值**，本次没有重新运行它们。B=1，单格宽度等于该观测的 union 宽度。

| 坐标 | 本次原生 endpoint 宽度 | 历史 Huan / Xiangru | 历史我方 |
|---|---:|---:|---:|
| x1 | 0.05039746018477409 | 0.050458377104931174 | 0.05045837710493828 |
| x2 | 0.05036163410529593 | 0.05040134838377153 | 0.05040134838377508 |
| x3 | 0.012173826448941938 | 0.012173826448944158 | 0.012173826448945935 |
| x4 | 0.01026905251303356 | 0.010269052513034227 | 0.01026905251303556 |

完整 tube/endpoint 下上界、宽度和历史比值在 `WIDTHS.csv`。目前明显的首步 endpoint 差异仍集中于 x1、x2；x3、x4 仅在末位有差异。此次原生复现使后续内部状态归因有了可靠参照，但不能把观察到的初始化 ULP 缺口直接当作这些宽差的解释。

## 五个实际边界

| 边界 | pre 项数 | SR Phi / J 数 | 局部时间域 |
|---|---:|---:|---|
| initial | 9 | 0 / 0 | [0,0] |
| before_injection | 9 | 0 / 0 | [0,0] |
| after_injection | 19 | 0 / 0 | [0,0] |
| after_advance | 28 | 1 / 1 | [0,.02] |
| after_endpoint | 22 | 1 / 1 | [0,.02] |

本地独立读取 446 个 hex 区间对，全部有限且有序；137 个实际 pre/tmv 系数导出的 RNDD/RNDU 对全部 singleton。initial 与 before_injection 除阶段名称外完全相同；注入只改原生 u1/u2 两行，其他 pre、tmv、domain、SR 不变。endpoint 操作保留 domain/tmv/SR，仅改变 pre；时间域仍是原始 [0,.02]，未擅自改为 [0,0]。

原注入的两个控制常数项指数数组是 8 个全零，其余项为 9 维。这来自原 `TaylorModelVec(c_vector,numVars=8)` 表示，原状态原样保留。本地审计最初假定所有项均为9维而拒绝，具体修正记于 `AUDIT_FIRST_PASS_SCHEMA_NOTE.json`。

新独立 `development/unicycle_first_gap_20260929/compare_injected_states_v2.py` 只将这类 **8 个全零的常数项**在比较副本中规范化为9零；8维非零项仍拒绝，旧 comparison 和 prepared SHA 不变。对实际五边界共十个 TM 组已通过解析、原值不变及非零项拒绝检查；`COMPARISON_V2_ACTUAL_SCHEMA_CHECK.json` 明确尚无 GPU 实际比较。v2 SHA `bd00a911706e2dedf78dc2eb7384b2d87e8e850d790c6ddb5fc8253ffc9c0e7f`。

## 初盒覆盖缺口：实际系数的精确检查

initial 的 tmv 是严格单位映射、R=0、8个局部空间域均[-1,1]，SR为空；全部 affine 系数是实际 singleton。以 Fraction 精确计算每维中心 ± 半径并比较原 `unicycle_boxes.json` 的 binary64 端点：7维覆盖，x2失败。

- 请求 x2 下界：-4.5。
- 实际 affine 下界：`-5066549580791807/1125899906842624` = -4.499999999999999 = `-0x1.1ffffffffffffp+2`。
- 缺口：`1/1125899906842624` = 8.881784197001252e-16，恰1 ULP。
- x2 上界保持请求值 -4.45。原 RPC 输入也使用这一收窄后的旧原生初始下界，所以“RPC域相同/包含”并不能补回请求初盒覆盖。

这是实际旧初始化的证据，和此前 QUAD 旧 `Flowpipe(box)` 单侧半径机制一致。此次没有套用 QUAD 修复、替换库或扩大初盒；原始历史比较仍可复现，但其严格初盒资格应为 false。后续若做覆盖修复，应作为独立条件并保留本次历史复现结果。

## 归档与边界

本地归档 `native_v1_evidence.tar.gz` SHA `661a5d0aa2bc824427171d5bf0274a5ce08f120c8c13eca49b3dc9c6df6bc834`。manifest 27 个成员全部逐 SHA/size 核对，`LOCAL_VERIFICATION.json` SHA `bc1c2fa1d0633ee463cbec67b9e768684a264a98b230568a63cc0bfad4c7b18a`。

- 实际 ranges SHA `adfde690a1a6de083bd07c15f34840bffa97453e25d010e04ee3b8d62491035d`。
- 实际 states SHA `f91a95b8d227231f4d2b240310802fd65fd1883b9c924c347b5845db4aab694f`。
- 本地独立审计 `ACTUAL_REVIEW.json` SHA `cc9a838d630577e9f63b28421189163128c05b0f1cc5f46d5a088785ad10b78b`；再现脚本 `audit_native_cpu_v1.py` 只读已有文件，不调用求解器。

本次完成的是固定历史控制输出下的原生一步诊断。三个 GPU 的同注入状态尚未实证；没有重新计算 NN、没有观测积分器内部 predictor/refinement、没有完成500步、没有端到端严格证书，也没有全程速度结论。下一步最有价值的是在根任务允许的 GPU 窗口，用已有三个独立 arm 保存相同边界，先区分初始化、注入和 advance 的首个实际差异。
