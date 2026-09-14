# Resident normal-composition 数值合同

## 范围与非目标

本合同只覆盖二状态、binary64、总次数 1–6、一个或两个输出的接受边界 normal composition。它对应 `insert_ctrunc_normal_dependency_preserving` 的一次完整调用，不覆盖周围的 dense Picard、验证 replay、历史队列更新或整套求解器。

默认路径不变。只有显式路线 Gr 在 task worker 的 context 中安装 resident dispatcher；Gp 和其他路线不会进入该块。不支持的形状返回 `NotImplemented` 走既有 CPU 路径，数值无效则失败关闭。这个 opt-in 原型不是 full GPU engine。

## 输入与输出边界

对每个输出分量，外层 Taylor model 写成

```text
F(u0,u1) = P_F(u0,u1) + I_F
```

两个内层模型写成

```text
G_j(v0,v1) = P_j(v0,v1) + I_j,  j=0,1.
```

输入还包括矩形定义域 D、目标总次数 p、cutoff，以及外/内模型的 truncation split。输入系数同时携带一个 RN 点值 `c` 与一个包含真实系数表达式的区间 `[c_lo,c_hi]`。普通在线输入初始为点区间；独立测试也覆盖真正的区间系数和零点值、非零误差区间。

输出是稀疏 RN 点多项式 `P_out` 和 remainder `I_out`，满足本块有限算术图的真实复合值包含于 `P_out(D)+I_out`。同时返回每个保留系数的误差区间，供独立核验；在线后续步骤消费重建后的完整 Taylor model，而不是加载保存答案。

## 冻结的基与计算图

canonical basis 按总次数升序、同次数内按 u0 指数升序：

```text
(0,0), (0,1), (1,0), (0,2), (1,1), (2,0), ...
```

完整 basis 序列、两个变量的角色、输出数、阶数、Horner 次序、截断/cutoff/范围规则、舍入模式和 cutoff 的 binary64 值都进入 structure key/fingerprint。缓存只保存不可变结构；每次请求的 point/lo/hi/remainder/domain/split 都复制并重新上传，active support 由当前系数区间在 kernel 内重建。

复合按 u0 外层、u1 内层的递归 Horner 图执行。每个 `(request, output component)` 是独立 CUDA 线程和独立 `LocalTM`/尾项局部数组；没有跨任务数值归并或共享 remainder。

## 点系数与区间系数

点系数遵循代码中固定循环顺序的 binary64 round-to-nearest 运算：

```text
point[k] = RN(point[k] + RN(left[i] * right[j]))
```

编译选项是：

```text
--std=c++14 --gpu-architecture=compute_70
--fmad=false --ftz=false --prec-div=true --prec-sqrt=true
```

因此乘和加不会被融合，subnormal 不被 flush-to-zero。点图本身不声称实数精确；它的误差由平行的系数区间图包含。

区间加法端点使用 `__dadd_rd/__dadd_ru`，区间乘法计算四个端点组合，各自用 `__dmul_rd/__dmul_ru` 后取最小/最大。最终系数差使用 `__dsub_rd/__dsub_ru`。输入必须有限，`lo <= point <= hi` 且 `lo <= hi`；否则线程返回非零状态且不发布 Taylor model。启动自检用 normal、subnormal、正负零、大数消去和下溢乘法逐项以 Fraction 检查 directed primitive。

## 一次 Taylor-model 乘法的 remainder 归属

设两个临时模型为 `A=P_A+I_A`、`B=P_B+I_B`。每次 Horner 乘法都独立支付：

```text
R_mul = range(P_A) * I_B
      + range(P_B) * I_A
      + I_A * I_B
      + range(degree > p 的系数区间尾项).
```

所有加乘均为向外区间运算。保留次数 `<=p` 的每个系数同时更新 RN 点值和 directed 系数区间；次数 `p+1..2p` 的系数只进入尾项数组。普通左右多项式范围使用 natural interval range；截断尾项使用两侧 split 的最大值，在 D 上做 `n×n` 分割后取 hull。split 上限为 16。

外层 `I_F` 在整个 Horner 复合结束后加到结果 remainder，恰好一次。诊断分别记录 truncation、cutoff、`P_A I_B`、`P_B I_A`、`I_A I_B`、外层 remainder 和保留系数误差的宽度，避免把来源混成无法审计的一笔。

## cutoff 规则

每个乘法和必要的 Horner 加法阶段后检查点系数。条件是：

```text
abs(point_coefficient) <= cutoff
```

满足条件时点系数及其系数区间都清零；被删系数区间组成的多项式在 D 上以未分割 natural range 求值，并把该范围加入 remainder。等于门槛也删除。独立矩阵覆盖门槛值、相邻的上一个/下一个 binary64 值、极小数、负数和严重相消。

## 保留系数舍入的最终付款

平行系数区间已经沿完整 Horner 图传播每次乘加的实数结果，但最终输出只保存 RN 点系数。对每个仍保留的系数 k，构造

```text
E_k = [sub_rd(c_lo[k], c_point[k]),
       sub_ru(c_hi[k], c_point[k])].
```

然后对误差多项式 `Σ E_k v^k` 在 D 上按结果 split 求范围，并只在输出边界加入 remainder 一次。这既避免漏付，也避免在每层把相同误差重复物化。

具体见证是 binary64 `1.1 × 1.1`：实数乘积与 RN 点积之差非零，旧 CPU block 的近零 remainder 不包含该差；resident block 的对应误差区间为 `[-0, 2^-52]` 并包含它。修复因此会合法改变后续完整 binary64 状态，不能要求新旧逐位一致。

## 范围求值

`range_polynomial` 对每个单项式分别以 directed interval power/multiply 求值，再 directed add。若 split 为 n>1，则把 D 两个维度各等分 n 份，对 n² 个盒分别求 natural interval range 后取 hull；否则在整个 D 上求值。分点计算遵循冻结的 binary64 图，独立 Fraction oracle复现该分点图的点值，再用有理数区间核验 CUDA 输出包含。

该范围方法是本有限块的合同，不声称最紧，也不把“小于 1.10 宽度比”当作安全证明。

## 设备驻留、传输和回执

一个同构请求组的执行顺序为：

1. 在主机把每个任务的 outer/inner point、系数上下界、remainder、domain 打成一个连续 numeric payload，split 单独打包；
2. 两次 H2D（numeric、metadata）；
3. 一次 `resident_normal_compose` kernel；
4. kernel 内的所有 Horner 中间系数、尾项和 remainder 被后续操作直接使用，不回主机；
5. 两次 D2H（numeric output、receipt）和一次同步；
6. 核对 magic 与 `batch*outputs` 完成数，再重建 caller-owned CPU Taylor models。

正式证据分别记录 packing、CUDA-event H2D、CUDA-event kernel、传输/同步 host span、D2H 后检查与对象重建、字节数、copy 数、kernel 数、同步数和逻辑设备局部中间字节。`logical_device_local_intermediate_bytes` 是按局部数组形状计算的逻辑量，不冒充 allocator 实测流量。

## 请求身份、取消与原子提交

请求 ID 包含长度编码的 run/task，以及 epoch、accepted generation、attempt、counter。提交前复制所有数值；服务端结果不会作为可变 caller storage 暴露。resident graph 与普通 range packet 永不混组，不同 structure key 的 resident 请求也不混组。

消费时同时核对当前 pending 对象、完整身份和返回 request ID。取消会使未提交 attempt 失效；已在设备执行但随后到达的结果计为 stale 并丢弃。checkpoint 只深拷贝最后接受状态，不保存 worker、Future、指针或 device buffer。只有整个步骤验证成功并且没有 pending 时，才在与取消相同的锁下更新完整 `(current, normal_state)`；失败、拒绝或取消不部分提交。

## 支持范围与失败策略

resident request factory 只接受：

- 两个归一化变量；
- order 1–6；
- 一个或两个输出；
- binary64 Taylor coefficients/remainder；
- 所有项均在完整 order-p basis 内；
- 每个 split 为无/1 或 2–16；
- 有限、有序且包含 point 的系数/余项/定义域区间。

维数、阶数或结构不支持时，显式计数后回到既有 CPU composition；错 fingerprint、同组混图不会静默执行。非有限或无效区间在设备状态中失败，不做隐藏 CPU 重算。正式 Gp/Gr 中结构 fallback 和硬件 fallback 都为零。

## 独立证据与边界

独立 `fractions.Fraction` oracle 重建同一声明的离散 Horner/cutoff/split 图，但不调用 CUDA interval helper、旧稀疏 TM 乘法或历史答案。order 4/6、64 lanes 上完成：

- 2752 个 RN 点系数图相等检查；
- 2752 个真实系数表达式被返回误差区间包含的检查；
- 128 个最终 remainder 包含检查；
- B1/B2/B8/B32、正序/逆序/切批的逐任务 bitwise 检查；
- 点/区间系数、异构 support/split、相消、cutoff 边界、subnormal、负数、奇偶幂、空 support、错程序和非有限输入；
- 取消后迟到结果、执行回执和计时分母检查。

在线比较再验证当前输出真实推进下一步，且 Gp/Gr 的接受决定、support 和 ledger 类别一致；宽度警戒为零。以上是有限实现合同证据，不是对所有次数、变量数、硬件、编译器或完整求解器的形式化证明。

## 版本责任

实际科学运行使用提交 `8953b5ea24e69d11ac5c2890a4cdb663305e9e79`。CUDA 源 SHA256 为 `7f163c0b9c662dbd19ef15f4f58745278c239c4625c198696111f00583f77b09`，正式运行记录的 PTX SHA256 为 `3f3d9b92ebb814d991be2b47724fcdb0168685dad919ee4f8db79e63f82507e5`，NVRTC 12.1，设备为 Tesla V100-SXM2-16GB（compute capability 7.0）。

代码没有复制 Flow* 数值实现；Flow* 只作为单独 GPL-3.0 比较依赖，由有限测试驱动链接。数学正确性责任仍由本项目的 CUDA 源、独立 oracle、在线状态证据与明确的非证明边界承担。
