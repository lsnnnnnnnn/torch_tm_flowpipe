本轮只研究稀疏多项式取范围的执行结构。此文件当前记录隔离原型合同；是否进入生产路径必须先由 `implementation_decision.json` 的真实局部时间与窗口份额决定。

输入是实际 CPU binary64 系数区间 `[B, output, terms]`、按原顺序排列的单项式支持，以及实际 domain 的两个端点 `[B, variables]`。同一支持中的每个位置都是一次原有项运算，包括显式零项；稀疏入口的支持由原 Polynomial 直接提供，不凭阈值重新筛选。不同输出或任务使用自己的系数与 domain，B2 是这个局部取范围接口的能力。

结构计划只保存不可变的指数、变量顺序、项索引和 normal parity 规则。LRU 只按这些结构值索引，不保存 Tensor、domain、系数、h、余项或 Python 对象 id。每次调用重新堆叠实际输入并检查 dtype、device、形状和区间合法性。原型的数值结果与变量幂只存在于这次调用；输入原地改变后的下一次调用必须重算。

变量幂保留原 `Interval.pow_int` 的标量调用路线，分别处理每个 batch/domain，而后只在独立项、输出、任务之间执行并行张量乘法。奇偶幂分支、四个乘法候选的排列、min/max 与 nextafter 位置不变。变量乘法仍按原顺序；幂为零的变量不增加一次乘法。normal evaluator 仍先处理时间、再乘原有 state factor、再处理其他变量。提供显式 step power table 的旧接口须保留原路径，不能忽略表中数值。

最终按支持集原顺序逐项相加，每次相加都单独向外舍入并检查区间。不能用 sum、cumsum、并行 reduction 或代数重排替换这条求和链。每一项的区间与最终区间均由逐位对照和独立 Fraction 自然区间 oracle 检查；更窄的输出不构成验收依据。NaN/非法区间与原路径一样拒绝，Inf 和溢出按原 interval 算术的实际结果或异常处理。空支持返回真零。

写入仅发生在本次调用分配的 term 张量中。原系数与 domain 不被修改，返回结果由新的 Interval 持有自己的存储。不存在跨步可变共享数值、当前 remainder 缓存或预先缓存总区间。端点误差 E、cutoff 支付、普通余项类别、当前 owner 与已传播 history 的归属、accepted-only commit、拒绝回滚及队列清空均由原流程执行。

prepared replay 在本轮 baseline 与 candidate 中始终显式开启；它的准入、收紧轮数和缓存内容保持原样。拟议边界开关与 prepared 开关分开，默认关闭。现有混合 dtype、非 CPU 和非标量稀疏入口回退到原 evaluator；局部张量接口自身只声明 CPU binary64。未实现完整 batch solver，也不声明 GPU 吞吐。

验证边界仅限已检查操作和本轮实际运行，不能外推为整个 solver 的形式化证明。生产接入、连续传递、恢复/失败检查及全程对照完成后，交付报告与原始结果共同确定最终状态。
