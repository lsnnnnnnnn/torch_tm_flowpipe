# 常数代入合同

输入系数、h、domain 和余项端点均解释为实际保存的浮点数所代表的精确实数。对每个剩余单项式 beta，计算独立系数区间 [q_beta]，包含 sum_k a_(beta,k) h^k。保留原点路径得到的 qhat，返回 Qhat + (R + E)，其中 E 包含 sum_beta ([q_beta] - qhat_beta) u^beta 在实际剩余 domain 上的像。

稀疏和稠密保证接口调用同一个 `endpoint_substitution.enclose_constant_substitution`。时间幂由基本区间乘法逐次生成；每次乘法、加法和误差相减均向外舍入。不以浮点 pow 或 scatter_add 的结果作为精确系数。每个固定时间幂的剩余单项式互异，因此系数区间赋值没有合并；不同时间幂之间逐次向外相加。

误差多项式使用真实 domain，各变量的幂同样由逐次区间乘法包住。它是逐点函数包含证明：对每个合法 u，真实的每个系数差属于对应区间，各单项式乘积和总和属于 E。不依赖轨迹采样，不要求稀疏与稠密结果逐位相同。独立测试以 Fraction 重算精确系数和小型范围；生产辅助函数不使用 Fraction。

精确的零、乘一与加零保留恒等式；非零 subnormal 不截零。h 必须位于原 domain，非有限输入或中间 enclosure 溢出明确失败。保证接口不默默外推余项。删去已经代入的变量是单项式一一对应，不再次合并；cutoff 只迁移保留点多项式被删掉的项，误差已位于普通余项中。

稠密 endpoint 保留原 ledger，并补齐旧 ledger 求和与实际余项之间缺少的向外 padding；新增代入误差只放在 `endpoint_substitution_roundoff`。返回 remainder 直接取新 ledger 的合计。转换存储形式复制完整 remainder；类别在稀疏转换后可合为 initial_remainder，数学载荷不丢失。

账本按类别重新求和可能与实际余项的运算顺序不同。`covering_total` 每次补入缺少的向外 padding 后再次检查总和包含实际输入余项之和；若有限次对账仍不能建立包含则拒绝。该预算只用于拒绝无法表示的对账，不替代包含断言，不改变 ODE 验证预算。端点之后的 dense cutoff 也保持合计与返回余项完全一致。新增类别不进入普通验证器的固定完整类别表，避免对未执行端点代入的验证步骤制造一个额外零项。

发布端点和内部端点分别从原 segment 计算。发布结果的余项被 normal 插入消费；内部结果不会再叠加到发布结果上。normal 左映射保留 segment，下一步实际输入使用新的 center/scales 和右映射。accepted-boundary SR 将端点本步余项经非线性分支记入 current owner，已传播历史由独立 linear queue 分支消费。具体两步、回滚及容量边界证据另存，完成前不宣称使用链已经验证。

直接调用图还包括 G1/G2 source lift 与 S1 structured typed-source 三类重建路径。它们刻意去掉端点普通余项，改用验证器的完整分解。`_endpoint_remainder_decomposition` 为这些消费者派生端点分解：原验证分解保留，本次 E 加入既有 roundoff_safeguard，端点 cutoff 移出项加入 cutoff，并检查合计包含完整发布余项。G1/G2 用它生成下一步 affine source，S1 用它生成 ordinary/structured 边界状态；不再另加完整端点 R，故不会重复加入原验证余项或历史。S1 发布整段的账本也保守保留这笔普通误差；这不是改变 ODE 求解公式。每次读取派生分解无状态修改。

成本包括独立区间幂、系数组合和误差范围计算，计入求解时间。没有实现提速；正式成本待干净数值提交上的完整运行测量。


兼容性与受影响预期：完整回归发现旧 CUDA smoke 的状态系数与时间 domain 位于不同设备。辅助函数现在将 domain 无损移动到系数设备，并提升混合 binary32/binary64 输入，避免向较窄系数数组赋值而丢失 enclosure。Python float 的实际 binary64 值在合法域检查前不被缩窄。冻结 CPU binary64 输入只经过恒等转换，幂、乘法、合并及误差范围的运算顺序均不变。这是现有入口的设备/类型兼容修正，没有 CUDA 后端开发。

只更新两处已受影响的旧预期：C2 两步 reset/right hash，以及 C3 关闭归一化实验控制的严格域门槛首次失败位置（11 → 3）。先在干净 scientific SHA `196a50e9131336d68df07ad0af353deca0092d19` 对前者两步、后者三个接受端点用独立 Fraction 检查完整函数包含，再记录 `raw_minimal/affected_golden_audit.json`，最后更新预期。拒绝条件、回滚断言、两个原始端点包含测试均未改。固定长跑和 native adaptive 来自该 SHA；交付兼容修正版本与其身份分开，并对长跑保存的每个端点复算逐位一致性。
