# ARCH-COMP 2026 Balancing/CartPole 执行门槛

2026-10-02；本门槛首版只读来源与已有记录，后续具名 raw4 四方法的新尝试见文末。此门槛把论文五特征控制器和固定仓库四原态控制器分别命名；两者不能共用一个未注明语义的四方成绩格。

## 来源与已确定的物理合同

- [2026 AINNCS 报告 §3.12，印刷页 99–100](https://easychair.org/publications/paper/GsKW/download) 写四个物理态 `(x1,x2,x3,x4)`，控制周期 `0.02 s`，完整初盒 `[-0.1,0.1]×[-0.05,0.05]×[-0.1,0.1]×[-0.05,0.05]`，终止时刻 `10 s`。固定官方顶层 `README.md` 把表名 `Balancing/reach` 映射到 `benchmarks/CartPole` 目录；该目录的原始文件为 `specifications.txt`、`dynamics.m`、`model.onnx`、`README.md`。固定来源的既有逐项审计见 [Docking/Balancing 来源合同](ARCHCOMP26_DOCKING_BALANCING_SOURCE_CONTRACT_20261001.md)。
- 固定 `benchmarks/CartPole/dynamics.m` 的四条方程为 `x1'=x2`、`x2'=2f`、`x3'=x4`、`x4'=(0.08·0.41·(9.8 sin(x3)−2f cos(x3))−0.0021x4)/0.0105`。在样本保持语义下，500 个控制周期覆盖 `[0,10] s`；`0.02 s` 是控制更新间隔，数值积分内步长须按方法另记。
- 报告式 (17)、(19)写 `f(x1,x2,sin(x3),cos(x3),x4)`，共**五个特征**。固定 `dynamics.m` 的注释则明确控制器读取原始 `(x1,x2,x3,x4)`，输出 `f` 直接进入方程。固定 `model.onnx` 的既有来源审计报告输入 `[1,4]`、输出 `[1,1]`；本轮另对服务器已有 CartPole 模型副本只读解析，图为 `Gemm,Tanh,Gemm,Tanh,Gemm,Tanh`，输入 `[1,4]`、输出 `[1,1]`。图中没有 `sin/cos` 特征扩展。服务器文本副本位于 `/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/engine_accelerated/benchmarks/CartPole/official/{dynamics.m,specifications.txt,README_official.md}`；模型副本位于同级 `model.onnx`，副本结构检查不代替固定官方模型身份的既有审计。
- 报告页 100 的性质是 **所有** `t∈[8,10] s` 均有 `x1,x3,x4∈[-0.001,0.001]`。固定 `specifications.txt` 写 `t>8.0 s`，在给定时域内是 `(8,10]`。这是来源文字边界差异，应在 profile 和 checker 中明确。若物理轨迹在控制更新时连续且目标盒闭合，则两个全连续时间命题在数学上等价：右侧极限保证 `t=8` 也在闭盒内；这一事实不允许把只检查 `t=10` 的旧目标 checker 当成时间窗 checker。

## 两个可区分的 profile

| 名称 | 控制器输入与模型 | 性质窗口 | 当前门槛 |
| --- | --- | --- | --- |
| `balancing-paper-feature5` | `f(x1,x2,sin(x3),cos(x3),x4)`；需对应五输入控制器，或作者给出的权威五特征到固定四输入 ONNX 的确切映射 | `[8,10]` 全连续时间 | **来源阻塞**：现有固定 ONNX 接受四输入，未见五输入权重或权威映射；不能猜角度恢复、删特征或把公式视为笔误。 |
| `balancing-fixed-repo-raw4` | 固定 `model.onnx`；每周期起点将原始 `(x1,x2,x3,x4)` 按此顺序送入，单个输出 `f` 直接用于上述 ODE 并保持 `0.02 s` | 按固定规格 `(8,10]` 全连续时间 | **数学合同可冻结并开始实现**：来源已给定四输入、完整初盒、方程、500 周期、性质。它须单独列为固定仓库 profile，不充当论文五特征主合同。 |

`balancing-paper-feature5` 具体缺的是可读取的五输入控制器文件及输入顺序、输出尺度/后处理，**或**作者明确说明如何用五特征调用四输入模型的执行源码/映射。取得其一后，四方法才能针对同一个函数做模型加载和首周期输入/输出预检。论文时间窗本身已明确，应取 `[8,10]`，无需从固定规格猜左端点。

`balancing-fixed-repo-raw4` 不需要再推断特征映射；可以编写四个隔离入口和一个共享覆盖 ledger，从同一完整初盒开始。开跑前仍须完成工程预检：四方法各确认固定模型加载与原态顺序、同一周期起点更新/周期内保持、500 周期覆盖；保存每段连续 flowpipe；性质 checker 仅对指定时间窗中的三态逐段判定，明确 `t=8` 的处理；记录接受/拒绝、早停、内步长和资源预算。若先做单盒/短前缀，只能标为预检，不能填写完整 10 秒成绩。性质判定必须区分 `VERIFIED`、有可信见证的 `FALSIFIED` 与 `UNKNOWN`。

## 既有证据的边界

当前 [64 格无摘要工作矩阵](evidence/archcomp26_nohash_work_matrix_20261001.md) 中 `balancing-reach` 四格均为未尝试。[2026-09-23 CartPole 历史记录](progress_review_20260923/REPORT.md) 是更小初盒、`50×0.02=1 s`、内部 `200` 积分步的另一合同；native 只保存 141/200 步，五个 GPU 模式接受 156/200 步并在第 157 步拒绝。历史 [配置](../research/gpu_verified_20260930/report/configs/arch_cartpole_official_f32.yaml) 将盒约束放在 `constraints_target`，只在终点使用。这些数据既没有完整 2026 初盒，也没有 500 周期及 `[8,10]`/`(8,10]` 全时性质，不能迁入上述任一四方结果格。

## 2026-10-02 后续尝试

上段“未尝试”是本执行门写成时的旧截点；当前工作矩阵须按最新 attempt 索引读取。具名 raw4 profile 的 Huan、当前 P3、Xiangru 分别在第 99、87、99 个 ODE 小步首拒，均未进入 8–10 秒性质窗。Xiangru 的新[原始收据与独立扫描](evidence/results/archcomp26_20261001/balancing_fixed_raw4_xiangru_20261002/SUMMARY.md)另记录 189 处保存 endpoint 超出同小步 tube，最大约 `2.49e-14`；不由此推断数值拒绝成因。[原生独立入口及审计](evidence/results/archcomp26_20261001/native_balancing_raw4_20261002/SUMMARY.md)保存首个解析失败、修正后一期 4/4 接受及 500 期请求在第 84 小步 `UNCOMPLETED_SAFE` 首停的全部原始记录。四方法 raw4 均无 10 秒完整结果；论文五特征控制器文件或权威四输入映射仍缺，不能把历史小初盒结果代入。
